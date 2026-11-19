"""
LLM Sentinel Scan Runner Engine.

Loads all attack suites, fans each probe through the target adapter
with configurable retries and exponential back-off, then persists every
result (success or failure) to SQLite via ResultStore.

Usage
-----
    from llm_sentinel.adapters import get_adapter
    from llm_sentinel.runner import ScanEngine

    adapter = get_adapter("mock")
    engine  = ScanEngine(adapter, max_retries=2, concurrency=4)
    session = engine.run()
    print(f"Completed {session.total} probes — {session.succeeded} succeeded.")
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Union

from ..adapters.adapter import BaseAdapter
from ..attacks.loader import load_entries_flat, AttackLoadError
from ..attacks.models import AttackEntry
from .session import ProbeResult, ProbeStatus, ScanSession
from .store import ResultStore

logger = logging.getLogger(__name__)

# Sentinel exception types that should trigger a retry
_RETRYABLE_EXCEPTIONS = (OSError, TimeoutError, ConnectionError)

# Try to include httpx errors if the library is available
try:
    import httpx
    _RETRYABLE_EXCEPTIONS = (*_RETRYABLE_EXCEPTIONS, httpx.TransportError, httpx.TimeoutException)
except ImportError:
    pass


def _backoff_seconds(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential back-off with a ceiling: base * 2^(attempt-1), capped at `cap`."""
    return min(base * (2 ** (attempt - 1)), cap)


class ScanEngine:
    """
    Orchestrates loading attack entries, dispatching probes, and storing results.

    Parameters
    ----------
    adapter       : Target adapter implementing BaseAdapter.send()
    system_prompt : Optional default system prompt injected for every probe
                    (unless the attack entry defines its own system_prompt_override).
    attacks_dir   : Root directory to scan for YAML suites. Defaults to repo attacks/.
    db_path       : SQLite database file path. Defaults to reports/sentinel.db.
    max_retries   : Total attempts per probe (1 = no retry, 3 = 1 original + 2 retries).
    concurrency   : Number of parallel probe threads.
    timeout       : Per-request timeout in seconds (passed to adapter if respected).
    on_result     : Optional callback fired after each probe completes.
                    Signature: (result: ProbeResult) -> None
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        system_prompt: Optional[str] = None,
        attacks_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
        max_retries: int = 3,
        concurrency: int = 4,
        timeout: float = 30.0,
        on_result: Optional[Callable[[ProbeResult], None]] = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1.")

        self.adapter = adapter
        self.system_prompt = system_prompt
        self.attacks_dir = Path(attacks_dir) if attacks_dir else None
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.timeout = timeout
        self.on_result = on_result
        self.store = ResultStore(db_path=db_path)

    # ── Core probe execution ──────────────────────────────────────────────────

    def _execute_probe(
        self,
        entry: AttackEntry,
        scan_id: str,
    ) -> ProbeResult:
        """
        Send a single attack probe with retry + exponential back-off.
        Returns a ProbeResult regardless of success or failure.
        """
        effective_system_prompt = entry.system_prompt_override or self.system_prompt
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            t_start = time.monotonic()
            try:
                response_text = self.adapter.send(
                    prompt=entry.prompt,
                    system_prompt=effective_system_prompt,
                )
                latency_ms = (time.monotonic() - t_start) * 1000.0

                result = ProbeResult(
                    attack_id=entry.id,
                    attack_name=entry.name,
                    category=entry.category,
                    attack_type=entry.type,
                    severity=entry.severity,
                    model=self.adapter.model,
                    prompt=entry.prompt,
                    system_prompt=effective_system_prompt,
                    status=ProbeStatus.SUCCESS,
                    response=response_text,
                    error_message=None,
                    timestamp=datetime.now(timezone.utc),
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    max_attempts=self.max_retries,
                    scan_id=scan_id,
                )
                logger.info(
                    "[%s] ✓ attack=%-10s  attempt=%d/%d  latency=%.0fms",
                    scan_id[:8],
                    entry.id,
                    attempt,
                    self.max_retries,
                    latency_ms,
                )
                return result

            except TimeoutError as exc:
                latency_ms = (time.monotonic() - t_start) * 1000.0
                last_error = exc
                logger.warning(
                    "[%s] ⏱ TIMEOUT  attack=%s  attempt=%d/%d",
                    scan_id[:8], entry.id, attempt, self.max_retries,
                )
                # Timeouts are always retried
                if attempt < self.max_retries:
                    sleep_s = _backoff_seconds(attempt)
                    logger.debug("Back-off %.1fs before retry.", sleep_s)
                    time.sleep(sleep_s)
                else:
                    return ProbeResult(
                        attack_id=entry.id,
                        attack_name=entry.name,
                        category=entry.category,
                        attack_type=entry.type,
                        severity=entry.severity,
                        model=self.adapter.model,
                        prompt=entry.prompt,
                        system_prompt=effective_system_prompt,
                        status=ProbeStatus.TIMEOUT,
                        response=None,
                        error_message=str(exc),
                        timestamp=datetime.now(timezone.utc),
                        latency_ms=round(latency_ms, 2),
                        attempt=attempt,
                        max_attempts=self.max_retries,
                        scan_id=scan_id,
                    )

            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                logger.warning(
                    "[%s] ✗ RETRYABLE ERROR  attack=%s  attempt=%d/%d  err=%s",
                    scan_id[:8], entry.id, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    sleep_s = _backoff_seconds(attempt)
                    logger.debug("Back-off %.1fs before retry.", sleep_s)
                    time.sleep(sleep_s)

            except Exception as exc:
                # Non-retryable: log immediately and bail out
                latency_ms = (time.monotonic() - t_start) * 1000.0
                logger.error(
                    "[%s] ✗ FATAL ERROR  attack=%s  err=%s",
                    scan_id[:8], entry.id, exc, exc_info=True,
                )
                return ProbeResult(
                    attack_id=entry.id,
                    attack_name=entry.name,
                    category=entry.category,
                    attack_type=entry.type,
                    severity=entry.severity,
                    model=self.adapter.model,
                    prompt=entry.prompt,
                    system_prompt=effective_system_prompt,
                    status=ProbeStatus.ERROR,
                    response=None,
                    error_message=f"{type(exc).__name__}: {exc}",
                    timestamp=datetime.now(timezone.utc),
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    max_attempts=self.max_retries,
                    scan_id=scan_id,
                )

        # All retry attempts exhausted for retryable errors
        return ProbeResult(
            attack_id=entry.id,
            attack_name=entry.name,
            category=entry.category,
            attack_type=entry.type,
            severity=entry.severity,
            model=self.adapter.model,
            prompt=entry.prompt,
            system_prompt=effective_system_prompt,
            status=ProbeStatus.ERROR,
            response=None,
            error_message=f"Failed after {self.max_retries} attempts. Last error: {last_error}",
            timestamp=datetime.now(timezone.utc),
            latency_ms=None,
            attempt=self.max_retries,
            max_attempts=self.max_retries,
            scan_id=scan_id,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        entries: Optional[List[AttackEntry]] = None,
        scan_id: Optional[str] = None,
    ) -> ScanSession:
        """
        Run a full scan and return a completed ScanSession.

        Parameters
        ----------
        entries  : Explicit list of AttackEntry objects to run. If None,
                   all entries are auto-discovered from the attacks directory.
        scan_id  : Custom scan identifier. Auto-generated UUID4 if not provided.

        Returns
        -------
        ScanSession with all results populated and persisted to SQLite.
        """
        sid = scan_id or str(uuid.uuid4())

        # ── 1. Load attack entries ─────────────────────────────────────────
        if entries is None:
            try:
                entries = load_entries_flat(self.attacks_dir)
            except AttackLoadError as exc:
                raise RuntimeError(f"Cannot load attack suites: {exc}") from exc

        if not entries:
            raise ValueError("No attack entries to run. Check your attacks/ directory.")

        logger.info(
            "Starting scan '%s' — %d attack(s) via adapter=%s concurrency=%d retries=%d",
            sid, len(entries), self.adapter, self.concurrency, self.max_retries,
        )

        # ── 2. Initialise + persist session skeleton ───────────────────────
        session = ScanSession(scan_id=sid, model=self.adapter.model)
        self.store.save_session(session)

        # ── 3. Fan out probes across thread pool ───────────────────────────
        results: List[ProbeResult] = []

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._execute_probe, entry, sid): entry
                for entry in entries
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    # Catch any unexpected exception from the future itself
                    logger.error("Unexpected future error for '%s': %s", entry.id, exc)
                    result = ProbeResult(
                        attack_id=entry.id,
                        attack_name=entry.name,
                        category=entry.category,
                        attack_type=entry.type,
                        severity=entry.severity,
                        model=self.adapter.model,
                        prompt=entry.prompt,
                        system_prompt=self.system_prompt,
                        status=ProbeStatus.ERROR,
                        response=None,
                        error_message=f"Future exception: {exc}",
                        timestamp=datetime.now(timezone.utc),
                        scan_id=sid,
                    )

                # Persist immediately so partial results survive crashes
                self.store.save_result(result)
                results.append(result)
                session.results.append(result)

                if self.on_result:
                    try:
                        self.on_result(result)
                    except Exception as cb_exc:
                        logger.warning("on_result callback raised: %s", cb_exc)

        # ── 4. Finalise and update session ─────────────────────────────────
        session.finish()
        self.store.save_session(session)

        logger.info(
            "Scan '%s' complete — total=%d  success=%d  errors=%d  duration=%.1fs",
            sid,
            session.total,
            session.succeeded,
            session.errors,
            (session.finished_at - session.started_at).total_seconds(),
        )

        return session
