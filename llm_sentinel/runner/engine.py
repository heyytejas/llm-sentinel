"""
LLM Sentinel Scan Runner Engine.

Loads attack library cases via load_library(), fans each probe through
the target adapter with configurable repeats, retries, and exponential back-off,
manages corpus poisoning and isolation for indirect/poisoning cases, and persists
all results to SQLite via ResultStore.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from ..adapters.adapter import BaseAdapter, MockTarget, get_adapter
from ..attacks.loader import load_hosts, load_library, AttackLoadError
from ..attacks.models import AttackEntry, SeedAttackCase
from ..detectors.rule_based import RuleBasedDetector
from .session import ProbeResult, ProbeStatus, ScanSession
from .store import ResultStore

logger = logging.getLogger(__name__)

# Default directories
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_SEED_PATH = _DEFAULT_DATA_DIR / "seed_attacks.yaml"
_DEFAULT_ADDITIONS_PATH = _DEFAULT_DATA_DIR / "attack_library_additions.yaml"
_DEFAULT_PAYLOADS_DIR = _DEFAULT_DATA_DIR / "payloads"

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
    Orchestrates loading attack entries from load_library(), dispatching probes,
    managing document poisoning for indirect/poisoning cases, and storing results.

    Parameters
    ----------
    adapter        : Target adapter implementing BaseAdapter.send()
    system_prompt  : Optional default system prompt injected for every probe.
    attacks_dir    : Kept for backwards compatibility.
    seed_path      : Path to seed attacks YAML.
    additions_path : Path to additions attacks YAML.
    payloads_dir   : Path to payloads directory.
    db_path        : SQLite database file path. Defaults to reports/sentinel.db.
    max_retries    : Total attempts per probe (1 = no retry, 3 = 1 original + 2 retries).
    concurrency    : Number of parallel probe threads.
    timeout        : Per-request timeout in seconds.
    repeats        : Number of repetitions per case (default 3 in library mode).
    limit          : Optional limit on the number of cases to evaluate.
    on_result      : Optional callback fired after each probe completes.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        system_prompt: Optional[str] = None,
        attacks_dir: Optional[Union[str, Path]] = None,
        seed_path: Optional[Union[str, Path]] = None,
        additions_path: Optional[Union[str, Path]] = None,
        payloads_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
        max_retries: int = 3,
        concurrency: int = 4,
        timeout: float = 30.0,
        repeats: Optional[int] = None,
        limit: Optional[int] = None,
        on_result: Optional[Callable[[ProbeResult], None]] = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1.")

        self.adapter = adapter
        self.system_prompt = system_prompt
        self.attacks_dir = Path(attacks_dir) if attacks_dir else None
        self.seed_path = Path(seed_path) if seed_path else _DEFAULT_SEED_PATH
        self.additions_path = Path(additions_path) if additions_path else _DEFAULT_ADDITIONS_PATH
        self.payloads_dir = Path(payloads_dir) if payloads_dir else _DEFAULT_PAYLOADS_DIR
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.timeout = timeout
        self.repeats = repeats
        self.limit = limit
        self.on_result = on_result
        self.store = ResultStore(db_path=db_path)

        # Hosts template map for indirect document construction
        try:
            self.hosts: Dict[str, str] = load_hosts(
                seed_path=self.seed_path,
                additions_path=self.additions_path if self.additions_path.exists() else None,
            )
        except Exception:
            self.hosts = {}

        # Thread lock to isolate corpus mutations for indirect/poisoning cases
        self._corpus_lock = threading.Lock()

    # ── Core probe execution ──────────────────────────────────────────────────

    def _execute_probe(
        self,
        entry: Union[AttackEntry, SeedAttackCase],
        scan_id: str,
        repetition: int = 1,
    ) -> ProbeResult:
        """
        Send a single probe with retry + exponential back-off.
        For indirect and poisoning cases, safely builds, indexes, queries,
        and removes the poisoned document so cases never cross-contaminate.
        """
        effective_system_prompt = (
            getattr(entry, "system_prompt_override", None)
            or self.system_prompt
            or getattr(self.adapter, "system_prompt", None)
        )
        attack_type = getattr(entry, "type", "direct")
        is_corpus_case = (attack_type in ("indirect", "poisoning")) and hasattr(entry, "build_document")

        if is_corpus_case:
            prompt_text = getattr(entry, "user_query", None) or getattr(entry, "query", None) or getattr(entry, "prompt", "")
        else:
            prompt_text = getattr(entry, "query", None) or getattr(entry, "prompt", "")

        doc_id = f"injected_{entry.id}_{repetition}"
        retrieved_chunks: List[str] = []

        if is_corpus_case:
            with self._corpus_lock:
                try:
                    poisoned_doc = entry.build_document(hosts=self.hosts, payloads_dir=self.payloads_dir)
                    if hasattr(self.adapter, "add_document"):
                        self.adapter.add_document(doc_id, poisoned_doc)
                    if hasattr(self.adapter, "reindex"):
                        self.adapter.reindex()
                    elif hasattr(self.adapter, "re_index"):
                        self.adapter.re_index()

                    if hasattr(self.adapter, "get_retrieved_chunks"):
                        retrieved_chunks = self.adapter.get_retrieved_chunks(prompt_text)
                    else:
                        retrieved_chunks = [poisoned_doc]

                    return self._send_with_retry(
                        prompt_text=prompt_text,
                        effective_system_prompt=effective_system_prompt,
                        entry=entry,
                        scan_id=scan_id,
                        repetition=repetition,
                        retrieved_chunks=retrieved_chunks,
                    )
                finally:
                    if hasattr(self.adapter, "remove_document"):
                        self.adapter.remove_document(doc_id)
                    if hasattr(self.adapter, "reindex"):
                        self.adapter.reindex()
                    elif hasattr(self.adapter, "re_index"):
                        self.adapter.re_index()
        else:
            return self._send_with_retry(
                prompt_text=prompt_text,
                effective_system_prompt=effective_system_prompt,
                entry=entry,
                scan_id=scan_id,
                repetition=repetition,
                retrieved_chunks=[],
            )

    def _send_with_retry(
        self,
        prompt_text: str,
        effective_system_prompt: Optional[str],
        entry: Union[AttackEntry, SeedAttackCase],
        scan_id: str,
        repetition: int,
        retrieved_chunks: List[str],
    ) -> ProbeResult:
        attack_id = entry.id
        case_id = entry.id
        attack_name = getattr(entry, "name", entry.id)
        category = getattr(entry, "category", "")
        group = getattr(entry, "group", category)
        attack_type = getattr(entry, "type", "direct")
        severity = getattr(entry, "severity", "medium")
        is_control = getattr(entry, "is_control", False)

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            t_start = time.monotonic()
            try:
                response_text = self.adapter.send(
                    prompt=prompt_text,
                    system_prompt=effective_system_prompt,
                )
                latency_ms = (time.monotonic() - t_start) * 1000.0

                # Extract canary tokens from target
                target_canaries: List[str] = []
                for obj in [self.adapter, getattr(self.adapter, "backend_adapter", None), getattr(self.adapter, "backend", None)]:
                    if obj:
                        if hasattr(obj, "canaries") and obj.canaries:
                            for c in obj.canaries:
                                if c and c not in target_canaries:
                                    target_canaries.append(c)
                        elif hasattr(obj, "canary") and obj.canary:
                            if obj.canary not in target_canaries:
                                target_canaries.append(obj.canary)

                # Evaluate response with RuleBasedDetector
                detector = RuleBasedDetector(
                    canary_tokens=target_canaries,
                    patterns=getattr(entry, "patterns", None),
                )
                if hasattr(entry, "success_if") and entry.success_if:
                    detection = detector.evaluate(response_text, success_condition=entry.success_if)
                else:
                    detection = detector.evaluate(response_text)

                verdict = "SUCCESS" if detection.attack_succeeded else "FAILURE"

                final_retrieved_chunks = retrieved_chunks
                if not final_retrieved_chunks and hasattr(self.adapter, "last_retrieved_chunks") and self.adapter.last_retrieved_chunks:
                    final_retrieved_chunks = list(self.adapter.last_retrieved_chunks)

                result = ProbeResult(
                    attack_id=attack_id,
                    attack_name=attack_name,
                    category=category,
                    attack_type=attack_type,
                    severity=severity,
                    model=self.adapter.model,
                    prompt=prompt_text,
                    system_prompt=effective_system_prompt,
                    status=ProbeStatus.SUCCESS,
                    response=response_text,
                    error_message=None,
                    timestamp=datetime.now(timezone.utc),
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    max_attempts=self.max_retries,
                    scan_id=scan_id,
                    case_id=case_id,
                    group=group,
                    type=attack_type,
                    is_control=is_control,
                    repetition=repetition,
                    query=prompt_text,
                    retrieved_chunks=final_retrieved_chunks,
                    rule_based_verdict=verdict,
                )
                logger.info(
                    "[%s] ✓ attack=%-10s rep=%d attempt=%d/%d latency=%.0fms verdict=%s",
                    scan_id[:8],
                    entry.id,
                    repetition,
                    attempt,
                    self.max_retries,
                    latency_ms,
                    verdict,
                )
                return result

            except TimeoutError as exc:
                latency_ms = (time.monotonic() - t_start) * 1000.0
                last_error = exc
                logger.warning(
                    "[%s] ⏱ TIMEOUT attack=%s rep=%d attempt=%d/%d",
                    scan_id[:8], entry.id, repetition, attempt, self.max_retries,
                )
                if attempt < self.max_retries:
                    sleep_s = _backoff_seconds(attempt)
                    time.sleep(sleep_s)
                else:
                    return ProbeResult(
                        attack_id=attack_id,
                        attack_name=attack_name,
                        category=category,
                        attack_type=attack_type,
                        severity=severity,
                        model=self.adapter.model,
                        prompt=prompt_text,
                        system_prompt=effective_system_prompt,
                        status=ProbeStatus.TIMEOUT,
                        response=None,
                        error_message=str(exc),
                        timestamp=datetime.now(timezone.utc),
                        latency_ms=round(latency_ms, 2),
                        attempt=attempt,
                        max_attempts=self.max_retries,
                        scan_id=scan_id,
                        case_id=case_id,
                        group=group,
                        type=attack_type,
                        is_control=is_control,
                        repetition=repetition,
                        query=prompt_text,
                        retrieved_chunks=retrieved_chunks,
                        rule_based_verdict="ERROR",
                    )

            except _RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                logger.warning(
                    "[%s] ✗ RETRYABLE ERROR attack=%s rep=%d attempt=%d/%d err=%s",
                    scan_id[:8], entry.id, repetition, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    sleep_s = _backoff_seconds(attempt)
                    time.sleep(sleep_s)

            except Exception as exc:
                latency_ms = (time.monotonic() - t_start) * 1000.0
                logger.error(
                    "[%s] ✗ FATAL ERROR attack=%s rep=%d err=%s",
                    scan_id[:8], entry.id, repetition, exc, exc_info=True,
                )
                return ProbeResult(
                    attack_id=attack_id,
                    attack_name=attack_name,
                    category=category,
                    attack_type=attack_type,
                    severity=severity,
                    model=self.adapter.model,
                    prompt=prompt_text,
                    system_prompt=effective_system_prompt,
                    status=ProbeStatus.ERROR,
                    response=None,
                    error_message=f"{type(exc).__name__}: {exc}",
                    timestamp=datetime.now(timezone.utc),
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    max_attempts=self.max_retries,
                    scan_id=scan_id,
                    case_id=case_id,
                    group=group,
                    type=attack_type,
                    is_control=is_control,
                    repetition=repetition,
                    query=prompt_text,
                    retrieved_chunks=retrieved_chunks,
                    rule_based_verdict="ERROR",
                )

        return ProbeResult(
            attack_id=attack_id,
            attack_name=attack_name,
            category=category,
            attack_type=attack_type,
            severity=severity,
            model=self.adapter.model,
            prompt=prompt_text,
            system_prompt=effective_system_prompt,
            status=ProbeStatus.ERROR,
            response=None,
            error_message=f"Failed after {self.max_retries} attempts. Last error: {last_error}",
            timestamp=datetime.now(timezone.utc),
            latency_ms=None,
            attempt=self.max_retries,
            max_attempts=self.max_retries,
            scan_id=scan_id,
            case_id=case_id,
            group=group,
            type=attack_type,
            is_control=is_control,
            repetition=repetition,
            query=prompt_text,
            retrieved_chunks=retrieved_chunks,
            rule_based_verdict="ERROR",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        entries: Optional[List[Union[AttackEntry, SeedAttackCase]]] = None,
        scan_id: Optional[str] = None,
        repeats: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> ScanSession:
        """
        Run a full scan and return a completed ScanSession.

        Parameters
        ----------
        entries : Explicit list of attack/control objects to run. If None,
                  both attacks and controls are loaded via load_library().
        scan_id : Custom scan identifier. Auto-generated UUID4 if not provided.
        repeats : Number of times to run each case (default 3 in library mode).
        limit   : Limit the number of cases to evaluate.

        Returns
        -------
        ScanSession with all results populated and persisted to SQLite.
        """
        sid = scan_id or str(uuid.uuid4())

        # ── 1. Load attack & control entries ──────────────────────────────
        if entries is None:
            try:
                attacks, controls = load_library(
                    seed_path=self.seed_path,
                    additions_path=self.additions_path,
                    payloads_dir=self.payloads_dir,
                )
                entries = list(attacks) + list(controls)
            except Exception as exc:
                raise RuntimeError(f"Cannot load attack library: {exc}") from exc

            effective_repeats = repeats if repeats is not None else (self.repeats if self.repeats is not None else 3)
        else:
            effective_repeats = repeats if repeats is not None else (self.repeats if self.repeats is not None else 1)

        if not entries:
            raise ValueError("No attack entries to run. Check your attacks/ directory.")

        effective_limit = limit if limit is not None else self.limit
        if effective_limit is not None and effective_limit > 0:
            entries = entries[:effective_limit]

        logger.info(
            "Starting scan '%s' — %d case(s) x %d repeat(s) = %d total probes via adapter=%s concurrency=%d",
            sid, len(entries), effective_repeats, len(entries) * effective_repeats,
            self.adapter, self.concurrency,
        )

        # ── 2. Initialise + persist session skeleton ───────────────────────
        session = ScanSession(scan_id=sid, model=self.adapter.model)
        self.store.save_session(session)

        # ── 3. Fan out probes across thread pool ───────────────────────────
        work_items: List[Tuple[Union[AttackEntry, SeedAttackCase], int]] = []
        for rep in range(1, effective_repeats + 1):
            for entry in entries:
                work_items.append((entry, rep))

        results: List[ProbeResult] = []

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._execute_probe, entry, sid, rep): (entry, rep)
                for entry, rep in work_items
            }
            for future in as_completed(futures):
                entry, rep = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Unexpected future error for '%s' rep=%d: %s", entry.id, rep, exc)
                    result = ProbeResult(
                        attack_id=entry.id,
                        attack_name=getattr(entry, "name", entry.id),
                        category=getattr(entry, "category", ""),
                        attack_type=getattr(entry, "type", "direct"),
                        severity=getattr(entry, "severity", "medium"),
                        model=self.adapter.model,
                        prompt=getattr(entry, "query", None) or getattr(entry, "prompt", ""),
                        system_prompt=self.system_prompt,
                        status=ProbeStatus.ERROR,
                        response=None,
                        error_message=f"Future exception: {exc}",
                        timestamp=datetime.now(timezone.utc),
                        scan_id=sid,
                        case_id=entry.id,
                        group=getattr(entry, "group", ""),
                        type=getattr(entry, "type", "direct"),
                        is_control=getattr(entry, "is_control", False),
                        repetition=rep,
                        query=getattr(entry, "query", None) or getattr(entry, "prompt", ""),
                        retrieved_chunks=[],
                        rule_based_verdict="ERROR",
                    )

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


# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Sentinel Scan Runner Engine")
    parser.add_argument(
        "--target",
        type=str,
        choices=["rag", "raw"],
        default="rag",
        help="Target application wrapper: 'rag' (default) or 'raw'.",
    )
    parser.add_argument(
        "--repeats", "-r",
        type=int,
        default=3,
        help="Number of times to run each case (default: 3)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of cases to evaluate",
    )
    parser.add_argument(
        "--mock-target", "--mock",
        action="store_true",
        help="Run in mock-target mode for zero API cost testing",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        help="Target provider name (openai, anthropic, ollama, mock). Default: mock",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model identifier",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=4,
        help="Number of parallel probe threads (default: 4)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite results database",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    if parsed.mock_target or parsed.provider == "mock":
        backend = MockTarget(model=parsed.model or "mock-sentinel-v1")
    else:
        backend = get_adapter(parsed.provider, model=parsed.model) if parsed.model else get_adapter(parsed.provider)

    if parsed.target == "rag":
        from ..target.rag_app import RagTarget
        adapter = RagTarget(backend_adapter=backend)
    else:
        adapter = backend

    engine = ScanEngine(
        adapter=adapter,
        db_path=parsed.db_path,
        concurrency=parsed.concurrency,
        repeats=parsed.repeats,
        limit=parsed.limit,
    )
    session = engine.run()
    print(f"Scan complete: {session.total} probes run. Succeeded: {session.succeeded}, Errors: {session.errors}")
    return 0
