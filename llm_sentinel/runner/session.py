"""
Data models for a runner probe session.

ProbeResult holds everything produced from sending one AttackEntry
through one adapter, including error state, timing, and retry metadata.
ScanSession aggregates all results from a single scan run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class ProbeStatus(str, Enum):
    SUCCESS = "success"       # Got a valid response from the target
    ERROR = "error"           # Adapter raised an exception after all retries
    TIMEOUT = "timeout"       # Request exceeded timeout limit
    SKIPPED = "skipped"       # Probe intentionally skipped


@dataclass
class ProbeResult:
    """Outcome of sending a single attack probe to the target model."""

    # Identity
    attack_id: str
    attack_name: str
    category: str
    attack_type: str
    severity: str

    # Execution context
    model: str
    prompt: str
    system_prompt: Optional[str]

    # Outcome
    status: ProbeStatus
    response: Optional[str]           # Raw text response from model; None on error
    error_message: Optional[str]      # Exception message if status == ERROR/TIMEOUT

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: Optional[float] = None  # Round-trip time in milliseconds

    # Retry telemetry
    attempt: int = 1                   # Which attempt produced this result (1-based)
    max_attempts: int = 1

    # Scan session linkage
    scan_id: Optional[str] = None      # FK to ScanSession.scan_id

    # Extended library & RAG metadata
    case_id: Optional[str] = None
    group: Optional[str] = None
    type: Optional[str] = None
    is_control: bool = False
    repetition: int = 1
    query: Optional[str] = None
    retrieved_chunks: Optional[List[str]] = None
    rule_based_verdict: Optional[str] = None
    judge_verdict: Optional[str] = None
    judge_reason: Optional[str] = None


@dataclass
class ScanSession:
    """Aggregates all probe results from one complete scan run."""

    scan_id: str
    model: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    results: List[ProbeResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.status == ProbeStatus.SUCCESS)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status in (ProbeStatus.ERROR, ProbeStatus.TIMEOUT))

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc)
