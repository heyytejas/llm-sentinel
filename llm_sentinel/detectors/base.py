"""
Base interface and data models for detectors in LLM Sentinel.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionResult:
    """Outcome of a detector evaluation on an LLM response."""
    attack_succeeded: bool
    matched_rule: Optional[str] = None
    matched_pattern: Optional[str] = None
    details: str = ""

    def __bool__(self) -> bool:
        return self.attack_succeeded

    @property
    def verdict(self) -> str:
        """Normalized verdict string: 'SUCCESS' or 'FAILURE'."""
        return "SUCCESS" if self.attack_succeeded else "FAILURE"


class BaseDetector(ABC):
    """Abstract base class for all detection modules."""

    @abstractmethod
    def evaluate(self, response: str, **kwargs) -> DetectionResult:
        """
        Evaluate an LLM response text against detection rules.

        Returns:
            DetectionResult indicating whether an attack succeeded.
        """
        pass
