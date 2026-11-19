"""
Pydantic models for LLM Sentinel attack suite schema.

Each AttackEntry represents a single adversarial probe with
detection criteria and metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class AttackCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    SENSITIVE_DISCLOSURE = "sensitive_disclosure"
    SYSTEM_PROMPT_LEAKAGE = "system_prompt_leakage"
    JAILBREAK = "jailbreak"


class AttackType(str, Enum):
    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    PII_EXTRACTION = "pii_extraction"
    CREDENTIAL_EXTRACTION = "credential_extraction"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    PERSONA_ADOPTION = "persona_adoption"
    HYPOTHETICAL_FRAMING = "hypothetical_framing"
    OBFUSCATION = "obfuscation"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SemanticSimilarityRule(BaseModel):
    """Requires embedding cosine similarity >= threshold against a reference text."""
    threshold: float = Field(ge=0.0, le=1.0, description="Cosine similarity threshold (0.0–1.0)")
    reference: str = Field(min_length=1, description="Reference text to compare against")


class SuccessCondition(BaseModel):
    """
    Detection rules for evaluating whether an attack probe succeeded.
    At least one rule must be provided. Rules are evaluated with OR logic.
    """
    contains_any: Optional[List[str]] = Field(
        default=None,
        description="Attack succeeds if response contains any of these substrings (case-insensitive).",
    )
    not_contains: Optional[List[str]] = Field(
        default=None,
        description="Attack succeeds if response does NOT contain any of these (refusal expected but absent).",
    )
    regex_match: Optional[str] = Field(
        default=None,
        description="Attack succeeds if response matches this regex pattern.",
    )
    semantic_similarity: Optional[SemanticSimilarityRule] = Field(
        default=None,
        description="Attack succeeds if embedding similarity vs reference meets threshold.",
    )
    judge_prompt: Optional[str] = Field(
        default=None,
        description="Attack succeeds if LLM judge answers YES to this question about the response.",
    )

    @model_validator(mode="after")
    def at_least_one_rule(self) -> "SuccessCondition":
        rules = [
            self.contains_any,
            self.not_contains,
            self.regex_match,
            self.semantic_similarity,
            self.judge_prompt,
        ]
        if not any(r is not None for r in rules):
            raise ValueError("SuccessCondition must define at least one detection rule.")
        return self


class AttackEntry(BaseModel):
    """A single adversarial probe in an attack suite."""

    id: str = Field(
        min_length=3,
        pattern=r"^[A-Z]+-\d{3}$",
        description="Unique probe ID in format PREFIX-NNN (e.g. INJ-001).",
    )
    name: str = Field(min_length=3, description="Short human-readable attack name.")
    category: AttackCategory
    type: AttackType
    prompt: str = Field(min_length=1, description="Adversarial probe text sent to target model.")
    system_prompt_override: Optional[str] = Field(
        default=None,
        description="Overrides the default system prompt for this probe only.",
    )
    success_if: SuccessCondition
    severity: Severity
    description: Optional[str] = Field(default=None, description="Extended technique and risk explanation.")
    tags: List[str] = Field(default_factory=list, description="Freeform labels for filtering and grouping.")

    model_config = {"use_enum_values": True}


class AttackSuite(BaseModel):
    """A validated collection of attack entries loaded from a YAML file."""

    source_file: Optional[str] = Field(default=None, description="Path of the YAML file this suite was loaded from.")
    entries: List[AttackEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "AttackSuite":
        ids = [e.id for e in self.entries]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate attack IDs found in suite: {duplicates}")
        return self

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)
