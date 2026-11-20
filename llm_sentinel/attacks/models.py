"""
Pydantic models for LLM Sentinel attack suite schema.

Two model hierarchies live here:

  Legacy (used by the runner via load_entries_flat / load_suite):
    AttackCategory, AttackType, Severity, SemanticSimilarityRule,
    SuccessCondition, AttackEntry, AttackSuite

  Seed library (used by load_library):
    SeedAttackCase, SeedLibrary
    SeedAttackCase.build_document() — resolves payloads and injects into host
    templates for indirect attack cases.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ── Legacy models (kept for runner / load_entries_flat backward compat) ────────

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


# ── Seed library models (used by load_library) ─────────────────────────────────

# Matches {{payload:NAME}} placeholders in injection strings.
_PAYLOAD_PLACEHOLDER_RE = re.compile(r"\{\{payload:([^}]+)\}\}")


class SeedAttackCase(BaseModel):
    """
    A single attack case from the seed library YAML.

    Fields are conditionally required based on ``type``:

    - ``direct``    → ``query`` is required.
    - ``indirect``  → ``host``, ``injection``, and ``user_query`` are required;
                      ``position`` is optional (default "middle").
    - ``poisoning`` → ``setup_document`` and ``user_query`` are required.

    Host identifier validation (against the loaded hosts dict) is performed
    in ``load_library()`` after the model is constructed, since the hosts dict
    is a runtime dependency not available inside Pydantic validators.
    """

    # ── Core ──────────────────────────────────────────────────────────────────
    id: str = Field(
        pattern=r"^[A-Z]+-\d{3}$",
        description="Unique probe ID in format PREFIX-NNN (e.g. CI-001).",
    )
    category: str = Field(description="OWASP category tag, e.g. LLM01.")
    group: str = Field(description="Logical attack group name.")
    type: Literal["direct", "indirect", "poisoning"]
    severity: str = Field(description="Severity level: low, medium, high, or critical.")

    # ── Direct-only ───────────────────────────────────────────────────────────
    query: Optional[str] = Field(
        default=None,
        description="Prompt sent directly to the target model (required for type='direct').",
    )

    # ── Indirect-only ─────────────────────────────────────────────────────────
    host: Optional[str] = Field(
        default=None,
        description="Host template key from the 'hosts' dict (required for type='indirect').",
    )
    injection: Optional[str] = Field(
        default=None,
        description=(
            "Adversarial text injected into the host document. "
            "May contain {{payload:NAME}} placeholders (required for type='indirect')."
        ),
    )
    user_query: Optional[str] = Field(
        default=None,
        description="Query the user submits after the poisoned document is indexed "
                    "(required for type='indirect' and type='poisoning').",
    )
    position: Literal["start", "middle", "end"] = Field(
        default="middle",
        description=(
            "Where to inject into the host template: "
            "'middle' replaces {{INJECTION}}, "
            "'start' prepends before the host text, "
            "'end' appends after the host text."
        ),
    )

    # ── Poisoning-only ────────────────────────────────────────────────────────
    setup_document: Optional[str] = Field(
        default=None,
        description=(
            "Document or host template used to poison the retrieval corpus "
            "(required for type='poisoning'). May be a host key or inline text."
        ),
    )

    # ── Control flags ─────────────────────────────────────────────────────────
    is_control: bool = Field(
        default=False,
        description="When True this case is a control probe, not an attack.",
    )
    expected_behavior: Optional[str] = Field(
        default=None,
        description="Description of secure/expected model behavior for control probes.",
    )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    enabled: bool = Field(
        default=True,
        description="When False, load_library() skips this case entirely.",
    )

    # ── Rules / detection ─────────────────────────────────────────────────────
    success_rule: List[str] = Field(
        default_factory=list,
        description="Detection rules: canary_in_response | pattern_match | judge.",
    )
    patterns: List[str] = Field(
        default_factory=list,
        description="Strings or regexes searched for in the model response.",
    )
    judge_goal: Optional[str] = Field(
        default=None,
        description="Guidance for the LLM-as-judge on what constitutes a successful attack.",
    )
    payload_ref: Optional[str] = Field(
        default=None,
        description="Name of the fixture file under data/payloads/ (without .txt extension).",
    )
    source: str = Field(description="Citation or origin of this attack case.")

    # Allow extra YAML fields (e.g. 'note', 'description') without failing validation.
    model_config = {"extra": "ignore"}

    # ── Type-specific field validation ────────────────────────────────────────

    @model_validator(mode="after")
    def _enforce_type_requirements(self) -> "SeedAttackCase":
        if self.type == "direct":
            if not self.query:
                raise ValueError(
                    f"Attack '{self.id}': type='direct' requires a non-empty 'query' field."
                )

        elif self.type == "indirect":
            missing = [
                name
                for name, value in [
                    ("host", self.host),
                    ("injection", self.injection),
                    ("user_query", self.user_query),
                ]
                if not value
            ]
            if missing:
                raise ValueError(
                    f"Attack '{self.id}': type='indirect' requires the following "
                    f"missing field(s): {missing}."
                )

        elif self.type == "poisoning":
            missing = [
                name
                for name, value in [
                    ("setup_document", self.setup_document),
                    ("user_query", self.user_query),
                ]
                if not value
            ]
            if missing:
                raise ValueError(
                    f"Attack '{self.id}': type='poisoning' requires the following "
                    f"missing field(s): {missing}."
                )

        return self

    # ── Document builder ──────────────────────────────────────────────────────

    def build_document(self, hosts: Dict[str, str], payloads_dir: Union[str, Path]) -> str:
        """
        Build the poisoned host document for indirect attack cases.

        Steps
        -----
        1. Resolve every ``{{payload:NAME}}`` placeholder in ``self.injection``
           by reading ``payloads_dir/NAME.txt`` as plain text.  The file is
           **never executed** — only read.
        2. Substitute the resolved injection into the host template according
           to ``self.position``:

           - ``"middle"`` : replaces ``{{INJECTION}}`` in the template.
           - ``"start"``  : strips ``{{INJECTION}}`` from the template and
                            prepends the injection text.
           - ``"end"``    : strips ``{{INJECTION}}`` from the template and
                            appends the injection text.

        Parameters
        ----------
        hosts:
            Mapping of host key → template text (from the seed YAML ``hosts``
            section).
        payloads_dir:
            Directory containing ``NAME.txt`` payload fixture files.

        Returns
        -------
        str
            The complete document text ready to be indexed / shown to the model.

        Raises
        ------
        ValueError
            If called on a non-indirect case.
        FileNotFoundError
            If a referenced payload fixture file is missing.
        """
        if self.type not in ("indirect", "poisoning"):
            raise ValueError(
                f"build_document() is only valid for indirect and poisoning cases; "
                f"case '{self.id}' has type='{self.type}'."
            )

        payloads_dir = Path(payloads_dir)

        if self.type == "poisoning":
            raw_text = self.setup_document or ""
            if raw_text in hosts:
                template = hosts[raw_text]
            else:
                template = raw_text

            # Resolve any {{payload:NAME}} in template
            resolved_template = template
            for match in _PAYLOAD_PLACEHOLDER_RE.finditer(template):
                name = match.group(1)
                payload_path = payloads_dir / f"{name}.txt"
                if not payload_path.exists():
                    raise FileNotFoundError(
                        f"Payload file missing for '{{{{payload:{name}}}}}' "
                        f"referenced by attack '{self.id}'. "
                        f"Expected: '{payload_path}'."
                    )
                payload_text = payload_path.read_text(encoding="utf-8")
                resolved_template = resolved_template.replace(match.group(0), payload_text)

            if self.injection:
                resolved_inj = self.injection
                for match in _PAYLOAD_PLACEHOLDER_RE.finditer(self.injection):
                    name = match.group(1)
                    payload_path = payloads_dir / f"{name}.txt"
                    if not payload_path.exists():
                        raise FileNotFoundError(
                            f"Payload file missing for '{{{{payload:{name}}}}}' "
                            f"referenced by attack '{self.id}'. "
                            f"Expected: '{payload_path}'."
                        )
                    payload_text = payload_path.read_text(encoding="utf-8")
                    resolved_inj = resolved_inj.replace(match.group(0), payload_text)

                if "{{INJECTION}}" in resolved_template:
                    return resolved_template.replace("{{INJECTION}}", resolved_inj)
                if self.position == "start":
                    return resolved_inj + "\n\n" + resolved_template.strip()
                else:
                    return resolved_template.strip() + "\n\n" + resolved_inj

            return resolved_template

        # Indirect cases:
        # Resolve {{payload:NAME}} placeholders in the injection string.
        resolved_injection = self.injection
        for match in _PAYLOAD_PLACEHOLDER_RE.finditer(self.injection):
            name = match.group(1)
            payload_path = payloads_dir / f"{name}.txt"
            if not payload_path.exists():
                raise FileNotFoundError(
                    f"Payload file missing for '{{{{payload:{name}}}}}' "
                    f"referenced by attack '{self.id}'. "
                    f"Expected: '{payload_path}'. "
                    f"Add the plain-text fixture file and retry."
                )
            payload_text = payload_path.read_text(encoding="utf-8")
            resolved_injection = resolved_injection.replace(match.group(0), payload_text)

        template = hosts[self.host]

        if self.position == "middle":
            return template.replace("{{INJECTION}}", resolved_injection)

        # For start/end: strip the {{INJECTION}} marker and surrounding blank
        # lines from the template, then prepend or append the injection.
        stripped_template = template.replace("{{INJECTION}}", "").strip()

        if self.position == "start":
            return resolved_injection + "\n\n" + stripped_template
        else:  # "end"
            return stripped_template + "\n\n" + resolved_injection


class SeedLibrary(BaseModel):
    """
    Top-level structure of the seed attack library YAML.

    Contains a ``hosts`` mapping (document templates) and a ``cases`` list
    (individual attack probes).  Duplicate IDs within the same file are
    rejected at construction time.
    """

    hosts: Dict[str, str] = Field(description="Named host document templates.")
    cases: List[SeedAttackCase]

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> "SeedLibrary":
        ids = [c.id for c in self.cases]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for id_ in ids:
            if id_ in seen:
                duplicates.add(id_)
            seen.add(id_)
        if duplicates:
            raise ValueError(
                f"Duplicate attack IDs found in seed library: {sorted(duplicates)}"
            )
        return self
