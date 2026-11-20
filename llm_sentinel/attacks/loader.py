"""
YAML Attack Suite Loader for LLM Sentinel.

Provides two independent loading paths:

  Legacy path (used by the runner):
    load_suite()         — load & validate a single YAML suite file
    load_all_suites()    — recursively discover & load all suites in a directory
    load_entries_flat()  — flatten all suites into one list of AttackEntry objects

  Seed library path (used for the new attack library):
    load_library()       — load data/seed_attacks.yaml (+ optional additions file),
                           skip disabled cases, and return (attacks, controls) tuple.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Union

import yaml
from pydantic import ValidationError

from .models import AttackEntry, AttackSuite, SeedAttackCase, SeedLibrary

logger = logging.getLogger(__name__)

# Default attacks directory (repo root / attacks/)
_DEFAULT_ATTACKS_DIR = Path(__file__).resolve().parents[2] / "attacks"

# Default data directory (repo root / data/)
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class AttackLoadError(Exception):
    """Raised when a YAML file fails to parse or fails schema validation."""
    pass


# ── Legacy loader (runner compatibility) ──────────────────────────────────────

def load_suite(path: Union[str, Path]) -> AttackSuite:
    """
    Load and validate a single YAML attack suite file.

    Args:
        path: Path to a YAML file containing a list of attack entries.

    Returns:
        A validated AttackSuite instance.

    Raises:
        AttackLoadError: If the file cannot be read, parsed, or validated.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise AttackLoadError(f"Attack suite file not found: {path}")
    if not path.is_file():
        raise AttackLoadError(f"Path is not a file: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise AttackLoadError(f"Expected a .yaml / .yml file, got: {path.suffix}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise AttackLoadError(f"YAML parse error in '{path}': {exc}") from exc

    if not isinstance(raw, list):
        raise AttackLoadError(
            f"Expected a YAML list of attack entries in '{path}', got {type(raw).__name__}."
        )
    if len(raw) == 0:
        raise AttackLoadError(f"Attack suite '{path}' is empty.")

    try:
        suite = AttackSuite(source_file=str(path), entries=raw)
    except ValidationError as exc:
        # Surface each individual field error clearly
        errors = "\n".join(
            f"  [{' > '.join(str(l) for l in e['loc'])}] {e['msg']}"
            for e in exc.errors()
        )
        raise AttackLoadError(
            f"Validation failed for '{path}':\n{errors}"
        ) from exc

    logger.info("Loaded %d attack(s) from '%s'", len(suite), path.name)
    return suite


def load_all_suites(directory: Union[str, Path, None] = None) -> List[AttackSuite]:
    """
    Recursively discover and load all YAML attack suites in a directory.

    Args:
        directory: Root directory to scan. Defaults to the repo 'attacks/' folder.

    Returns:
        List of validated AttackSuite instances (one per YAML file).

    Notes:
        Files that fail validation are logged as errors and skipped so a
        single bad file does not abort the entire scan.
    """
    root = Path(directory).resolve() if directory else _DEFAULT_ATTACKS_DIR

    if not root.is_dir():
        raise AttackLoadError(f"Attack directory not found: {root}")

    yaml_files = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    # Skip schema docs (SCHEMA.md, .gitkeep, etc.) – already filtered by glob
    yaml_files = [f for f in yaml_files if not f.name.startswith(".")]

    if not yaml_files:
        logger.warning("No YAML attack suites found under '%s'.", root)
        return []

    suites: List[AttackSuite] = []
    for yaml_file in yaml_files:
        try:
            suite = load_suite(yaml_file)
            suites.append(suite)
        except AttackLoadError as exc:
            logger.error("Skipping '%s': %s", yaml_file.name, exc)

    logger.info(
        "Loaded %d suite(s) with %d total attack(s) from '%s'.",
        len(suites),
        sum(len(s) for s in suites),
        root,
    )
    return suites


def load_entries_flat(directory: Union[str, Path, None] = None) -> List[AttackEntry]:
    """
    Load all attack entries across all suites into a single flat list.

    Args:
        directory: Root directory to scan. Defaults to the repo 'attacks/' folder.

    Returns:
        Flat list of all validated AttackEntry instances.
    """
    suites = load_all_suites(directory)
    return [entry for suite in suites for entry in suite.entries]


# ── Seed library loader ───────────────────────────────────────────────────────

def load_library(
    seed_path: Union[str, Path, None] = None,
    additions_path: Union[str, Path, None] = None,
    payloads_dir: Union[str, Path, None] = None,
) -> Tuple[List[SeedAttackCase], List[SeedAttackCase]]:
    """
    Load the attack library from the seed YAML file (and optional additions).

    Loading steps
    -------------
    1. Parse ``seed_path`` (defaults to ``data/seed_attacks.yaml``).
    2. If ``additions_path`` exists (defaults to
       ``data/attack_library_additions.yaml``), merge its ``cases`` list.
       Missing additions file is silently ignored.
    3. Validate that all ``host`` references on indirect cases exist in the
       combined ``hosts`` dict.  Unknown hosts raise ``ValueError``.
    4. Raise ``ValueError`` if any ``id`` is duplicated across both files.
    5. Skip cases with ``enabled: false``.
    6. Split the remaining enabled cases into ``attacks`` (``is_control=False``)
       and ``controls`` (``is_control=True``).

    Parameters
    ----------
    seed_path:
        Path to the seed YAML file.  Defaults to ``<repo>/data/seed_attacks.yaml``.
    additions_path:
        Path to an optional additions YAML file.
        Defaults to ``<repo>/data/attack_library_additions.yaml``.
        Silently skipped if the file does not exist.
    payloads_dir:
        Directory containing ``NAME.txt`` payload fixture files.
        Defaults to ``<repo>/data/payloads/``.
        Not used by this function directly, but stored for later use
        by ``SeedAttackCase.build_document()``.

    Returns
    -------
    (attacks, controls):
        ``attacks`` — enabled cases with ``is_control=False``.
        ``controls`` — enabled cases with ``is_control=True``.

    Raises
    ------
    AttackLoadError
        If the seed file cannot be read, parsed, or fails schema validation.
    ValueError
        If any indirect case references an unknown host identifier, or if
        duplicate IDs are found across the loaded files.
    """
    seed_path = Path(seed_path) if seed_path else _DATA_DIR / "seed_attacks.yaml"
    additions_path = (
        Path(additions_path) if additions_path
        else _DATA_DIR / "attack_library_additions.yaml"
    )
    # payloads_dir is resolved here for callers that want to pass it through
    # to build_document(); load_library() itself does not read payload files.
    _payloads_dir = Path(payloads_dir) if payloads_dir else _DATA_DIR / "payloads"  # noqa: F841

    # ── 1. Load seed file ─────────────────────────────────────────────────────
    if not seed_path.exists():
        raise AttackLoadError(f"Seed library file not found: {seed_path}")

    try:
        with seed_path.open("r", encoding="utf-8") as fh:
            raw_seed = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise AttackLoadError(f"YAML parse error in '{seed_path}': {exc}") from exc

    if not isinstance(raw_seed, dict) or "hosts" not in raw_seed or "cases" not in raw_seed:
        raise AttackLoadError(
            f"Seed library '{seed_path}' must have top-level 'hosts' and 'cases' keys; "
            f"got: {list(raw_seed.keys()) if isinstance(raw_seed, dict) else type(raw_seed).__name__}."
        )

    try:
        library = SeedLibrary(**raw_seed)
    except ValidationError as exc:
        errors = "\n".join(
            f"  [{' > '.join(str(loc) for loc in e['loc'])}] {e['msg']}"
            for e in exc.errors()
        )
        raise AttackLoadError(
            f"Validation failed for '{seed_path}':\n{errors}"
        ) from exc

    all_cases: List[SeedAttackCase] = list(library.cases)
    hosts: dict[str, str] = dict(library.hosts)

    # ── 2. Merge optional additions file ─────────────────────────────────────
    if additions_path.exists():
        try:
            with additions_path.open("r", encoding="utf-8") as fh:
                raw_adds = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise AttackLoadError(
                f"YAML parse error in '{additions_path}': {exc}"
            ) from exc

        if isinstance(raw_adds, dict):
            # Merge any additional host templates first
            if "hosts" in raw_adds and isinstance(raw_adds["hosts"], dict):
                hosts.update(raw_adds["hosts"])

            if "cases" in raw_adds and isinstance(raw_adds["cases"], list):
                try:
                    add_cases = [SeedAttackCase(**c) for c in raw_adds["cases"]]
                except ValidationError as exc:
                    errors = "\n".join(
                        f"  [{' > '.join(str(loc) for loc in e['loc'])}] {e['msg']}"
                        for e in exc.errors()
                    )
                    raise AttackLoadError(
                        f"Validation failed for '{additions_path}':\n{errors}"
                    ) from exc
                all_cases.extend(add_cases)

        logger.info(
            "Merged additions from '%s' (%d case(s) added).",
            additions_path.name,
            len(raw_adds.get("cases", [])) if isinstance(raw_adds, dict) else 0,
        )

    # ── 3. Global duplicate-ID check ──────────────────────────────────────────
    ids = [c.id for c in all_cases]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for id_ in ids:
        if id_ in seen:
            duplicates.add(id_)
        seen.add(id_)
    if duplicates:
        raise ValueError(
            f"Duplicate attack IDs found across library files: {sorted(duplicates)}. "
            f"Each case must have a unique 'id'."
        )

    # ── 4. Validate host references ───────────────────────────────────────────
    known_hosts = set(hosts.keys())
    for case in all_cases:
        if case.type == "indirect" and case.host not in known_hosts:
            raise ValueError(
                f"Attack '{case.id}' references unknown host '{case.host}'. "
                f"Known host identifiers: {sorted(known_hosts)}. "
                f"Add the template to the 'hosts' section or correct the 'host' field."
            )

    # ── 5. Skip disabled cases ────────────────────────────────────────────────
    enabled_cases = [c for c in all_cases if c.enabled]
    skipped = len(all_cases) - len(enabled_cases)
    if skipped:
        logger.info("Skipped %d disabled case(s).", skipped)

    # ── 6. Split into attacks and controls ────────────────────────────────────
    attacks = [c for c in enabled_cases if not c.is_control]
    controls = [c for c in enabled_cases if c.is_control]

    logger.info(
        "load_library(): %d attack(s), %d control(s) loaded from '%s'.",
        len(attacks),
        len(controls),
        seed_path.name,
    )

    return attacks, controls
