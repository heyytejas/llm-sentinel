"""
YAML Attack Suite Loader for LLM Sentinel.

Discovers, parses, and validates YAML attack suite files
using the AttackEntry / AttackSuite Pydantic models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import yaml
from pydantic import ValidationError

from .models import AttackEntry, AttackSuite

logger = logging.getLogger(__name__)

# Default attacks directory (repo root / attacks/)
_DEFAULT_ATTACKS_DIR = Path(__file__).resolve().parents[2] / "attacks"


class AttackLoadError(Exception):
    """Raised when a YAML file fails to parse or fails schema validation."""
    pass


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
