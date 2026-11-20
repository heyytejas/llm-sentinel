"""
LLM Sentinel - Streamlit Community Cloud Entrypoint.

Delegates directly to the dashboard module inside llm_sentinel/dashboard/app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_sentinel.dashboard.app import main

if __name__ == "__main__":
    main()
