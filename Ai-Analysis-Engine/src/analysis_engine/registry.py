"""
Shared callback registry. All nodes import from here — never from each other.

This breaks the circular import: previously planner.py and cleaning.py
imported RUN_CALLBACKS from analysis.py, which is wrong because analysis.py
is a peer node, not a shared module.

Usage:
    from analysis_engine.registry import RUN_CALLBACKS
    RUN_CALLBACKS[run_id] = my_callback
    cb = RUN_CALLBACKS.get(run_id)
"""
from typing import Callable

# run_id -> event_callback(event_type: str, data: dict)
RUN_CALLBACKS: dict[str, Callable[[str, dict], None]] = {}