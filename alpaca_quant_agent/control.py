"""Manual kill-switch, settable from the dashboard.

Distinct from risk/circuit_breaker.py (which halts new entries automatically
off computed portfolio state): this is a human-operated override, persisted
as a small JSON file next to the ledger so it survives daemon restarts. Every
cycle (cycle.run_one_cycle) checks it before screening new candidates -- exit
management on existing positions is never affected, only the opening of new
trades.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class HaltState:
    halted: bool
    reason: str = ""
    set_at: str | None = None


def _flag_path(db_path: str) -> Path:
    return Path(db_path).with_name("manual_halt.json")


def get_halt_state(db_path: str) -> HaltState:
    path = _flag_path(db_path)
    if not path.exists():
        return HaltState(halted=False)
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return HaltState(halted=False)
    return HaltState(
        halted=bool(data.get("halted", False)),
        reason=data.get("reason", ""),
        set_at=data.get("set_at"),
    )


def set_halt_state(db_path: str, halted: bool, reason: str = "") -> HaltState:
    path = _flag_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = HaltState(halted=halted, reason=reason, set_at=datetime.utcnow().isoformat() + "Z")
    path.write_text(json.dumps({"halted": state.halted, "reason": state.reason, "set_at": state.set_at}))
    return state
