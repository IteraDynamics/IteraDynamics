from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Action(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    EXIT = "EXIT"
    FLAT = "FLAT"
    HOLD = "HOLD"


@dataclass
class Intent:
    action: Action
    confidence: Optional[float] = None          # 0..1 if relevant
    target_exposure: Optional[float] = None     # 0..1 (fraction of equity) if strategy wants sizing
    horizon_hours: Optional[int] = None         # strategy can request horizon override (optional)
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
