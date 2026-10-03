"""Prototype configuration — subset of Kirk's full flag schema.

Each flag corresponds to a controlled comparison from Kirk's ablation map.
Defaults reflect Kirk's stated starting preferences (base-hit philosophy).
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class SLMethod(str, Enum):
    STRUCTURE = "structure"
    FIXED     = "fixed"
    ATR       = "atr"


class TPMethod(str, Enum):
    BASE_HIT    = "base_hit"
    LIQUIDITY   = "liquidity"
    FIXED_RR    = "fixed_rr"
    PARTIAL_RUN = "partial_runner"


class MinGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class Config(BaseModel):
    # State machine
    use_bos: bool = Field(True,  description="Require BOS before entry zone")
    bos_requires_close: bool = Field(True, description="BOS must be candle close, not wick")

    # Confluence toggles (location filters — LOCATION IS NOT CONFIRMATION)
    use_fvg: bool  = Field(True,  description="FVG as entry-zone confluence")
    use_ema20: bool = Field(True, description="EMA20 as confluence/filter")
    use_ema_direction_filter: bool = Field(False, description="Only trade in EMA direction")
    require_asian_sweep: bool = Field(False, description="Require Asian session sweep")
    use_fibonacci: bool = Field(True, description="Require pullback into Fibonacci zone")

    # Fibonacci zone (anchor: confirmed displacement leg)
    fib_zone_low:  float = Field(0.500, ge=0.0, le=1.0)
    fib_zone_high: float = Field(0.786, ge=0.0, le=1.0)

    # Displacement thresholds
    displacement_body_atr_multiplier: float = Field(1.5, gt=0)
    displacement_body_range_ratio_min: float = Field(0.70, ge=0.0, le=1.0)
    displacement_allow_multi_candle: bool = Field(True)

    # Stop-loss
    sl_method: SLMethod = Field(SLMethod.STRUCTURE)
    sl_max_width_pts: float = Field(20.0, gt=0, description="Skip trade if stop exceeds this")
    sl_fixed_pts: float = Field(12.0, gt=0)
    sl_buffer_pts: float = Field(0.5, ge=0)

    # Take-profit
    tp_method: TPMethod = Field(TPMethod.BASE_HIT)
    tp_base_hit_pts: float = Field(15.0, gt=0)
    rr_ratio: float = Field(1.5, gt=0)

    # Confluence grading
    minimum_confluence_grade: MinGrade = Field(MinGrade.B)
