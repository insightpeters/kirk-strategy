"""Oil Sweep-Break-Retest v1 — configuration and preset management."""
from __future__ import annotations

import json
from pathlib import Path
from pydantic import BaseModel, Field

PRESETS_DIR = Path(__file__).parent / "oil_presets"
PRESETS_DIR.mkdir(exist_ok=True)


class OilConfig(BaseModel):
    # ── Instrument ─────────────────────────────────────────────────────────────
    spread_pts: float = Field(0.03, ge=0, description="Estimated spread (points)")
    contract_size: float = Field(1000.0, gt=0, description="$ P&L per point per 1 lot")
    min_lot: float = Field(0.01, gt=0)
    lot_step: float = Field(0.01, gt=0)

    # ── Session windows (all America/New_York hours, 24h clock) ────────────────
    asian_start_hour: int = Field(20, ge=0, le=23, description="Asian range start (prev-day NY hour)")
    asian_end_hour: int = Field(2, ge=0, le=23, description="Asian range end (NY hour of session day)")
    london_end_hour: int = Field(5, ge=0, le=23, description="London extension end (NY hour)")
    entry_start_hour: int = Field(5, ge=0, le=23)
    entry_end_hour: int = Field(12, ge=0, le=23)
    eod_hour: int = Field(15, ge=0, le=23)
    eod_minute: int = Field(30, ge=0, le=59)

    # ── Range quality ──────────────────────────────────────────────────────────
    range_min_atr_mult: float = Field(1.5, gt=0,
        description="Require range width ≥ mult × ATR(14) to use session levels")

    # ── Sweep ──────────────────────────────────────────────────────────────────
    sweep_atr_mult: float = Field(0.10, gt=0,
        description="Min sweep = max(mult × ATR, 2 × spread)")
    sweep_close_bars: int = Field(3, gt=0,
        description="15m candles allowed to close back below swept level")

    # ── BOS ────────────────────────────────────────────────────────────────────
    bos_bars: int = Field(4, gt=0,
        description="15m candles after reclaim to achieve BOS + displacement")

    # ── Displacement (on the BOS candle) ──────────────────────────────────────
    disp_body_atr_mult: float = Field(0.8, gt=0, description="Min candle body ≥ mult × ATR")
    disp_body_range_pct: float = Field(0.60, gt=0, le=1.0, description="Min body/full-range ratio")
    disp_close_top_pct: float = Field(0.25, gt=0, le=0.5,
        description="Close must be in bottom pct of bar range (sell)")

    # ── Retracement (5m) ──────────────────────────────────────────────────────
    retrace_pct_low: float = Field(0.50, ge=0, le=1.0,
        description="Retrace zone lower bound (measured up from displacement low)")
    retrace_pct_high: float = Field(0.75, ge=0, le=1.0,
        description="Retrace zone upper bound")
    retrace_bars_5m: int = Field(6, gt=0,
        description="5m candles (from displacement close) to touch zone + confirm")

    # ── Stop loss ──────────────────────────────────────────────────────────────
    sl_buffer_atr_mult: float = Field(0.20, gt=0,
        description="SL buffer = max(mult × ATR, 2 × spread)")
    max_sl_atr_mult: float = Field(2.0, gt=0,
        description="Skip trade if entry-to-SL > mult × ATR")
    max_spread_sl_pct: float = Field(0.10, gt=0, le=1.0,
        description="Reject entry if spread > pct × initial SL distance")

    # ── Exits ──────────────────────────────────────────────────────────────────
    tp1_r: float = Field(1.5, gt=0, description="Target for tickets 1–2")
    tp2_r: float = Field(2.0, gt=0, description="Target for tickets 3–4")
    tp3_r: float = Field(3.0, gt=0, description="Target for tickets 5–6")
    tp4_r: float = Field(4.0, gt=0, description="Target for tickets 7–8")
    use_runner: bool = Field(False,
        description="After 2R, trail last-2 tickets above confirmed 15m swing highs + buffer")

    # ── Position sizing ────────────────────────────────────────────────────────
    account_equity: float = Field(10_000.0, gt=0)
    risk_pct: float = Field(0.01, gt=0, le=1.0, description="Fraction of equity at risk per basket")
    max_tickets: int = Field(8, ge=1, le=8)

    # ── Daily limits ───────────────────────────────────────────────────────────
    max_baskets_per_day: int = Field(2, gt=0)
    max_losers_per_day: int = Field(2, gt=0)
    max_daily_loss_pct: float = Field(0.02, gt=0, le=1.0,
        description="Stop new entries after equity falls this fraction from day-open")

    # ── News blocking ──────────────────────────────────────────────────────────
    news_block_before_min: int = Field(15, ge=0)
    news_block_after_min: int = Field(30, ge=0)
    use_eia_block: bool = Field(True,
        description="Block entries around EIA Wednesday 10:30 AM ET (approx)")

    # ── Cost model ─────────────────────────────────────────────────────────────
    slippage_pts: float = Field(0.03, ge=0,
        description="One-way slippage estimate per fill (points)")
    commission_per_lot: float = Field(3.0, ge=0,
        description="Commission per lot per side ($); 0 if spread-only broker")


# ── Preset I/O ────────────────────────────────────────────────────────────────

def save_preset(name: str, config: OilConfig) -> Path:
    path = PRESETS_DIR / f"{name}.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_preset(name: str) -> OilConfig:
    path = PRESETS_DIR / f"{name}.json"
    return OilConfig.model_validate_json(path.read_text(encoding="utf-8"))


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))
