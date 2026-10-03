"""10-state FSM engine for Kirk's sweep-reversal strategy.

Processes M5 bars sequentially up to reveal_idx.  Every state transition is
logged with the bar index that caused it.  The UI calls run_engine(...) on
every slider tick, receiving a fresh EngineResult snapshot.

LOCATION IS NOT CONFIRMATION — FVG/Fibonacci/EMA20 are confluence locations,
not entry triggers.  Entry means an order was actually filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import Config, SLMethod, TPMethod, MinGrade
from synthetic import SetupMeta


# ── State labels ──────────────────────────────────────────────────────────────
STATE_LABELS = {
    0: "WAITING",
    1: "LIQUIDITY IDENTIFIED",
    2: "SWEPT",
    3: "DISPLACEMENT",
    4: "BOS",
    5: "ENTRY ZONE",
    6: "WAIT RETRACEMENT",
    7: "ENTRY FILLED",
    8: "TRADE MANAGEMENT",
    9: "EXIT / RESET",
}


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class Transition:
    from_state: int
    to_state: int
    bar_idx: int
    reason: str


@dataclass
class ConfluenceScore:
    asian_sweep: bool = False
    has_fvg: bool = False
    fvg_overlap: bool = False
    fibonacci_zone: bool = False
    ema_aligned: bool = False
    ema_distance_ok: bool = False
    score: int = 0
    grade: str = "C"

    def compute_grade(self) -> None:
        self.score = sum([
            self.asian_sweep,
            self.has_fvg,
            self.fvg_overlap,
            self.fibonacci_zone,
            self.ema_aligned,
            self.ema_distance_ok,
        ])
        if self.score >= 4:
            self.grade = "A"
        elif self.score >= 2:
            self.grade = "B"
        else:
            self.grade = "C"


@dataclass
class EngineResult:
    current_state: int = 0
    transitions: list[Transition] = field(default_factory=list)
    confluence: Optional[ConfluenceScore] = None

    # Liquidity
    asian_high: Optional[float] = None
    asian_low: Optional[float] = None
    liq_direction: Optional[str] = None   # "HIGH" or "LOW"

    # Sweep
    sweep_bar: Optional[int] = None
    sweep_extreme: Optional[float] = None

    # Displacement
    disp_start: Optional[int] = None
    disp_end: Optional[int] = None

    # BOS
    bos_bar: Optional[int] = None
    bos_level: Optional[float] = None

    # Entry zone geometry
    fib_zone_low_price: Optional[float] = None
    fib_zone_high_price: Optional[float] = None
    fvg_top: Optional[float] = None
    fvg_bot: Optional[float] = None
    ema20_at_entry: Optional[float] = None

    # Trade
    trade_direction: Optional[str] = None   # "LONG" or "SHORT"
    entry_bar: Optional[int] = None
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    rr: Optional[float] = None

    # Exit
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   # "TP_HIT" or "SL_HIT"
    pnl_pts: Optional[float] = None

    # No-trade veto
    veto_reason: Optional[str] = None


# ── Indicator helpers ─────────────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ── Main engine ───────────────────────────────────────────────────────────────
def run_engine(
    df: pd.DataFrame,
    meta,                   # SetupMeta | None  — not used in computation
    config: Config,
    reveal_idx: int,
    asian_end_bar: int = 48,   # bar index where Asian session closes (default = 48 × 5 min = midnight)
) -> EngineResult:
    """Process bars 0..reveal_idx and return current FSM state + all derived data."""

    result = EngineResult()

    if reveal_idx < 0 or len(df) == 0:
        return result

    bars = df.iloc[: reveal_idx + 1].copy().reset_index(drop=True)
    n = len(bars)

    atr_series = compute_atr(bars)
    ema_series = compute_ema(bars["close"], 20)

    # ─────────────────────────────────────────────────────────────────────────
    # STATE 0 → 1: Asian session must be complete (bar 47 closed = bar index 47)
    # ─────────────────────────────────────────────────────────────────────────
    if reveal_idx < asian_end_bar:
        result.current_state = 0
        return result

    asian_bars = bars.iloc[:asian_end_bar]
    asian_high = float(asian_bars["high"].max())
    asian_low  = float(asian_bars["low"].min())

    result.asian_high = round(asian_high, 2)
    result.asian_low  = round(asian_low,  2)
    result.current_state = 1
    result.transitions.append(Transition(0, 1, asian_end_bar - 1, "Asian session closed — PDH/PDL and Asian H/L usable"))

    # ─────────────────────────────────────────────────────────────────────────
    # STATE 1 → 2 → 3: Collect all sweep candidates then find the first one
    # that leads to valid displacement within 10 bars.  Skipping failed sweeps
    # lets the engine catch the real setup when an early sweep fizzles out.
    # ─────────────────────────────────────────────────────────────────────────
    sweep_candidates: list[tuple[int, str, float]] = []
    for i in range(asian_end_bar, n):
        bar = bars.iloc[i]
        if bar["high"] > asian_high and bar["close"] < asian_high:
            sweep_candidates.append((i, "HIGH", float(bar["high"])))
        elif bar["low"] < asian_low and bar["close"] > asian_low:
            sweep_candidates.append((i, "LOW", float(bar["low"])))

    if not sweep_candidates:
        return result  # stays at state 1

    sweep_bar_idx: Optional[int] = None
    sweep_extreme: Optional[float] = None
    liq_direction: Optional[str] = None
    disp_bars: list[int] = []

    for cand_idx, cand_dir, cand_extreme in sweep_candidates:
        pre_atr = compute_atr(bars.iloc[:cand_idx])
        pre_atr_val = max(float(pre_atr.iloc[-1]) if len(pre_atr) > 0 else 1.0, 0.01)
        exp_dir = "DOWN" if cand_dir == "HIGH" else "UP"

        cand_disp: list[int] = []
        for j in range(cand_idx + 1, min(n, cand_idx + 10)):
            bar = bars.iloc[j]
            body = abs(float(bar["close"]) - float(bar["open"]))
            bar_range = float(bar["high"]) - float(bar["low"])
            body_ratio = body / bar_range if bar_range > 0 else 0
            is_dir = (
                (exp_dir == "DOWN" and bar["close"] < bar["open"]) or
                (exp_dir == "UP"   and bar["close"] > bar["open"])
            )
            is_strong   = body >= config.displacement_body_atr_multiplier * pre_atr_val
            is_decisive = body_ratio >= config.displacement_body_range_ratio_min
            if is_dir and is_strong and is_decisive:
                cand_disp.append(j)
            elif cand_disp:
                break

        if cand_disp:
            sweep_bar_idx = cand_idx
            liq_direction = cand_dir
            sweep_extreme = cand_extreme
            disp_bars     = cand_disp
            break

    # No sweep produced displacement — show the most recent sweep seen (State 2)
    if sweep_bar_idx is None:
        last = sweep_candidates[-1]
        result.liq_direction = last[1]
        result.sweep_bar     = last[0]
        result.sweep_extreme = round(last[2], 2)
        result.current_state = 2
        result.transitions.append(
            Transition(1, 2, last[0],
                       f"Sweep {'above Asian High' if last[1]=='HIGH' else 'below Asian Low'} "
                       f"@ {last[2]:.2f} — awaiting displacement")
        )
        return result

    result.liq_direction = liq_direction
    result.sweep_bar     = sweep_bar_idx
    result.sweep_extreme = round(sweep_extreme, 2)
    result.current_state = 2
    result.transitions.append(
        Transition(1, 2, sweep_bar_idx,
                   f"Sweep — wick {'above Asian High' if liq_direction=='HIGH' else 'below Asian Low'} "
                   f"@ {sweep_extreme:.2f}, bar closes back")
    )

    disp_start   = disp_bars[0]
    disp_end     = disp_bars[-1]
    expected_dir = "DOWN" if liq_direction == "HIGH" else "UP"
    result.disp_start = disp_start
    result.disp_end   = disp_end
    result.current_state = 3
    result.transitions.append(
        Transition(2, 3, disp_start,
                   f"Displacement — {len(disp_bars)} strong {'bearish' if expected_dir=='DOWN' else 'bullish'} bar(s)")
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STATE 3 → 4: BOS (Break of Structure)
    # Reference: the minimum (bearish) / maximum (bullish) of the 30 bars
    # immediately before the sweep bar — that is the swing the displacement
    # must break to confirm structure change.
    # ─────────────────────────────────────────────────────────────────────────
    # BOS reference: the pivot formed in the 8 bars immediately before the sweep.
    # This is the last structural support/resistance that price must close through
    # to confirm the reversal.  Using the full post-Asian window includes the
    # Asian extremes which are too far for normal displacement to reach.
    ref_slice = bars.iloc[max(asian_end_bar, sweep_bar_idx - 8):sweep_bar_idx]
    if len(ref_slice) < 2:
        ref_slice = bars.iloc[max(0, sweep_bar_idx - 8):sweep_bar_idx]

    if liq_direction == "HIGH":
        swing_ref = float(ref_slice["low"].min())
    else:
        swing_ref = float(ref_slice["high"].max())

    bos_bar_idx: Optional[int] = None
    bos_found = False

    for i in range(disp_end + 1, min(n, disp_end + 40)):   # widened: real BOS can take 2-3 hrs
        bar = bars.iloc[i]
        if liq_direction == "HIGH":
            check = float(bar["close"]) if config.bos_requires_close else float(bar["low"])
            if check < swing_ref:
                bos_bar_idx = i
                bos_found = True
                break
        else:
            check = float(bar["close"]) if config.bos_requires_close else float(bar["high"])
            if check > swing_ref:
                bos_bar_idx = i
                bos_found = True
                break

    if config.use_bos and not bos_found:
        return result

    if bos_found:
        result.bos_bar   = bos_bar_idx
        result.bos_level = round(swing_ref, 2)
        result.current_state = 4
        result.transitions.append(
            Transition(3, 4, bos_bar_idx,
                       f"BOS — {'close below' if liq_direction=='HIGH' else 'close above'} "
                       f"swing @ {swing_ref:.2f}")
        )
    else:
        # use_bos = False: advance directly from displacement
        bos_bar_idx = disp_end
        result.bos_level = round(swing_ref, 2)
        result.current_state = 4
        result.transitions.append(Transition(3, 4, disp_end, "BOS skipped (use_bos=False)"))

    # ─────────────────────────────────────────────────────────────────────────
    # STATE 4 → 5: Define entry zone geometry (Fibonacci + FVG)
    # Fibonacci anchored to the confirmed displacement leg.
    # LOCATION IS NOT CONFIRMATION — this zone is a location, not an entry.
    # ─────────────────────────────────────────────────────────────────────────
    if liq_direction == "HIGH":
        # Bearish: displacement runs DOWN from sweep close
        disp_high_px = float(bars.iloc[sweep_bar_idx]["close"])
        disp_low_px  = float(bars.iloc[disp_end]["low"])
        fib_range    = disp_high_px - disp_low_px
        # Zone: price retracing UP into [50%, 78.6%] of the DOWN leg
        fib_zone_lo = disp_low_px + fib_range * config.fib_zone_low
        fib_zone_hi = disp_low_px + fib_range * config.fib_zone_high
    else:
        # Bullish: displacement runs UP from sweep close
        disp_low_px  = float(bars.iloc[sweep_bar_idx]["close"])
        disp_high_px = float(bars.iloc[disp_end]["high"])
        fib_range    = disp_high_px - disp_low_px
        # Zone: price retracing DOWN into [50%, 78.6%] of the UP leg
        fib_zone_hi = disp_high_px - fib_range * config.fib_zone_low
        fib_zone_lo = disp_high_px - fib_range * config.fib_zone_high

    result.fib_zone_low_price  = round(fib_zone_lo, 2)
    result.fib_zone_high_price = round(fib_zone_hi, 2)

    # FVG: gap between bar before first displacement and last displacement bar
    pre_disp_idx = disp_start - 1
    if pre_disp_idx >= 0:
        pre_bar  = bars.iloc[pre_disp_idx]
        post_bar = bars.iloc[disp_end]
        if liq_direction == "HIGH":
            fvg_top = float(pre_bar["low"])
            fvg_bot = float(post_bar["high"])
        else:
            fvg_bot = float(pre_bar["high"])
            fvg_top = float(post_bar["low"])
        if fvg_top < fvg_bot:
            fvg_top, fvg_bot = fvg_bot, fvg_top
        result.fvg_top = round(fvg_top, 2)
        result.fvg_bot = round(fvg_bot, 2)

    result.current_state = 5
    result.transitions.append(Transition(4, 5, bos_bar_idx, "Entry zone defined — Fibonacci + FVG computed"))
    result.current_state = 6
    result.transitions.append(Transition(5, 6, bos_bar_idx, "Monitoring for pullback into zone"))

    # ─────────────────────────────────────────────────────────────────────────
    # STATE 6 → 7: Entry — price enters zone with reaction bar
    # A reaction bar: bearish (SHORT) or bullish (LONG) candle inside the zone.
    # ─────────────────────────────────────────────────────────────────────────
    entry_bar_idx: Optional[int] = None
    entry_price: Optional[float] = None
    trade_direction: Optional[str] = None

    search_start = bos_bar_idx + 1

    for i in range(search_start, min(n, search_start + 60)):   # up to 5 hrs for retracement
        bar = bars.iloc[i]
        bar_hi = float(bar["high"])
        bar_lo = float(bar["low"])

        in_fib = True
        if config.use_fibonacci:
            in_fib = bar_hi >= fib_zone_lo and bar_lo <= fib_zone_hi

        in_fvg = True
        if config.use_fvg and result.fvg_top and result.fvg_bot:
            in_fvg = bar_hi >= result.fvg_bot and bar_lo <= result.fvg_top

        zone_touched = in_fib   # FVG improves grade but is not a mandatory gate

        if liq_direction == "HIGH":
            is_reaction = float(bar["close"]) < float(bar["open"])   # bearish reaction
        else:
            is_reaction = float(bar["close"]) > float(bar["open"])   # bullish reaction

        if zone_touched and is_reaction:
            entry_bar_idx   = i
            entry_price     = float(bar["close"])
            trade_direction = "SHORT" if liq_direction == "HIGH" else "LONG"
            break

    if entry_bar_idx is None:
        return result

    # ── Confluence scoring ────────────────────────────────────────────────────
    conf = ConfluenceScore()
    conf.asian_sweep = True   # we got here, so a sweep occurred

    if config.use_fvg and result.fvg_top and result.fvg_bot:
        conf.has_fvg = True
        bar_e = bars.iloc[entry_bar_idx]
        conf.fvg_overlap = float(bar_e["high"]) >= result.fvg_bot and float(bar_e["low"]) <= result.fvg_top

    if config.use_fibonacci:
        conf.fibonacci_zone = True   # entry was inside the Fibonacci zone by construction

    if config.use_ema20:
        ema_val = float(ema_series.iloc[entry_bar_idx])
        result.ema20_at_entry = round(ema_val, 2)
        if trade_direction == "SHORT":
            conf.ema_aligned = entry_price < ema_val
        else:
            conf.ema_aligned = entry_price > ema_val

    conf.compute_grade()

    # Grade gate
    grade_rank = {"A": 3, "B": 2, "C": 1}
    if grade_rank[conf.grade] < grade_rank[config.minimum_confluence_grade.value]:
        result.confluence = conf
        result.veto_reason = f"Grade {conf.grade} below minimum {config.minimum_confluence_grade.value}"
        return result

    result.confluence = conf

    # ── Stop-loss ─────────────────────────────────────────────────────────────
    if config.sl_method == SLMethod.STRUCTURE:
        if trade_direction == "SHORT":
            sl = sweep_extreme + config.sl_buffer_pts
        else:
            sl = sweep_extreme - config.sl_buffer_pts
    elif config.sl_method == SLMethod.FIXED:
        sl = (entry_price + config.sl_fixed_pts if trade_direction == "SHORT"
              else entry_price - config.sl_fixed_pts)
    else:   # ATR
        atr_e = max(float(atr_series.iloc[entry_bar_idx]), 0.01)
        sl = (entry_price + 1.5 * atr_e if trade_direction == "SHORT"
              else entry_price - 1.5 * atr_e)

    sl_width = abs(entry_price - sl)

    if sl_width > config.sl_max_width_pts:
        result.entry_price  = round(entry_price, 2)
        result.sl_price     = round(sl, 2)
        result.trade_direction = trade_direction
        result.confluence   = conf
        result.veto_reason  = f"Stop too wide: {sl_width:.1f} pts > max {config.sl_max_width_pts:.0f} pts — SKIP"
        return result

    # ── Take-profit ───────────────────────────────────────────────────────────
    if config.tp_method == TPMethod.BASE_HIT:
        tp = (entry_price - config.tp_base_hit_pts if trade_direction == "SHORT"
              else entry_price + config.tp_base_hit_pts)
    elif config.tp_method == TPMethod.FIXED_RR:
        tp = (entry_price - sl_width * config.rr_ratio if trade_direction == "SHORT"
              else entry_price + sl_width * config.rr_ratio)
    elif config.tp_method == TPMethod.LIQUIDITY:
        tp = asian_low if trade_direction == "SHORT" else asian_high
    else:
        tp = (entry_price - sl_width * config.rr_ratio if trade_direction == "SHORT"
              else entry_price + sl_width * config.rr_ratio)

    rr = abs(tp - entry_price) / sl_width if sl_width > 0 else 0.0

    result.trade_direction = trade_direction
    result.entry_bar   = entry_bar_idx
    result.entry_price = round(entry_price, 2)
    result.sl_price    = round(sl, 2)
    result.tp_price    = round(tp, 2)
    result.rr          = round(rr, 2)

    result.current_state = 7
    result.transitions.append(
        Transition(6, 7, entry_bar_idx,
                   f"Entry filled — {trade_direction} @ {entry_price:.2f} | "
                   f"SL {sl:.2f} | TP {tp:.2f} | {rr:.1f}R")
    )
    result.current_state = 8
    result.transitions.append(Transition(7, 8, entry_bar_idx, "Trade management active"))

    # ─────────────────────────────────────────────────────────────────────────
    # STATE 8 → 9: Exit (TP or SL hit)
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(entry_bar_idx + 1, n):
        bar = bars.iloc[i]
        if trade_direction == "SHORT":
            if float(bar["low"]) <= tp:
                result.exit_bar    = i
                result.exit_price  = tp
                result.exit_reason = "TP_HIT"
                result.pnl_pts     = round(entry_price - tp, 2)
                break
            if float(bar["high"]) >= sl:
                result.exit_bar    = i
                result.exit_price  = sl
                result.exit_reason = "SL_HIT"
                result.pnl_pts     = round(entry_price - sl, 2)
                break
        else:
            if float(bar["high"]) >= tp:
                result.exit_bar    = i
                result.exit_price  = tp
                result.exit_reason = "TP_HIT"
                result.pnl_pts     = round(tp - entry_price, 2)
                break
            if float(bar["low"]) <= sl:
                result.exit_bar    = i
                result.exit_price  = sl
                result.exit_reason = "SL_HIT"
                result.pnl_pts     = round(sl - entry_price, 2)
                break

    if result.exit_bar is not None:
        result.current_state = 9
        result.transitions.append(
            Transition(8, 9, result.exit_bar,
                       f"{result.exit_reason} @ {result.exit_price:.2f} | "
                       f"P&L: {result.pnl_pts:+.2f} pts")
        )

    return result
