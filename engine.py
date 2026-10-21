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
class NarrativeEvent:
    bar_idx: int
    status: str          # "info" | "pass" | "fail" | "wait" | "scan" | "entry" | "exit_tp" | "exit_sl"
    headline: str
    details: list[str] = field(default_factory=list)


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
    narrative: list[NarrativeEvent] = field(default_factory=list)
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

    # Narrative helper — appends a plain-English event at each decision point
    _n: list[NarrativeEvent] = []
    def _N(bar_idx: int, status: str, headline: str, *details: str) -> None:
        _n.append(NarrativeEvent(bar_idx, status, headline, list(details)))

    # ─────────────────────────────────────────────────────────────────────────
    # STATE 0 → 1: Asian session must be complete (bar 47 closed = bar index 47)
    # ─────────────────────────────────────────────────────────────────────────
    if reveal_idx < asian_end_bar:
        result.current_state = 0
        _N(reveal_idx, "wait", "Asian session in progress — building the range",
           f"Need {asian_end_bar - reveal_idx} more bars until midnight NY",
           "Liquidity levels cannot be set until the Asian session closes")
        result.narrative = _n
        return result

    asian_bars = bars.iloc[:asian_end_bar]
    asian_high = float(asian_bars["high"].max())
    asian_low  = float(asian_bars["low"].min())
    asian_range = asian_high - asian_low
    dp = config.price_decimals

    result.asian_high = round(asian_high, dp)
    result.asian_low  = round(asian_low,  dp)
    result.current_state = 1
    result.transitions.append(Transition(0, 1, asian_end_bar - 1, "Asian session closed — PDH/PDL and Asian H/L usable"))
    _N(asian_end_bar - 1, "info",
       f"Asian range locked — High {asian_high:.{dp}f}  ·  Low {asian_low:.{dp}f}",
       f"Range: {asian_range:.{dp}f} pts across {asian_end_bar} bars (20:00 → 00:00 NY)",
       f"Watching for a wick above {asian_high:.{dp}f} (bearish setup) or below {asian_low:.{dp}f} (bullish setup)",
       "A valid sweep must close back on the originating side")

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
    skipped_sweeps = 0

    for cand_idx, cand_dir, cand_extreme in sweep_candidates:
        pre_atr = compute_atr(bars.iloc[:cand_idx])
        pre_atr_val = max(float(pre_atr.iloc[-1]) if len(pre_atr) > 0 else 1e-6, 1e-10)
        exp_dir = "DOWN" if cand_dir == "HIGH" else "UP"
        atr_thresh = config.displacement_body_atr_multiplier * pre_atr_val

        cand_disp: list[int] = []
        best_body = 0.0
        best_ratio = 0.0
        for j in range(cand_idx + 1, min(n, cand_idx + 10)):
            bar = bars.iloc[j]
            body = abs(float(bar["close"]) - float(bar["open"]))
            bar_range = float(bar["high"]) - float(bar["low"])
            body_ratio = body / bar_range if bar_range > 0 else 0
            best_body = max(best_body, body)
            best_ratio = max(best_ratio, body_ratio)
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
        else:
            skipped_sweeps += 1
            _N(cand_idx, "scan",
               f"Sweep {'above' if cand_dir=='HIGH' else 'below'} "
               f"{'Asian High' if cand_dir=='HIGH' else 'Asian Low'} @ {cand_extreme:.2f} — no displacement",
               f"Checked {min(n, cand_idx + 10) - cand_idx - 1} bars after the sweep",
               f"Best bar body: {best_body:.2f} pts vs threshold {atr_thresh:.2f} pts  "
               f"({'✓' if best_body >= atr_thresh else '✗'})",
               f"Best body/range ratio: {best_ratio:.0%} vs minimum {config.displacement_body_range_ratio_min:.0%}  "
               f"({'✓' if best_ratio >= config.displacement_body_range_ratio_min else '✗'})",
               "No strong directional follow-through — skipping this sweep, continuing scan")

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
        _N(last[0], "fail",
           f"Scanned {len(sweep_candidates)} sweep candidate(s) — none produced displacement",
           "No valid setup found in this session",
           "The Asian High and Low were tested but price never displaced convincingly")
        result.narrative = _n
        return result

    # Successful sweep + displacement found
    side = "above Asian High" if liq_direction == "HIGH" else "below Asian Low"
    _N(sweep_bar_idx, "pass",
       f"Sweep confirmed — wick {side} @ {sweep_extreme:.2f}",
       f"{'High' if liq_direction=='HIGH' else 'Low'} of the sweep bar: {sweep_extreme:.2f}  "
       f"vs Asian {'High' if liq_direction=='HIGH' else 'Low'}: {asian_high if liq_direction=='HIGH' else asian_low:.2f}",
       "Bar closed back — liquidity has been taken",
       f"Now demanding {'bearish' if liq_direction=='HIGH' else 'bullish'} displacement"
       + (f"  (skipped {skipped_sweeps} earlier sweep(s) with no follow-through)" if skipped_sweeps else ""))

    result.liq_direction = liq_direction
    result.sweep_bar     = sweep_bar_idx
    result.sweep_extreme = round(sweep_extreme, dp)
    result.current_state = 2
    result.transitions.append(
        Transition(1, 2, sweep_bar_idx,
                   f"Sweep — wick {'above Asian High' if liq_direction=='HIGH' else 'below Asian Low'} "
                   f"@ {sweep_extreme:.{dp}f}, bar closes back")
    )

    disp_start   = disp_bars[0]
    disp_end     = disp_bars[-1]
    expected_dir = "DOWN" if liq_direction == "HIGH" else "UP"
    pre_atr_final = compute_atr(bars.iloc[:sweep_bar_idx])
    pre_atr_val_final = max(float(pre_atr_final.iloc[-1]) if len(pre_atr_final) > 0 else 1e-6, 1e-10)
    disp_body_sizes = [
        abs(float(bars.iloc[i]["close"]) - float(bars.iloc[i]["open"]))
        for i in disp_bars
    ]
    _N(disp_start, "pass",
       f"Displacement confirmed — {len(disp_bars)} strong {'bearish' if expected_dir=='DOWN' else 'bullish'} bar(s)",
       f"Bar bodies: {', '.join(f'{b:.2f}' for b in disp_body_sizes)} pts  "
       f"(threshold: {config.displacement_body_atr_multiplier * pre_atr_val_final:.2f} pts ✓)",
       f"Body/range ratio ≥ {config.displacement_body_range_ratio_min:.0%} on each bar ✓",
       "Price is moving impulsively in the expected direction — now looking for Break of Structure")

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
    check_word = "Close" if config.bos_requires_close else "Wick"

    _N(disp_end + 1, "wait",
       f"Watching for Break of Structure — swing reference: {swing_ref:.2f}",
       f"The displacement must {'close' if config.bos_requires_close else 'wick'} "
       f"{'below' if liq_direction=='HIGH' else 'above'} {swing_ref:.2f}",
       f"(Swing ref = {'lowest low' if liq_direction=='HIGH' else 'highest high'} "
       f"of the 8 bars immediately before the sweep)",
       "BOS confirms that structure has genuinely shifted — not just a spike and retrace")

    for i in range(disp_end + 1, min(n, disp_end + config.bos_search_bars)):
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
        _N(min(n - 1, disp_end + config.bos_search_bars), "fail",
           f"Break of Structure not found — setup abandoned",
           f"Needed a close {'below' if liq_direction=='HIGH' else 'above'} {swing_ref:.2f}",
           f"Searched {min(n, disp_end + 40) - disp_end - 1} bars after displacement",
           "Price displaced but never convincingly broke the pre-sweep structure",
           "No trade taken")
        result.narrative = _n
        return result

    if bos_found:
        bos_bar_close = float(bars.iloc[bos_bar_idx]["close"])
        _N(bos_bar_idx, "pass",
           f"Break of Structure — {'close below' if liq_direction=='HIGH' else 'close above'} {swing_ref:.2f}",
           f"{check_word} value: {bos_bar_close:.2f}  vs swing ref {swing_ref:.2f}  "
           f"({'✓' if (liq_direction=='HIGH' and bos_bar_close < swing_ref) or (liq_direction=='LOW' and bos_bar_close > swing_ref) else '✗'})",
           "Structure is broken — the market has confirmed the directional intent",
           "Now mapping the Fibonacci retracement zone and Fair Value Gap")
        result.bos_bar   = bos_bar_idx
        result.bos_level = round(swing_ref, dp)
        result.current_state = 4
        result.transitions.append(
            Transition(3, 4, bos_bar_idx,
                       f"BOS — {'close below' if liq_direction=='HIGH' else 'close above'} "
                       f"swing @ {swing_ref:.{dp}f}")
        )
    else:
        bos_bar_idx = disp_end
        result.bos_level = round(swing_ref, dp)
        result.current_state = 4
        result.transitions.append(Transition(3, 4, disp_end, "BOS skipped (use_bos=False)"))
        _N(disp_end, "info", "BOS check skipped (use_bos=False)", "Advancing from displacement directly")

    # ─────────────────────────────────────────────────────────────────────────
    # STATE 4 → 5: Define entry zone geometry (Fibonacci + FVG)
    # Fibonacci anchored to the confirmed displacement leg.
    # LOCATION IS NOT CONFIRMATION — this zone is a location, not an entry.
    # ─────────────────────────────────────────────────────────────────────────
    if liq_direction == "HIGH":
        # Bearish: anchor fib from the absolute sweep extreme (the wick high) to the displacement low.
        # Using the full swept-leg rather than just the sweep close gives a properly-sized zone.
        disp_low_px = float(bars.iloc[disp_end]["low"])
        fib_range   = sweep_extreme - disp_low_px
        # Zone: price retracing UP into [50%, 78.6%] of the full swept leg
        fib_zone_lo = disp_low_px + fib_range * config.fib_zone_low
        fib_zone_hi = disp_low_px + fib_range * config.fib_zone_high
    else:
        # Bullish: anchor fib from the displacement high down to the absolute sweep extreme (the wick low)
        disp_high_px = float(bars.iloc[disp_end]["high"])
        fib_range    = disp_high_px - sweep_extreme
        # Zone: price retracing DOWN into [50%, 78.6%] of the full swept leg
        fib_zone_hi = disp_high_px - fib_range * config.fib_zone_low
        fib_zone_lo = disp_high_px - fib_range * config.fib_zone_high

    result.fib_zone_low_price  = round(fib_zone_lo, dp)
    result.fib_zone_high_price = round(fib_zone_hi, dp)

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
        result.fvg_top = round(fvg_top, dp)
        result.fvg_bot = round(fvg_bot, dp)

    result.current_state = 5
    result.transitions.append(Transition(4, 5, bos_bar_idx, "Entry zone defined — Fibonacci + FVG computed"))
    result.current_state = 6
    result.transitions.append(Transition(5, 6, bos_bar_idx, "Monitoring for pullback into zone"))

    fvg_detail = (f"Fair Value Gap: {result.fvg_bot:.2f} – {result.fvg_top:.2f}  "
                  f"(overlap with Fib zone improves grade)"
                  if result.fvg_top else "No Fair Value Gap identified in displacement")
    trade_dir_label = "SELL" if liq_direction == "HIGH" else "BUY"
    _N(bos_bar_idx, "info",
       f"Entry zone mapped — {trade_dir_label} area: {fib_zone_lo:.2f} – {fib_zone_hi:.2f}",
       f"Fibonacci 50%–78.6% anchored sweep extreme {sweep_extreme:.2f} → disp end: {fib_zone_lo:.2f} – {fib_zone_hi:.2f}  ({fib_range:.2f} pt range)",
       fvg_detail,
       f"LOCATION IS NOT CONFIRMATION — waiting for price to pull back AND show a reaction candle")
    _N(bos_bar_idx + 1, "wait",
       f"Monitoring for retracement into {fib_zone_lo:.2f} – {fib_zone_hi:.2f}",
       "Do not chase the move — the entry window opens only when price returns to the zone",
       f"A {'bearish' if liq_direction=='HIGH' else 'bullish'} reaction candle inside the zone is required")

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
        bars_searched = min(n, search_start + 60) - search_start
        _N(min(n - 1, search_start + 60), "fail",
           "No confirmation in zone — setup expired",
           f"Searched {bars_searched} bars ({bars_searched * 5} min) after BOS for a reaction candle",
           f"Price {'never retraced into' if True else 'entered'} the zone {fib_zone_lo:.2f}–{fib_zone_hi:.2f}",
           "Everything passed up to this point — the pullback or confirmation never came",
           "No trade taken")
        result.narrative = _n
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

    # Confluence narrative
    conf_items = []
    if conf.asian_sweep:   conf_items.append("Asian sweep ✓")
    if conf.fibonacci_zone: conf_items.append("Fibonacci zone ✓")
    if conf.has_fvg:       conf_items.append("FVG present ✓")
    if conf.fvg_overlap:   conf_items.append("FVG overlaps entry zone ✓")
    if conf.ema_aligned:   conf_items.append("EMA20 aligned ✓")
    _N(entry_bar_idx, "info",
       f"Confluence check — Grade {conf.grade}  ({conf.score}/{6} items)",
       "  ·  ".join(conf_items) if conf_items else "No confluence items scored",
       f"Minimum required grade: {config.minimum_confluence_grade.value}")

    # Grade gate
    grade_rank = {"A": 3, "B": 2, "C": 1}
    if grade_rank[conf.grade] < grade_rank[config.minimum_confluence_grade.value]:
        result.confluence = conf
        result.veto_reason = f"Grade {conf.grade} below minimum {config.minimum_confluence_grade.value}"
        _N(entry_bar_idx, "fail",
           f"Setup vetoed — confluence grade too low",
           f"Got Grade {conf.grade}, need at least Grade {config.minimum_confluence_grade.value}",
           "The structure passed but not enough supporting factors aligned",
           "No trade taken")
        result.narrative = _n
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
        result.entry_price  = round(entry_price, dp)
        result.sl_price     = round(sl, dp)
        result.trade_direction = trade_direction
        result.confluence   = conf
        result.veto_reason  = f"Stop too wide: {sl_width:.1f} pts > max {config.sl_max_width_pts:.0f} pts — SKIP"
        _N(entry_bar_idx, "fail",
           f"Setup vetoed — structural stop too wide",
           f"Stop would be at {sl:.2f} — that's {sl_width:.1f} pts from entry {entry_price:.2f}",
           f"Maximum allowed stop width: {config.sl_max_width_pts:.0f} pts",
           "Never tighten the stop to force the trade — if the structure is too wide, skip it",
           "No trade taken")
        result.narrative = _n
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
    tp_pts = abs(tp - entry_price)
    dollar_risk = 100.0
    units = dollar_risk / sl_width if sl_width > 0 else 0.0

    result.trade_direction = trade_direction
    result.entry_bar   = entry_bar_idx
    result.entry_price = round(entry_price, dp)
    result.sl_price    = round(sl, dp)
    result.tp_price    = round(tp, dp)
    result.rr          = round(rr, 2)

    _N(entry_bar_idx, "entry",
       f"Entry filled — {trade_direction} @ {entry_price:.{dp}f}",
       f"Stop Loss: {sl:.{dp}f}  ({sl_width:.{dp}f} pts {'above' if trade_direction=='SHORT' else 'below'} entry — at sweep extreme + buffer)",
       f"Take Profit: {tp:.{dp}f}  ({tp_pts:.{dp}f} pts {'below' if trade_direction=='SHORT' else 'above'} entry)  |  R:R: {rr:.1f}R",
       f"Position size: {units:.1f} units  |  Risk: $100 of $10,000 account  |  Grade: {conf.grade}",
       "Trade is live — monitoring for TP or SL")

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
        dollar_pnl = units * result.pnl_pts
        if result.exit_reason == "TP_HIT":
            _N(result.exit_bar, "exit_tp",
               f"Target hit @ {result.exit_price:.2f}  —  +{result.pnl_pts:.2f} pts  /  +${dollar_pnl:.2f}",
               f"Price reached the take-profit level {result.exit_price:.2f}",
               f"Trade duration: {result.exit_bar - entry_bar_idx} bars ({(result.exit_bar - entry_bar_idx) * 5} min)",
               f"P&L: +{result.pnl_pts:.2f} pts  ·  {result.rr:.1f}R  ·  +${dollar_pnl:.2f} on $100 risk")
        else:
            _N(result.exit_bar, "exit_sl",
               f"Stop hit @ {result.exit_price:.2f}  —  {result.pnl_pts:.2f} pts  /  −$100",
               f"Price reached the stop level {result.exit_price:.2f}",
               f"Trade duration: {result.exit_bar - entry_bar_idx} bars ({(result.exit_bar - entry_bar_idx) * 5} min)",
               f"P&L: {result.pnl_pts:.2f} pts  ·  −$100 on $100 risk  (max loss enforced)")
    else:
        _N(n - 1, "wait",
           "Trade still open — no TP or SL hit yet in this data window",
           f"Holding {trade_direction} from {entry_price:.2f}  |  SL: {sl:.2f}  |  TP: {tp:.2f}")

    result.narrative = _n
    return result
