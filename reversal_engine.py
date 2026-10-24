"""HOD/LOD Reversal Engine — sweep → rejection → BOS → retest → basket entry.

Kirk's rules specify 3-minute bars; yfinance minimum is 5 minutes.
We use 5m for the trigger and 15m for context throughout.
All parameters are starting points for paper-trading research only.

Position sizing: XAUUSD — 1 standard lot = 100 oz.
  P&L per point = lots × 100  →  risk_usd = sl_pts × total_lots × 100
  Total basket risk = 1R (never multiplied by ticket count).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

__version__ = "1.0.0"

# ── State labels ──────────────────────────────────────────────────────────────
REV_STATES = {
    0: "WATCHING",
    1: "SWEEP DETECTED",
    2: "REJECTED — AWAITING BOS",
    3: "BOS CONFIRMED — AWAITING RETEST",
    4: "IN BASKET",
    5: "PARTIAL EXIT (TP1)",
    6: "CLOSED",
}

SHORT_LEVEL_KEYS = ("yesterday_high", "asian_high", "london_high", "session_high")
LONG_LEVEL_KEYS  = ("yesterday_low",  "asian_low",  "london_low",  "session_low")

LEVEL_LABEL: dict[str, str] = {
    "yesterday_high": "Yesterday H",
    "yesterday_low":  "Yesterday L",
    "asian_high":     "Asian H",
    "asian_low":      "Asian L",
    "london_high":    "London H",
    "london_low":     "London L",
    "session_high":   "Session H",
    "session_low":    "Session L",
}


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class RevEvent:
    bar_idx: int
    status: str        # info | pass | fail | wait | scan | entry | exit_tp | exit_sl
    headline: str
    details: list[str] = field(default_factory=list)


@dataclass
class Ticket:
    """One entry in the basket. First half exits at TP1, second half at TP2."""
    lots:        float
    entry:       float
    sl:          float
    tp1:         float
    tp2:         float
    tp_target:   str  = "TP1"   # which target this ticket aims for
    status:      str  = "OPEN"  # OPEN | TP1 | TP2 | SL
    exit_bar:    int  = -1
    exit_price:  float = 0.0
    pnl_usd:     float = 0.0


@dataclass
class ReversalResult:
    current_state: int = 0
    narrative: list[RevEvent] = field(default_factory=list)

    direction: str = "SHORT"

    # Swept level
    swept_level_price: Optional[float] = None
    swept_level_key:   Optional[str]   = None
    sweep_bar:         Optional[int]   = None
    sweep_extreme:     Optional[float] = None
    rejection_bar:     Optional[int]   = None

    # Structure
    swing_ref_price: Optional[float] = None
    swing_ref_bar:   Optional[int]   = None
    bos_bar:         Optional[int]   = None

    # Retest / entry zone
    retest_level:     Optional[float] = None
    retest_tolerance: Optional[float] = None

    # Entry
    entry_bar:    Optional[int]   = None
    entry_price:  Optional[float] = None
    sl_price:     Optional[float] = None
    tp1_price:    Optional[float] = None
    tp2_price:    Optional[float] = None
    sl_pts:       float = 0.0
    rr_available: float = 0.0

    # Basket
    tickets:         list[Ticket] = field(default_factory=list)
    n_tickets:       int   = 0
    total_lots:      float = 0.0
    basket_risk_usd: float = 0.0

    # P&L
    realized_pnl_usd: float = 0.0
    open_pnl_usd:     float = 0.0

    skip_reason: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, cp = bars["high"], bars["low"], bars["close"].shift(1)
    tr = pd.concat([(h - l), (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _median_body(bars: pd.DataFrame, n: int = 20) -> float:
    n = min(n, len(bars))
    if n == 0:
        return 1e-9
    bodies = sorted(
        abs(float(bars.iloc[i]["close"]) - float(bars.iloc[i]["open"]))
        for i in range(len(bars) - n, len(bars))
    )
    mid = len(bodies) // 2
    return (bodies[mid - 1] + bodies[mid]) / 2 if len(bodies) % 2 == 0 else bodies[mid]


def _last_swing(
    bars: pd.DataFrame,
    before_bar: int,
    side: str,           # "LOW" | "HIGH"
    lookback: int = 80,
) -> tuple[int, float] | tuple[None, None]:
    """Last confirmed swing with 2 bars each side, fully confirmed before before_bar."""
    start = min(before_bar - 3, len(bars) - 3)
    for i in range(start, 1, -1):
        if i + 2 >= before_bar:
            continue
        if before_bar - i > lookback:
            break
        if side == "LOW":
            v = float(bars.iloc[i]["low"])
            if (v < float(bars.iloc[i - 1]["low"]) and v < float(bars.iloc[i - 2]["low"]) and
                    v < float(bars.iloc[i + 1]["low"]) and v < float(bars.iloc[i + 2]["low"])):
                return i, v
        else:
            v = float(bars.iloc[i]["high"])
            if (v > float(bars.iloc[i - 1]["high"]) and v > float(bars.iloc[i - 2]["high"]) and
                    v > float(bars.iloc[i + 1]["high"]) and v > float(bars.iloc[i + 2]["high"])):
                return i, v
    return None, None


def _nearest_opposing(key_levels: dict[str, float], entry: float, direction: str) -> float | None:
    if direction == "SHORT":
        candidates = [v for k, v in key_levels.items() if k in LONG_LEVEL_KEYS and v < entry]
        return max(candidates) if candidates else None
    else:
        candidates = [v for k, v in key_levels.items() if k in SHORT_LEVEL_KEYS and v > entry]
        return min(candidates) if candidates else None


def _build_basket(
    entry: float, sl: float, tp1: float, tp2: float,
    equity: float, risk_pct: float, max_pos: int, spread_pts: float,
    direction: str,
) -> list[Ticket]:
    """Compute XAUUSD basket.  P&L = pts × lots × 100.  Total risk ≤ equity × risk_pct."""
    risk_usd = equity * risk_pct
    sl_pts   = abs(entry - sl) + spread_pts
    if sl_pts <= 0:
        return []
    total_lots = risk_usd / (sl_pts * 100.0)
    step, minimum = 0.01, 0.01
    per_pos = int(total_lots / max_pos / step) * step
    if per_pos < minimum:
        per_pos = minimum
        n = max(1, int(total_lots / minimum))
    else:
        n = max_pos
    n = min(n, max_pos)

    n_tp1 = (n + 1) // 2   # first half → TP1
    tickets = []
    for idx in range(n):
        target = "TP1" if idx < n_tp1 else "TP2"
        tickets.append(Ticket(lots=per_pos, entry=entry, sl=sl,
                               tp1=tp1, tp2=tp2, tp_target=target))
    return tickets


# ── Main engine ───────────────────────────────────────────────────────────────
def run_reversal_engine(
    df_5m: pd.DataFrame,
    key_levels: dict[str, float],
    reveal_idx: int,
    account_equity: float = 10_000.0,
    risk_pct: float = 0.01,
    max_positions: int = 8,
    sweep_close_bars: int = 3,
    disp_body_mult: float = 1.5,
    retest_atr_mult: float = 0.10,
    retest_wait_bars: int = 5,
    sl_atr_mult: float = 0.20,
    min_rr: float = 2.0,
    tp1_r: float = 2.0,
    spread_pts: float = 0.30,
) -> ReversalResult:
    result = ReversalResult()
    if reveal_idx < 0 or df_5m.empty:
        return result

    bars = df_5m.iloc[: reveal_idx + 1].copy().reset_index(drop=True)
    n = len(bars)
    if n < 20:
        return result

    atr_s = _atr(bars)
    _ev: list[RevEvent] = []

    def _E(bar: int, status: str, headline: str, *details: str) -> None:
        _ev.append(RevEvent(bar, status, headline, list(details)))

    level_summary = "  ·  ".join(
        f"{LEVEL_LABEL.get(k, k)}: {v:.2f}" for k, v in key_levels.items()
    )
    _E(0, "info", f"Watching {len(key_levels)} key level(s)", level_summary)

    # FSM variables
    state           = 0
    sweep_bar_i:    Optional[int]   = None
    swept_key:      Optional[str]   = None
    swept_price:    Optional[float] = None
    sweep_extreme:  Optional[float] = None
    sweep_dir:      str             = "SHORT"
    rejection_bar:  Optional[int]   = None
    swing_ref_bar:  Optional[int]   = None
    swing_ref_px:   Optional[float] = None
    bos_bar_i:      Optional[int]   = None
    atr_at_bos:     float           = 1.0
    retest_lvl:     Optional[float] = None
    retest_tol:     Optional[float] = None
    retest_start:   Optional[int]   = None

    i = 0
    while i <= reveal_idx and i < n:
        bar = bars.iloc[i]
        h = float(bar["high"]); l = float(bar["low"])
        c = float(bar["close"]); o = float(bar["open"])
        atr_v = max(float(atr_s.iloc[i]), 0.01)

        # ── 0: WATCHING ───────────────────────────────────────────────────
        if state == 0:
            for lk in SHORT_LEVEL_KEYS:
                lv = key_levels.get(lk)
                if lv and h > lv:
                    sweep_bar_i = i; swept_key = lk; swept_price = lv
                    sweep_extreme = h; sweep_dir = "SHORT"; state = 1
                    _E(i, "pass", f"Sweep above {LEVEL_LABEL[lk]} @ {lv:.2f}",
                       f"Bar high: {h:.2f}  ·  Need close back below {lv:.2f} within {sweep_close_bars} bar(s)")
                    break
            if state == 0:
                for lk in LONG_LEVEL_KEYS:
                    lv = key_levels.get(lk)
                    if lv and l < lv:
                        sweep_bar_i = i; swept_key = lk; swept_price = lv
                        sweep_extreme = l; sweep_dir = "LONG"; state = 1
                        _E(i, "pass", f"Sweep below {LEVEL_LABEL[lk]} @ {lv:.2f}",
                           f"Bar low: {l:.2f}  ·  Need close back above {lv:.2f} within {sweep_close_bars} bar(s)")
                        break

        # ── 1: SWEEP DETECTED — awaiting rejection ────────────────────────
        elif state == 1:
            bars_since = i - sweep_bar_i
            if sweep_dir == "SHORT":
                if h > sweep_extreme:
                    sweep_extreme = h  # update to latest wick high
                if c < swept_price:
                    srb, srp = _last_swing(bars, sweep_bar_i, "LOW")
                    if srb is None:
                        _E(i, "fail", "Rejected but no confirmed swing low found before sweep — skip")
                        result.skip_reason = "no swing low before sweep"
                        state = 0
                    else:
                        rejection_bar = i; swing_ref_bar = srb; swing_ref_px = srp
                        state = 2
                        _E(i, "pass", f"Rejection — closed below {swept_price:.2f}",
                           f"Sweep extreme: {sweep_extreme:.2f}",
                           f"Swing low reference: {srp:.2f} (bar {srb})",
                           f"Need bearish displacement closing below {srp:.2f}")
                elif bars_since >= sweep_close_bars:
                    _E(i, "fail", f"No close-back within {sweep_close_bars} bars — sweep failed",
                       f"Level {swept_price:.2f} held above; skipping")
                    result.skip_reason = f"no rejection within {sweep_close_bars} bars"
                    state = 0
            else:  # LONG
                if l < sweep_extreme:
                    sweep_extreme = l
                if c > swept_price:
                    srb, srp = _last_swing(bars, sweep_bar_i, "HIGH")
                    if srb is None:
                        _E(i, "fail", "Rejected but no confirmed swing high — skip")
                        result.skip_reason = "no swing high before sweep"
                        state = 0
                    else:
                        rejection_bar = i; swing_ref_bar = srb; swing_ref_px = srp
                        state = 2
                        _E(i, "pass", f"Rejection — closed above {swept_price:.2f}",
                           f"Swing high reference: {srp:.2f}  ·  Need bullish displacement closing above it")
                elif bars_since >= sweep_close_bars:
                    result.skip_reason = "no rejection"
                    state = 0

        # ── 2: REJECTED — awaiting displacement + BOS ─────────────────────
        elif state == 2:
            body      = abs(c - o)
            bar_range = h - l
            med       = _median_body(bars.iloc[:i], 20)

            if sweep_dir == "SHORT":
                if h > sweep_extreme + 0.5:
                    _E(i, "fail", f"New higher high {h:.2f} — setup cancelled, watching for fresh sweep")
                    result.skip_reason = "new higher high after rejection"
                    state = 0; i += 1; continue
                in_bot  = bar_range > 0 and c <= l + 0.25 * bar_range
                if (c < o and body >= disp_body_mult * med and in_bot and
                        swing_ref_px is not None and c < swing_ref_px):
                    bos_bar_i = i; atr_at_bos = atr_v
                    retest_lvl = swing_ref_px
                    retest_tol = retest_atr_mult * atr_v
                    retest_start = i; state = 3
                    _E(i, "pass", f"Displacement + BOS — closed below swing low {swing_ref_px:.2f}",
                       f"Body: {body:.2f} pts ≥ {disp_body_mult:.1f}× median {med:.2f} pts ✓",
                       f"Close in bottom quarter: {c:.2f} ✓",
                       f"Retest zone: {retest_lvl:.2f} ± {retest_tol:.2f}  ·  window: {retest_wait_bars} bars")
            else:
                if l < sweep_extreme - 0.5:
                    result.skip_reason = "new lower low after rejection"
                    state = 0; i += 1; continue
                in_top = bar_range > 0 and c >= h - 0.25 * bar_range
                if (c > o and body >= disp_body_mult * med and in_top and
                        swing_ref_px is not None and c > swing_ref_px):
                    bos_bar_i = i; atr_at_bos = atr_v
                    retest_lvl = swing_ref_px
                    retest_tol = retest_atr_mult * atr_v
                    retest_start = i; state = 3
                    _E(i, "pass", f"Displacement + BOS — closed above swing high {swing_ref_px:.2f}",
                       f"Body: {body:.2f} ≥ {disp_body_mult:.1f}× median {med:.2f} ✓",
                       f"Retest zone: {retest_lvl:.2f} ± {retest_tol:.2f}  ·  window: {retest_wait_bars} bars")

        # ── 3: BOS CONFIRMED — awaiting retest ───────────────────────────
        elif state == 3:
            elapsed = i - retest_start

            if sweep_dir == "SHORT":
                if h > sweep_extreme + 0.5:
                    _E(i, "fail", "New higher high after BOS — cancelled; need fresh sweep")
                    result.skip_reason = "new higher high after BOS"
                    state = 0; i += 1; continue

                in_zone  = h >= retest_lvl - retest_tol
                confirms = c < retest_lvl  # bearish close below broken swing
                if in_zone and confirms:
                    entry_px  = c
                    sl_raw    = sweep_extreme + sl_atr_mult * atr_at_bos
                    sl_pts_v  = abs(entry_px - sl_raw) + spread_pts
                    tp1_px    = entry_px - tp1_r * sl_pts_v
                    tp2_raw   = _nearest_opposing(key_levels, entry_px, "SHORT")
                    tp2_px    = tp2_raw if tp2_raw else entry_px - 3 * sl_pts_v
                    rr_avail  = (entry_px - tp2_px) / sl_pts_v if sl_pts_v > 0 else 0

                    if rr_avail < min_rr:
                        _E(i, "fail", f"Vetoed — RR {rr_avail:.1f}R < minimum {min_rr}R",
                           f"Nearest opposing liquidity {tp2_px:.2f} is too close to entry {entry_px:.2f}")
                        result.skip_reason = f"insufficient RR ({rr_avail:.1f}R)"
                        result.entry_price = round(entry_px, 2)
                        result.sl_price    = round(sl_raw, 2)
                        result.tp1_price   = round(tp1_px, 2)
                        result.tp2_price   = round(tp2_px, 2)
                        result.rr_available = round(rr_avail, 2)
                        break

                    tix = _build_basket(entry_px, sl_raw, tp1_px, tp2_px,
                                        account_equity, risk_pct, max_positions,
                                        spread_pts, "SHORT")
                    _set_entry(result, "SHORT", i, entry_px, sl_raw, tp1_px,
                               tp2_px, sl_pts_v, rr_avail, tix)
                    state = 4
                    _E(i, "entry",
                       f"Entry — SHORT @ {entry_px:.2f}  ·  {len(tix)} tickets × {tix[0].lots if tix else 0} lot",
                       f"SL: {sl_raw:.2f}  ({sl_pts_v:.2f} pts)  ·  TP1: {tp1_px:.2f} (2R)  ·  TP2: {tp2_px:.2f}",
                       f"Total lots: {result.total_lots}  ·  Basket risk: ${result.basket_risk_usd:.2f}",
                       f"RR to TP2: {rr_avail:.1f}R  ·  Exit: {(len(tix)+1)//2} tickets @ TP1, {len(tix)//2} @ TP2")

                elif elapsed >= retest_wait_bars:
                    _E(i, "fail", f"No retest in {retest_wait_bars} bars — setup expired",
                       f"Price never returned to {retest_lvl:.2f} ± {retest_tol:.2f}")
                    result.skip_reason = "retest window expired"
                    state = 0
                else:
                    _E(i, "wait", f"Awaiting retest ({elapsed + 1}/{retest_wait_bars})",
                       f"Need bar.high ≥ {retest_lvl - retest_tol:.2f} then bearish close < {retest_lvl:.2f}")

            else:  # LONG retest
                if l < sweep_extreme - 0.5:
                    result.skip_reason = "new lower low after BOS"
                    state = 0; i += 1; continue
                in_zone  = l <= retest_lvl + retest_tol
                confirms = c > retest_lvl
                if in_zone and confirms:
                    entry_px  = c
                    sl_raw    = sweep_extreme - sl_atr_mult * atr_at_bos
                    sl_pts_v  = abs(entry_px - sl_raw) + spread_pts
                    tp1_px    = entry_px + tp1_r * sl_pts_v
                    tp2_raw   = _nearest_opposing(key_levels, entry_px, "LONG")
                    tp2_px    = tp2_raw if tp2_raw else entry_px + 3 * sl_pts_v
                    rr_avail  = (tp2_px - entry_px) / sl_pts_v if sl_pts_v > 0 else 0
                    if rr_avail < min_rr:
                        result.skip_reason = f"insufficient RR ({rr_avail:.1f}R)"
                        _E(i, "fail", f"Vetoed — RR {rr_avail:.1f}R < {min_rr}R")
                        break
                    tix = _build_basket(entry_px, sl_raw, tp1_px, tp2_px,
                                        account_equity, risk_pct, max_positions,
                                        spread_pts, "LONG")
                    _set_entry(result, "LONG", i, entry_px, sl_raw, tp1_px,
                               tp2_px, sl_pts_v, rr_avail, tix)
                    state = 4
                    _E(i, "entry",
                       f"Entry — LONG @ {entry_px:.2f}  ·  {len(tix)} tickets",
                       f"SL: {sl_raw:.2f}  ·  TP1: {tp1_px:.2f}  ·  TP2: {tp2_px:.2f}",
                       f"Total lots: {result.total_lots}  ·  Basket risk: ${result.basket_risk_usd:.2f}")
                elif elapsed >= retest_wait_bars:
                    result.skip_reason = "retest window expired"
                    state = 0
                else:
                    _E(i, "wait", f"Awaiting retest ({elapsed + 1}/{retest_wait_bars})")

        # ── 4/5: IN BASKET ────────────────────────────────────────────────
        elif state in (4, 5):
            _process_basket(result, bars, i)
            n_open  = sum(1 for t in result.tickets if t.status == "OPEN")
            n_tp1_h = sum(1 for t in result.tickets if t.status == "TP1")
            realized = sum(t.pnl_usd for t in result.tickets if t.status != "OPEN")
            cur = float(bars.iloc[i]["close"])
            unrealized = sum(
                ((t.entry - cur) if result.direction == "SHORT" else (cur - t.entry)) * t.lots * 100
                for t in result.tickets if t.status == "OPEN"
            )
            result.realized_pnl_usd = round(realized, 2)
            result.open_pnl_usd     = round(unrealized, 2)
            if n_open == 0:
                state = 6
                _E(i, "exit_tp" if realized > 0 else "exit_sl",
                   f"Basket fully closed  ·  Total P&L: ${realized:+.2f}")
            elif n_tp1_h > 0 and state == 4:
                state = 5
                _E(i, "exit_tp",
                   f"Partial exit — {n_tp1_h} ticket(s) hit TP1  ·  ${realized:+.2f} locked",
                   f"{n_open} ticket(s) remain, watching TP2 @ {result.tp2_price:.2f}")

        i += 1

    result.current_state = state
    result.narrative     = _ev

    # Populate display fields from FSM variables
    if sweep_bar_i is not None:
        result.swept_level_price = swept_price
        result.swept_level_key   = swept_key
        result.sweep_bar         = sweep_bar_i
        result.sweep_extreme     = sweep_extreme
        result.rejection_bar     = rejection_bar
        result.direction         = sweep_dir
    if swing_ref_bar is not None:
        result.swing_ref_price = swing_ref_px
        result.swing_ref_bar   = swing_ref_bar
    if bos_bar_i is not None:
        result.bos_bar         = bos_bar_i
        result.retest_level    = retest_lvl
        result.retest_tolerance = retest_tol

    return result


# ── Private helpers ───────────────────────────────────────────────────────────
def _set_entry(
    result: ReversalResult, direction: str, bar: int,
    entry: float, sl: float, tp1: float, tp2: float,
    sl_pts: float, rr: float, tickets: list[Ticket],
) -> None:
    result.direction    = direction
    result.entry_bar    = bar
    result.entry_price  = round(entry, 2)
    result.sl_price     = round(sl, 2)
    result.tp1_price    = round(tp1, 2)
    result.tp2_price    = round(tp2, 2)
    result.sl_pts       = round(sl_pts, 2)
    result.rr_available = round(rr, 2)
    result.tickets      = tickets
    result.n_tickets    = len(tickets)
    result.total_lots   = round(sum(t.lots for t in tickets), 3)
    result.basket_risk_usd = round(sl_pts * result.total_lots * 100, 2)


def _process_basket(result: ReversalResult, bars: pd.DataFrame, i: int) -> None:
    """Check TP1/TP2/SL for each open ticket. SL takes priority."""
    bar = bars.iloc[i]
    h = float(bar["high"]); l = float(bar["low"])
    for tk in result.tickets:
        if tk.status != "OPEN":
            continue
        if result.direction == "SHORT":
            if h >= tk.sl:
                tk.status = "SL"; tk.exit_bar = i; tk.exit_price = tk.sl
                tk.pnl_usd = (tk.entry - tk.sl) * tk.lots * 100
            elif tk.tp_target == "TP1" and l <= tk.tp1:
                tk.status = "TP1"; tk.exit_bar = i; tk.exit_price = tk.tp1
                tk.pnl_usd = (tk.entry - tk.tp1) * tk.lots * 100
            elif tk.tp_target == "TP2" and l <= tk.tp2:
                tk.status = "TP2"; tk.exit_bar = i; tk.exit_price = tk.tp2
                tk.pnl_usd = (tk.entry - tk.tp2) * tk.lots * 100
        else:
            if l <= tk.sl:
                tk.status = "SL"; tk.exit_bar = i; tk.exit_price = tk.sl
                tk.pnl_usd = (tk.sl - tk.entry) * tk.lots * 100
            elif tk.tp_target == "TP1" and h >= tk.tp1:
                tk.status = "TP1"; tk.exit_bar = i; tk.exit_price = tk.tp1
                tk.pnl_usd = (tk.tp1 - tk.entry) * tk.lots * 100
            elif tk.tp_target == "TP2" and h >= tk.tp2:
                tk.status = "TP2"; tk.exit_bar = i; tk.exit_price = tk.tp2
                tk.pnl_usd = (tk.tp2 - tk.entry) * tk.lots * 100
