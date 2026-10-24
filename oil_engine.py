"""Oil Sweep-Break-Retest FSM — sells only (v1).

Dual timeframe: 15m for sweep/BOS/displacement, 5m for retracement/entry/exit.
Closed candles only. No repainting.

Session (America/New_York, DST-aware):
  Asian range    : 20:00 prev-day → 02:00
  London ext     : 02:00 → 05:00
  Entry window   : 05:00 → 12:00
  EOD close      : 15:30

Contract spec (USOIL / USOUSD retail spot):
  1 lot = $1,000 per point  →  risk_usd = sl_pts × lots × 1000
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from oil_config import OilConfig

__version__ = "1.0.0"

NY  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OIL_STATES = {
    0: "WATCHING",
    1: "SWEPT",
    2: "RECLAIMED",
    3: "AWAITING RETRACE",
    4: "IN BASKET",
    5: "CLOSED",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class OilEvent:
    ts: str        # "HH:MM" NY
    tf: str        # "15m" | "5m" | "sys"
    status: str    # info | pass | fail | wait | skip | entry | exit_tp | exit_sl
    headline: str
    details: list[str] = field(default_factory=list)


@dataclass
class OilTicket:
    idx: int
    lots: float
    entry: float
    sl: float
    tp: float
    tp_r: float
    runner: bool = False    # trailing mode after 2R
    status: str = "OPEN"   # OPEN | TP | SL | EOD | RUNNER_SL
    exit_ts: str = ""
    exit_price: float = 0.0
    pnl_gross: float = 0.0  # before costs


@dataclass
class OilResult:
    current_state: int = 0
    skip_reason: Optional[str] = None
    narrative: list[OilEvent] = field(default_factory=list)

    # Levels
    eligible_highs: dict[str, float] = field(default_factory=dict)
    range_quality: str = "NO_DATA"    # PASS | EXTENDED | SKIP_SESSION | NO_DATA
    atr_at_2am: float = 0.0

    # Sweep
    swept_key: Optional[str] = None
    swept_level: Optional[float] = None
    sweep_extreme: Optional[float] = None
    sweep_ts: Optional[str] = None

    # BOS / displacement
    swing_ref_price: Optional[float] = None
    bos_ts: Optional[str] = None
    disp_high: Optional[float] = None
    disp_low: Optional[float] = None

    # Retracement
    retrace_lo: Optional[float] = None
    retrace_hi: Optional[float] = None
    retrace_touch_ts: Optional[str] = None

    # Entry
    entry_ts: Optional[str] = None
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    sl_pts: float = 0.0
    atr_at_entry: float = 0.0

    # Basket
    tickets: list[OilTicket] = field(default_factory=list)
    n_tickets: int = 0
    total_lots: float = 0.0
    basket_risk_usd: float = 0.0

    # P&L
    gross_pnl_usd: float = 0.0
    costs_usd: float = 0.0
    net_pnl_usd: float = 0.0
    exit_reason: str = ""     # TP_ALL | SL_ALL | MIXED | EOD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _session_bounds(date_ny: datetime.date, cfg: OilConfig) -> dict[str, pd.Timestamp]:
    prev = date_ny - datetime.timedelta(days=1)

    def _ny(d: datetime.date, h: int, m: int = 0) -> pd.Timestamp:
        return pd.Timestamp(datetime.datetime(d.year, d.month, d.day, h, m, tzinfo=NY))

    return {
        "asian_start": _ny(prev, cfg.asian_start_hour),
        "asian_end":   _ny(date_ny, cfg.asian_end_hour),
        "london_end":  _ny(date_ny, cfg.london_end_hour),
        "entry_start": _ny(date_ny, cfg.entry_start_hour),
        "entry_end":   _ny(date_ny, cfg.entry_end_hour),
        "eod":         _ny(date_ny, cfg.eod_hour, cfg.eod_minute),
    }


def _last_swing_low(bars: pd.DataFrame, before_row: int, lookback: int = 80) -> tuple[int, float] | tuple[None, None]:
    """Last confirmed swing low (2 bars each side) fully confirmed before before_row."""
    for j in range(before_row - 3, 1, -1):
        if before_row - j > lookback:
            break
        if j + 2 >= before_row:
            continue
        v = float(bars["low"].iloc[j])
        if (v < float(bars["low"].iloc[j - 1]) and v < float(bars["low"].iloc[j - 2]) and
                v < float(bars["low"].iloc[j + 1]) and v < float(bars["low"].iloc[j + 2])):
            return j, v
    return None, None


def _last_swing_high(bars: pd.DataFrame, after_row: int, lookback: int = 60) -> tuple[float] | tuple[None]:
    """Last confirmed swing high (2 bars each side) fully confirmed AFTER after_row.
    Used for runner trailing."""
    end = len(bars) - 3
    for j in range(end, after_row + 2, -1):
        if j - after_row > lookback:
            break
        if j + 2 >= len(bars):
            continue
        v = float(bars["high"].iloc[j])
        if (v > float(bars["high"].iloc[j - 1]) and v > float(bars["high"].iloc[j - 2]) and
                v > float(bars["high"].iloc[j + 1]) and v > float(bars["high"].iloc[j + 2])):
            return v
    return None


def _build_basket(
    entry: float, sl: float, sl_pts: float,
    target_rs: list[float],
    cfg: OilConfig,
) -> list[OilTicket]:
    risk_usd = cfg.account_equity * cfg.risk_pct
    total_lots_raw = risk_usd / (sl_pts * cfg.contract_size)
    step = cfg.lot_step

    per_ticket_raw = total_lots_raw / cfg.max_tickets
    per_ticket = int(per_ticket_raw / step) * step     # round DOWN

    if per_ticket < cfg.min_lot:
        # Can't fit max_tickets — try fewer
        per_ticket = cfg.min_lot
        n_tickets = min(cfg.max_tickets, int(total_lots_raw / cfg.min_lot))
        if n_tickets == 0:
            return []
    else:
        n_tickets = cfg.max_tickets

    tickets = []
    # Assign targets: 2 per R level, fill from first
    tgt_list: list[float] = []
    remaining = n_tickets
    for r in target_rs:
        take = min(2, remaining)
        tgt_list.extend([r] * take)
        remaining -= take
        if remaining == 0:
            break

    for idx, r in enumerate(tgt_list):
        tp = entry - r * sl_pts          # SHORT: price goes down
        is_runner = cfg.use_runner and idx >= len(tgt_list) - 2
        tickets.append(OilTicket(
            idx=idx, lots=per_ticket,
            entry=entry, sl=sl, tp=tp, tp_r=r,
            runner=is_runner,
        ))
    return tickets


def _is_eia_blocked(ts_ny: pd.Timestamp, cfg: OilConfig) -> bool:
    if not cfg.use_eia_block or ts_ny.weekday() != 2:
        return False
    eia = ts_ny.replace(hour=10, minute=30, second=0, microsecond=0)
    delta = (ts_ny - eia).total_seconds() / 60.0
    return -cfg.news_block_before_min <= delta <= cfg.news_block_after_min


def _ts_ny(ts) -> str:
    return pd.Timestamp(ts).tz_convert(NY).strftime("%H:%M")


# ── Main engine ───────────────────────────────────────────────────────────────

def run_oil_engine(
    df_15m: pd.DataFrame,        # UTC-indexed: open/high/low/close
    df_5m: pd.DataFrame,         # UTC-indexed: open/high/low/close
    date_ny: datetime.date,
    cfg: OilConfig,
    news_times: list[pd.Timestamp] | None = None,
) -> OilResult:
    """Run one trading session for date_ny. Returns OilResult.

    df_15m / df_5m must be timezone-aware UTC DatetimeIndex DataFrames.
    news_times: list of UTC pd.Timestamps for additional news events.
    """
    result = OilResult()
    if news_times is None:
        news_times = []

    bounds = _session_bounds(date_ny, cfg)

    def _E(ts, tf: str, status: str, headline: str, *details: str) -> None:
        result.narrative.append(OilEvent(_ts_ny(ts), tf, status, headline, list(details)))

    def _blocked(ts_utc: pd.Timestamp) -> bool:
        ts_ny = ts_utc.tz_convert(NY)
        if _is_eia_blocked(ts_ny, cfg):
            return True
        for nt in news_times:
            delta_min = (ts_ny - nt.tz_convert(NY)).total_seconds() / 60.0
            if -cfg.news_block_before_min <= delta_min <= cfg.news_block_after_min:
                return True
        return False

    # ── Eligible highs ─────────────────────────────────────────────────────────
    prev = date_ny - datetime.timedelta(days=1)
    yest_bars = df_15m[pd.Series(df_15m.index).apply(
        lambda t: pd.Timestamp(t).tz_convert(NY).date() == prev
    ).values]
    yest_high = float(yest_bars["high"].max()) if not yest_bars.empty else None

    asian_bars = df_15m[
        (df_15m.index >= bounds["asian_start"]) & (df_15m.index < bounds["asian_end"])
    ]
    london_bars = df_15m[
        (df_15m.index >= bounds["asian_end"]) & (df_15m.index < bounds["london_end"])
    ]

    # ATR(14) at 2AM on all 15m bars up to that point
    bars_to_2am = df_15m[df_15m.index <= bounds["asian_end"]]
    atr_2am_s = _atr(bars_to_2am) if len(bars_to_2am) >= 2 else pd.Series([1.0])
    atr_2am = max(float(atr_2am_s.iloc[-1]), 0.01)
    result.atr_at_2am = round(atr_2am, 4)

    threshold = cfg.range_min_atr_mult * atr_2am

    if not asian_bars.empty:
        a_h = float(asian_bars["high"].max())
        a_l = float(asian_bars["low"].min())
        a_w = a_h - a_l
    else:
        a_h = a_l = a_w = 0.0

    if a_w >= threshold:
        range_q, range_h = "PASS", a_h
    elif not london_bars.empty:
        ext_h = max(a_h, float(london_bars["high"].max()))
        ext_l = min(a_l, float(london_bars["low"].min())) if a_l > 0 else float(london_bars["low"].min())
        ext_w = ext_h - ext_l
        if ext_w >= threshold:
            range_q, range_h = "EXTENDED", ext_h
        else:
            range_q, range_h = "SKIP_SESSION", 0.0
    else:
        range_q, range_h = "SKIP_SESSION", 0.0

    result.range_quality = range_q
    eligible: dict[str, float] = {}
    if yest_high is not None:
        eligible["Yesterday H"] = yest_high
    if range_q in ("PASS", "EXTENDED") and range_h > 0:
        label = "Session H" if range_q == "PASS" else "Extended H"
        eligible[label] = range_h
    result.eligible_highs = eligible

    _E(bounds["entry_start"], "sys", "info",
       f"{date_ny}  ·  ATR={atr_2am:.3f}  ·  range={range_q}",
       *([f"Eligible: {', '.join(f'{k}={v:.3f}' for k,v in eligible.items())}"] if eligible else ["No eligible levels"]))

    if not eligible:
        result.skip_reason = "no eligible levels"
        return result

    # ── All 15m bars in context window (for ATR + swing detection) ─────────────
    ctx_15m = df_15m[
        (df_15m.index >= bounds["asian_start"]) & (df_15m.index <= bounds["eod"])
    ].reset_index(drop=False)
    dt_col = ctx_15m.columns[0]   # datetime column after reset
    ctx_15m = ctx_15m.rename(columns={dt_col: "dt"})

    atr_ctx = _atr(ctx_15m)

    # Map UTC timestamps to integer rows in ctx_15m
    ts_to_row = {row["dt"]: i for i, row in ctx_15m.iterrows()}

    # ── Entry window bars ──────────────────────────────────────────────────────
    ew_15m = ctx_15m[
        (ctx_15m["dt"] >= bounds["entry_start"]) & (ctx_15m["dt"] < bounds["entry_end"])
    ]
    if ew_15m.empty:
        result.skip_reason = "no 15m bars in entry window"
        return result

    # ── Phase 1: 15m FSM (sweep → reclaim → BOS+displacement) ─────────────────
    state = 0
    sweep_row       = None    # integer row in ctx_15m
    swept_key       = None
    swept_lv        = None
    sweep_extreme   = None
    reclaim_row     = None
    swing_ref_px    = None
    disp_row        = None
    disp_ts_utc     = None
    disp_high       = None
    disp_low        = None
    max_h_since_sw  = None

    for _, bar15 in ew_15m.iterrows():
        i = int(bar15.name)
        h = float(bar15["high"]); l = float(bar15["low"])
        c = float(bar15["close"]); o = float(bar15["open"])
        ts = bar15["dt"]
        atr_v = max(float(atr_ctx.iloc[i]), 0.01)

        if state >= 1:
            if max_h_since_sw is None or h > max_h_since_sw:
                max_h_since_sw = h

        # ── WATCHING ──────────────────────────────────────────────────────────
        if state == 0:
            if _blocked(ts):
                _E(ts, "15m", "wait", f"News block — skipping {_ts_ny(ts)}")
                continue
            for lk, lv in eligible.items():
                min_sw = max(cfg.sweep_atr_mult * atr_v, 2.0 * cfg.spread_pts)
                if h > lv + min_sw:
                    sweep_row = i; swept_key = lk; swept_lv = lv
                    sweep_extreme = h; state = 1; max_h_since_sw = h
                    result.swept_key = lk; result.swept_level = lv
                    result.sweep_extreme = sweep_extreme
                    result.sweep_ts = _ts_ny(ts)
                    _E(ts, "15m", "pass",
                       f"Sweep — {lk} ({lv:.3f}) violated · high {h:.3f}",
                       f"Threshold: {lv + min_sw:.3f} (ATR×{cfg.sweep_atr_mult}={cfg.sweep_atr_mult * atr_v:.3f})",
                       f"Need 15m close back < {lv:.3f} within {cfg.sweep_close_bars} bar(s)")
                    break

        # ── SWEPT ─────────────────────────────────────────────────────────────
        elif state == 1:
            if h > sweep_extreme:
                sweep_extreme = h; result.sweep_extreme = h
            bars_since = i - sweep_row
            if c < swept_lv:
                sl_i, sl_px = _last_swing_low(ctx_15m, sweep_row)
                if sl_i is None:
                    _E(ts, "15m", "fail", "Reclaimed but no confirmed swing low before sweep — reset")
                    result.skip_reason = "no pre-sweep swing low"
                    state = 0
                else:
                    reclaim_row = i; swing_ref_px = sl_px; state = 2
                    result.swing_ref_price = sl_px
                    _E(ts, "15m", "pass",
                       f"Reclaim — closed {c:.3f} below {swept_lv:.3f}",
                       f"Swing-low ref: {sl_px:.3f}  ·  need BOS+displacement within {cfg.bos_bars} bars")
            elif bars_since >= cfg.sweep_close_bars:
                _E(ts, "15m", "fail", f"No reclaim in {cfg.sweep_close_bars} bars — reset")
                result.skip_reason = "sweep expired (no reclaim)"
                state = 0

        # ── RECLAIMED — wait for BOS + displacement ───────────────────────────
        elif state == 2:
            bars_since = i - reclaim_row
            if h > sweep_extreme + 0.5:
                _E(ts, "15m", "fail", f"New higher high {h:.3f} — setup cancelled")
                result.skip_reason = "new high after reclaim"
                state = 0; continue

            body = abs(c - o)
            rng  = h - l
            body_ok  = body >= cfg.disp_body_atr_mult * atr_v
            ratio_ok = rng > 0 and (body / rng) >= cfg.disp_body_range_pct
            close_ok = rng > 0 and c <= l + cfg.disp_close_top_pct * rng
            bos_ok   = swing_ref_px is not None and c < swing_ref_px

            if bos_ok and body_ok and ratio_ok and close_ok:
                disp_row = i; disp_ts_utc = ts; disp_high = h; disp_low = l
                result.bos_ts = _ts_ny(ts)
                result.disp_high = h; result.disp_low = l
                retrace_rng = disp_high - disp_low
                result.retrace_lo = round(disp_low + cfg.retrace_pct_low  * retrace_rng, 4)
                result.retrace_hi = round(disp_low + cfg.retrace_pct_high * retrace_rng, 4)
                state = 3
                _E(ts, "15m", "pass",
                   f"BOS + Displacement — closed {c:.3f} < swing-low {swing_ref_px:.3f}",
                   f"Body {body:.3f} ≥ {cfg.disp_body_atr_mult}×ATR({atr_v:.3f}) ✓  "
                   f"body/rng {body/rng:.0%} ≥ {cfg.disp_body_range_pct:.0%} ✓  "
                   f"close pct {(c-l)/rng:.0%} ≤ {cfg.disp_close_top_pct:.0%} ✓",
                   f"Retrace zone: {result.retrace_lo:.3f}–{result.retrace_hi:.3f} · switching to 5m")
                break   # leave 15m loop, enter 5m phase

            elif bos_ok:
                _E(ts, "15m", "fail",
                   f"BOS at {c:.3f} but displacement failed",
                   f"body={body:.3f}(need≥{cfg.disp_body_atr_mult * atr_v:.3f}) "
                   f"ratio={body/(rng or 1):.0%}(need≥{cfg.disp_body_range_pct:.0%}) "
                   f"close%={(c-l)/(rng or 1):.0%}(need≤{cfg.disp_close_top_pct:.0%})")
                result.skip_reason = "BOS without displacement"
                state = 0

            elif bars_since >= cfg.bos_bars:
                _E(ts, "15m", "fail", f"BOS window expired after {cfg.bos_bars} bars")
                result.skip_reason = "BOS window expired"
                state = 0

    if state != 3:
        result.current_state = state
        return result

    # ── Phase 2: 5m retracement + confirmation ────────────────────────────────
    disp_close_ts = pd.Timestamp(disp_ts_utc) + pd.Timedelta("15min")
    window_end_ts = disp_close_ts + pd.Timedelta(f"{cfg.retrace_bars_5m * 5}min")

    bars_5m = df_5m[
        (df_5m.index >= disp_close_ts) & (df_5m.index <= window_end_ts)
    ]

    if bars_5m.empty:
        result.skip_reason = "no 5m data for retracement window"
        result.current_state = 3
        return result

    retrace_lo = result.retrace_lo
    retrace_hi = result.retrace_hi
    zone_touched  = False
    prev_5m_low   = None
    confirm_ts    = None
    confirm_close = None

    for ts5, bar5 in bars_5m.iterrows():
        h5 = float(bar5["high"]); l5 = float(bar5["low"])
        c5 = float(bar5["close"]); o5 = float(bar5["open"])

        # Cancel if sweep high exceeded
        if h5 > sweep_extreme + 0.5:
            _E(ts5, "5m", "fail",
               f"Price {h5:.3f} exceeded sweep extreme {sweep_extreme:.3f} — cancelled")
            result.skip_reason = "new high during retracement (exceeded sweep extreme)"
            result.current_state = 3
            return result

        if not zone_touched:
            if h5 >= retrace_lo:
                zone_touched = True
                result.retrace_touch_ts = _ts_ny(ts5)
                _E(ts5, "5m", "pass",
                   f"Retrace zone touched — H={h5:.3f}  zone={retrace_lo:.3f}–{retrace_hi:.3f}")
        else:
            if prev_5m_low is not None and c5 < o5 and c5 < prev_5m_low:
                confirm_ts = ts5; confirm_close = c5
                _E(ts5, "5m", "pass",
                   f"Bearish confirmation — close {c5:.3f} < open {o5:.3f} and < prior low {prev_5m_low:.3f}")
                break
            else:
                _E(ts5, "5m", "wait",
                   f"Awaiting confirmation — need bearish close < {(prev_5m_low or l5):.3f}")
        prev_5m_low = l5

    if confirm_ts is None:
        _E(disp_close_ts, "5m", "fail",
           f"No confirmation within {cfg.retrace_bars_5m} 5m bars — setup expired")
        result.skip_reason = "retracement/confirmation timed out"
        result.current_state = 3
        return result

    # ── Phase 3: Entry ────────────────────────────────────────────────────────
    after_confirm = df_5m[
        (df_5m.index > confirm_ts) & (df_5m.index <= bounds["eod"])
    ]
    if after_confirm.empty:
        result.skip_reason = "no bars after confirmation before EOD"
        result.current_state = 3
        return result

    entry_ts_raw   = after_confirm.index[0]
    entry_price    = float(after_confirm.iloc[0]["open"])

    if entry_price < disp_low:
        _E(entry_ts_raw, "5m", "skip",
           f"Entry skipped — open {entry_price:.3f} < displacement low {disp_low:.3f} (chasing)")
        result.skip_reason = f"entry below displacement low ({entry_price:.3f} < {disp_low:.3f})"
        result.current_state = 3
        return result

    # SL: max high from sweep bar through confirmation bar + buffer
    sweep_bar_ts = ctx_15m.iloc[sweep_row]["dt"]
    highs_15m = df_15m[
        (df_15m.index >= sweep_bar_ts) & (df_15m.index <= confirm_ts)
    ]["high"]
    highs_5m = df_5m[
        (df_5m.index >= disp_close_ts) & (df_5m.index <= confirm_ts)
    ]["high"]
    max_high = max(
        float(highs_15m.max()) if not highs_15m.empty else sweep_extreme,
        float(highs_5m.max()) if not highs_5m.empty else 0.0,
        sweep_extreme,
    )

    atr_entry_bars = df_15m[df_15m.index <= confirm_ts]
    atr_entry_s = _atr(atr_entry_bars) if len(atr_entry_bars) >= 2 else pd.Series([atr_2am])
    atr_entry = max(float(atr_entry_s.iloc[-1]), 0.01)

    sl_buffer = max(cfg.sl_buffer_atr_mult * atr_entry, 2.0 * cfg.spread_pts)
    sl_price  = max_high + sl_buffer
    sl_pts    = sl_price - entry_price

    if sl_pts <= 0:
        result.skip_reason = "SL at or below entry"
        result.current_state = 3
        return result

    if sl_pts > cfg.max_sl_atr_mult * atr_entry:
        _E(entry_ts_raw, "5m", "skip",
           f"SL too wide — {sl_pts:.3f} pts > {cfg.max_sl_atr_mult}×ATR {atr_entry:.3f}",
           f"Max allowed: {cfg.max_sl_atr_mult * atr_entry:.3f}")
        result.skip_reason = f"SL {sl_pts:.3f} > {cfg.max_sl_atr_mult}×ATR"
        result.current_state = 3
        return result

    if cfg.spread_pts > cfg.max_spread_sl_pct * sl_pts:
        result.skip_reason = f"spread ({cfg.spread_pts}) > {cfg.max_spread_sl_pct:.0%} of SL ({sl_pts:.3f})"
        result.current_state = 3
        return result

    tickets = _build_basket(
        entry_price, sl_price, sl_pts,
        [cfg.tp1_r, cfg.tp2_r, cfg.tp3_r, cfg.tp4_r], cfg,
    )
    if not tickets:
        result.skip_reason = "position too small for minimum lot"
        result.current_state = 3
        return result

    result.entry_ts    = _ts_ny(entry_ts_raw)
    result.entry_price = round(entry_price, 4)
    result.sl_price    = round(sl_price, 4)
    result.sl_pts      = round(sl_pts, 4)
    result.atr_at_entry = round(atr_entry, 4)
    result.tickets     = tickets
    result.n_tickets   = len(tickets)
    result.total_lots  = round(sum(t.lots for t in tickets), 3)
    result.basket_risk_usd = round(sl_pts * result.total_lots * cfg.contract_size, 2)

    _E(entry_ts_raw, "5m", "entry",
       f"SHORT @ {entry_price:.3f}  ·  {len(tickets)} tickets  ·  {result.total_lots} lots",
       f"SL {sl_price:.3f} ({sl_pts:.3f} pts = {sl_pts/atr_entry:.1f}×ATR)  ·  risk ${result.basket_risk_usd:.2f}",
       "  |  ".join(f"T{t.idx+1}:{t.tp:.3f}({t.tp_r}R)" for t in tickets))

    # ── Phase 4: Exit management on 5m bars ───────────────────────────────────
    exit_bars = df_5m[
        (df_5m.index > entry_ts_raw) & (df_5m.index <= bounds["eod"])
    ]
    two_r_hit  = False  # for runner activation
    runner_sl: dict[int, float] = {}   # ticket idx → current trailing SL

    for exit_ts, exit_bar in exit_bars.iterrows():
        h5 = float(exit_bar["high"]); l5 = float(exit_bar["low"])
        c5 = float(exit_bar["close"])

        open_tickets = [t for t in tickets if t.status == "OPEN"]
        if not open_tickets:
            break

        # Runner: check for new confirmed 15m swing high to trail SL
        if cfg.use_runner and two_r_hit:
            swing_h = _last_swing_high(ctx_15m, disp_row)
            if swing_h is not None:
                trail_sl = swing_h + sl_buffer
                for t in tickets:
                    if t.runner and t.status == "OPEN":
                        current = runner_sl.get(t.idx, t.sl)
                        if trail_sl < current:  # tighten only
                            runner_sl[t.idx] = trail_sl

        for t in open_tickets:
            eff_sl = runner_sl.get(t.idx, t.sl) if t.runner else t.sl

            if h5 >= eff_sl:
                t.status = "SL" if not t.runner else "RUNNER_SL"
                t.exit_ts = _ts_ny(exit_ts)
                t.exit_price = eff_sl
                t.pnl_gross = (t.entry - eff_sl) * t.lots * cfg.contract_size
                _E(exit_ts, "5m", "exit_sl",
                   f"Ticket {t.idx+1} SL @ {eff_sl:.3f}  P&L ${t.pnl_gross:+.2f}")
            elif l5 <= t.tp:
                t.status = "TP"
                t.exit_ts = _ts_ny(exit_ts)
                t.exit_price = t.tp
                t.pnl_gross = (t.entry - t.tp) * t.lots * cfg.contract_size
                _E(exit_ts, "5m", "exit_tp",
                   f"Ticket {t.idx+1} TP{int(t.tp_r*10)//10}R @ {t.tp:.3f}  P&L ${t.pnl_gross:+.2f}")
                if t.tp_r >= cfg.tp2_r:
                    two_r_hit = True

    # EOD close remaining
    eod_price_bars = df_5m[df_5m.index <= bounds["eod"]]
    eod_price = float(eod_price_bars["close"].iloc[-1]) if not eod_price_bars.empty else entry_price
    for t in tickets:
        if t.status == "OPEN":
            t.status = "EOD"
            t.exit_ts = f"{cfg.eod_hour:02d}:{cfg.eod_minute:02d}"
            t.exit_price = eod_price
            t.pnl_gross = (t.entry - eod_price) * t.lots * cfg.contract_size

    # ── Costs & final P&L ─────────────────────────────────────────────────────
    gross  = sum(t.pnl_gross for t in tickets)
    cost_per_lot_rt = (cfg.spread_pts + 2 * cfg.slippage_pts) * cfg.contract_size + 2 * cfg.commission_per_lot
    costs  = sum(t.lots * cost_per_lot_rt for t in tickets)

    result.gross_pnl_usd = round(gross, 2)
    result.costs_usd     = round(costs, 2)
    result.net_pnl_usd   = round(gross - costs, 2)
    result.current_state = 4

    statuses = {t.status for t in tickets}
    if "SL" in statuses or "RUNNER_SL" in statuses:
        result.exit_reason = "MIXED" if "TP" in statuses or "EOD" in statuses else "SL_ALL"
    elif "EOD" in statuses:
        result.exit_reason = "EOD"
    else:
        result.exit_reason = "TP_ALL"

    _E(bounds["eod"], "sys", "exit_tp" if gross > 0 else "exit_sl",
       f"Session closed  ·  Gross ${gross:+.2f}  ·  Costs ${costs:.2f}  ·  Net ${gross - costs:+.2f}",
       f"Exit reasons: {', '.join(sorted(statuses))}")

    return result
