"""Synthetic M5 XAUUSD price data generator.

Produces realistic bar sequences with embedded, controllable setups so the FSM
has something meaningful to detect.  Every important bar (sweep, displacement,
BOS, pullback, entry) is produced deterministically from the seed so the UI can
annotate them precisely.

Generator catalogue
───────────────────
generate_bearish_setup   — Asian High sweep, displacement down, BOS, pullback, entry.
                           outcome="tp": price runs to target.
                           outcome="sl": price reverses to hit the stop.
generate_bullish_setup   — Mirror: Asian Low sweep, displacement up, BOS, pullback, entry.
generate_wide_stop_setup — Extreme sweep spike (30 pts above Asian High).
                           Structural SL is ~40 pts from entry → max-stop-width veto.
"""

import zoneinfo
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

NY_TZ  = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = zoneinfo.ZoneInfo("UTC")


@dataclass
class SetupMeta:
    """Ground-truth metadata about the embedded setup, used for annotations."""
    direction: str           # "BEARISH" | "BULLISH"
    base_price: float

    asian_high: float
    asian_low: float
    asian_high_bar: int
    asian_low_bar: int

    sweep_bar: int
    sweep_wick_extreme: float   # wick tip that poked through the level

    disp_start: int
    disp_end: int
    fvg_top: float
    fvg_bot: float

    bos_bar: int
    bos_level: float

    pullback_start: int
    entry_bar: int

    fib_50:  float
    fib_618: float
    fib_786: float

    sl_price: float
    tp_price: float
    entry_price: float


def _make_bar(o, h, l, c, volume):
    h = max(h, o, c)
    l = min(l, o, c)
    return dict(open=round(o,2), high=round(h,2), low=round(l,2),
                close=round(c,2), volume=int(volume))


# ── Bearish (Asian High sweep) ─────────────────────────────────────────────────

def generate_bearish_setup(
    seed: int = 42,
    base_price: float = 2650.0,
    outcome: str = "tp",        # "tp" → price runs to target; "sl" → price hits stop
    tp_target_pts: float = 18.0,
) -> tuple[pd.DataFrame, SetupMeta]:
    """
    Asian-High sweep → strong displacement → BOS → pullback into Fibonacci zone → entry.

    Bar index reference
    ─────────────────────────────────────────────────
      0 – 47   Asian session  (8 PM – midnight NY)
     48 – 113  London / pre-market  (midnight – 8 AM NY)
    114 – 143  NY pre-entry drift  (8 AM – 10:30 AM NY)
    144         SWEEP
    145 – 147   DISPLACEMENT  (3 bars)
    148         BOS
    149 – 153   PULLBACK  (5 bars)
    154         ENTRY TRIGGER
    155 – 165   TRADE RUNS TO TARGET (or STOP)
    ─────────────────────────────────────────────────
    """
    rng = np.random.default_rng(seed)

    asian_high = base_price + 12.0
    asian_low  = base_price -  8.0
    mid_range  = base_price +  2.0

    SWEEP_BAR      = 144
    DISP_START     = 145
    DISP_END       = 147
    BOS_BAR        = 148
    PULLBACK_START = 149
    ENTRY_BAR      = 154
    TOTAL_BARS     = 166

    sweep_wick_high = asian_high + 2.5
    bos_swing_low   = base_price - 5.0

    disp_high = asian_high - 1.0
    disp_low  = base_price - 8.0

    fib_50  = disp_low + (disp_high - disp_low) * 0.500
    fib_618 = disp_low + (disp_high - disp_low) * 0.618
    fib_786 = disp_low + (disp_high - disp_low) * 0.786

    tp_price    = base_price - tp_target_pts
    sl_price    = sweep_wick_high + 0.5
    entry_price = fib_618 - 1.0

    # ── midpoint price path ───────────────────────────────────────────────────
    mid = np.zeros(TOTAL_BARS)

    for i in range(48):
        t = i / 47.0
        if t < 0.45:
            mid[i] = base_price + (asian_high - base_price) * (t / 0.45)
        else:
            mid[i] = asian_high - (asian_high - mid_range) * ((t - 0.45) / 0.55)
        mid[i] += rng.normal(0, 0.25)

    for i in range(48, SWEEP_BAR):
        t = (i - 48) / (SWEEP_BAR - 48)
        target = mid_range + (asian_high - 1.5 - mid_range) * t
        mid[i] = mid[i-1] * 0.88 + target * 0.12 + rng.normal(0, 0.30)

    mid[SWEEP_BAR] = disp_high

    step = (mid[SWEEP_BAR] - disp_low) / 3.0
    for i in range(DISP_START, DISP_END + 1):
        mid[i] = mid[i-1] - step + rng.normal(0, 0.15)

    mid[BOS_BAR] = bos_swing_low - 2.5 + rng.normal(0, 0.20)

    for i in range(PULLBACK_START, ENTRY_BAR):
        t = (i - PULLBACK_START) / (ENTRY_BAR - PULLBACK_START)
        mid[i] = mid[BOS_BAR] + (fib_618 - mid[BOS_BAR]) * t + rng.normal(0, 0.20)

    mid[ENTRY_BAR] = entry_price

    # Post-entry path depends on scenario outcome
    if outcome == "tp":
        for i in range(ENTRY_BAR + 1, TOTAL_BARS):
            t = (i - ENTRY_BAR) / (TOTAL_BARS - ENTRY_BAR - 1)
            mid[i] = mid[ENTRY_BAR] - (mid[ENTRY_BAR] - tp_price) * t + rng.normal(0, 0.40)
    else:   # "sl" — price reverses upward and hits the stop
        for i in range(ENTRY_BAR + 1, TOTAL_BARS):
            t = (i - ENTRY_BAR) / (TOTAL_BARS - ENTRY_BAR - 1)
            mid[i] = mid[ENTRY_BAR] + (sl_price - mid[ENTRY_BAR]) * t + rng.normal(0, 0.40)

    # ── build OHLCV bars ──────────────────────────────────────────────────────
    bars = []
    asian_high_bar_idx = int(np.argmax(mid[:48]))
    asian_low_bar_idx  = int(np.argmin(mid[:48]))

    pre_sweep_close = mid[SWEEP_BAR - 1]

    for i in range(TOTAL_BARS):
        m   = mid[i]
        vol = rng.integers(300, 1800)
        r   = max(abs(rng.normal(1.8, 0.5)), 0.5)

        if i == SWEEP_BAR:
            o = pre_sweep_close
            h = sweep_wick_high
            c = disp_high
            l = c - 0.6
        elif i == DISP_START:
            o = mid[SWEEP_BAR]
            c = o - step * 0.85
            h = o + 0.4
            l = c - 0.5
            vol = rng.integers(1200, 2000)
        elif i == DISP_START + 1:
            o = mid[i-1] - 0.3
            c = o - step * 0.90
            h = o + 0.3
            l = c - 0.4
            vol = rng.integers(1000, 1800)
        elif i == DISP_START + 2:
            o = mid[i-1] - 0.2
            c = o - step * 0.80
            h = o + 0.4
            l = c - 0.5
            vol = rng.integers(900, 1600)
        elif i == BOS_BAR:
            o = mid[i-1]
            c = bos_swing_low - 2.8
            h = o + 0.5
            l = c - 0.4
            vol = rng.integers(800, 1400)
        elif i == ENTRY_BAR:
            # Bearish rejection from Fibonacci zone — open above fib_618, close below
            o = fib_618 + 1.5
            c = entry_price   # = fib_618 - 1.0
            h = o + 0.6
            l = c - 0.5
        else:
            o = m + rng.uniform(-r * 0.35, r * 0.35)
            c = m + rng.uniform(-r * 0.35, r * 0.35)
            h = max(o, c) + abs(rng.normal(0.25, 0.15))
            l = min(o, c) - abs(rng.normal(0.25, 0.15))

        bars.append(_make_bar(o, h, l, c, vol))

    fvg_top = bars[SWEEP_BAR]['low']
    fvg_bot = bars[DISP_START + 2]['high']
    if fvg_top < fvg_bot:
        fvg_top, fvg_bot = fvg_bot, fvg_top

    start_ny = datetime(2024, 1, 15, 20, 0, tzinfo=NY_TZ)
    for i, bar in enumerate(bars):
        ts_ny  = start_ny + timedelta(minutes=5 * i)
        ts_utc = ts_ny.astimezone(UTC_TZ)
        bar['timestamp_utc'] = pd.Timestamp(ts_utc)
        bar['timestamp_ny']  = pd.Timestamp(ts_ny)
        bar['bar_idx']       = i

    df = pd.DataFrame(bars)

    meta = SetupMeta(
        direction="BEARISH", base_price=base_price,
        asian_high=asian_high, asian_low=asian_low,
        asian_high_bar=asian_high_bar_idx, asian_low_bar=asian_low_bar_idx,
        sweep_bar=SWEEP_BAR, sweep_wick_extreme=sweep_wick_high,
        disp_start=DISP_START, disp_end=DISP_END,
        fvg_top=round(fvg_top, 2), fvg_bot=round(fvg_bot, 2),
        bos_bar=BOS_BAR, bos_level=bos_swing_low,
        pullback_start=PULLBACK_START, entry_bar=ENTRY_BAR,
        fib_50=round(fib_50, 2), fib_618=round(fib_618, 2), fib_786=round(fib_786, 2),
        sl_price=round(sl_price, 2), tp_price=round(tp_price, 2),
        entry_price=round(entry_price, 2),
    )

    return df, meta


# ── Bullish (Asian Low sweep) ──────────────────────────────────────────────────

def generate_bullish_setup(seed: int = 99, base_price: float = 2650.0) -> tuple[pd.DataFrame, SetupMeta]:
    """Sweep of Asian LOW → bullish displacement → BOS up → pullback → buy."""
    rng = np.random.default_rng(seed)

    asian_high = base_price +  8.0
    asian_low  = base_price - 12.0
    mid_range  = base_price -  2.0

    SWEEP_BAR      = 144
    DISP_START     = 145
    DISP_END       = 147
    BOS_BAR        = 148
    PULLBACK_START = 149
    ENTRY_BAR      = 154
    TOTAL_BARS     = 166

    sweep_wick_low  = asian_low - 2.5
    bos_swing_high  = base_price + 5.0

    disp_low  = asian_low + 1.0
    disp_high = base_price + 8.0

    fib_50  = disp_high - (disp_high - disp_low) * 0.500
    fib_618 = disp_high - (disp_high - disp_low) * 0.618
    fib_786 = disp_high - (disp_high - disp_low) * 0.786

    tp_price    = base_price + 18.0
    sl_price    = sweep_wick_low - 0.5
    entry_price = fib_618 + 1.0

    mid = np.zeros(TOTAL_BARS)

    for i in range(48):
        t = i / 47.0
        if t < 0.45:
            mid[i] = base_price - (base_price - asian_low) * (t / 0.45)
        else:
            mid[i] = asian_low + (mid_range - asian_low) * ((t - 0.45) / 0.55)
        mid[i] += rng.normal(0, 0.25)

    for i in range(48, SWEEP_BAR):
        t = (i - 48) / (SWEEP_BAR - 48)
        target = mid_range - (mid_range - (asian_low + 1.5)) * t
        mid[i] = mid[i-1] * 0.88 + target * 0.12 + rng.normal(0, 0.30)

    mid[SWEEP_BAR] = disp_low

    step = (disp_high - mid[SWEEP_BAR]) / 3.0
    for i in range(DISP_START, DISP_END + 1):
        mid[i] = mid[i-1] + step + rng.normal(0, 0.15)

    mid[BOS_BAR] = bos_swing_high + 2.5 + rng.normal(0, 0.20)

    for i in range(PULLBACK_START, ENTRY_BAR):
        t = (i - PULLBACK_START) / (ENTRY_BAR - PULLBACK_START)
        mid[i] = mid[BOS_BAR] - (mid[BOS_BAR] - fib_618) * t + rng.normal(0, 0.20)

    mid[ENTRY_BAR] = entry_price

    for i in range(ENTRY_BAR + 1, TOTAL_BARS):
        t = (i - ENTRY_BAR) / (TOTAL_BARS - ENTRY_BAR - 1)
        mid[i] = mid[ENTRY_BAR] + (tp_price - mid[ENTRY_BAR]) * t + rng.normal(0, 0.40)

    bars = []
    asian_high_bar_idx = int(np.argmax(mid[:48]))
    asian_low_bar_idx  = int(np.argmin(mid[:48]))

    for i in range(TOTAL_BARS):
        m   = mid[i]
        vol = rng.integers(300, 1800)
        r   = max(abs(rng.normal(1.8, 0.5)), 0.5)

        if i == SWEEP_BAR:
            o = mid[i-1]
            l = sweep_wick_low
            c = disp_low
            h = c + 0.6
        elif DISP_START <= i <= DISP_END:
            o = mid[i-1] + 0.2
            c = o + step * 0.85
            l = o - 0.4
            h = c + 0.5
            vol = rng.integers(900, 2000)
        elif i == BOS_BAR:
            o = mid[i-1]
            c = bos_swing_high + 2.8
            l = o - 0.5
            h = c + 0.4
        elif i == ENTRY_BAR:
            # Bullish rejection — open below fib_618, close above
            o = fib_618 - 1.5
            c = entry_price   # = fib_618 + 1.0
            l = o - 0.6
            h = c + 0.5
        else:
            o = m + rng.uniform(-r * 0.35, r * 0.35)
            c = m + rng.uniform(-r * 0.35, r * 0.35)
            h = max(o, c) + abs(rng.normal(0.25, 0.15))
            l = min(o, c) - abs(rng.normal(0.25, 0.15))

        bars.append(_make_bar(o, h, l, c, vol))

    fvg_bot = bars[SWEEP_BAR]['high']
    fvg_top = bars[DISP_END]['low']
    if fvg_top < fvg_bot:
        fvg_top, fvg_bot = fvg_bot, fvg_top

    start_ny = datetime(2024, 1, 15, 20, 0, tzinfo=NY_TZ)
    for i, bar in enumerate(bars):
        ts_ny  = start_ny + timedelta(minutes=5 * i)
        ts_utc = ts_ny.astimezone(UTC_TZ)
        bar['timestamp_utc'] = pd.Timestamp(ts_utc)
        bar['timestamp_ny']  = pd.Timestamp(ts_ny)
        bar['bar_idx']       = i

    df   = pd.DataFrame(bars)
    meta = SetupMeta(
        direction="BULLISH", base_price=base_price,
        asian_high=asian_high, asian_low=asian_low,
        asian_high_bar=asian_high_bar_idx, asian_low_bar=asian_low_bar_idx,
        sweep_bar=SWEEP_BAR, sweep_wick_extreme=sweep_wick_low,
        disp_start=DISP_START, disp_end=DISP_END,
        fvg_top=round(fvg_top,2), fvg_bot=round(fvg_bot,2),
        bos_bar=BOS_BAR, bos_level=bos_swing_high,
        pullback_start=PULLBACK_START, entry_bar=ENTRY_BAR,
        fib_50=round(fib_50,2), fib_618=round(fib_618,2), fib_786=round(fib_786,2),
        sl_price=round(sl_price,2), tp_price=round(tp_price,2),
        entry_price=round(entry_price,2),
    )
    return df, meta


# ── Wide-stop veto scenario ────────────────────────────────────────────────────

def generate_wide_stop_setup(seed: int = 33, base_price: float = 2650.0) -> tuple[pd.DataFrame, SetupMeta]:
    """
    Same structure as bearish setup but the sweep wick spikes 32 pts above the
    Asian High.  The structural SL (stop beyond sweep extreme + buffer) would be
    ~40 pts from entry — far exceeding the default sl_max_width of 20 pts.
    The engine vetoes the trade rather than forcing a tighter stop.
    """
    rng = np.random.default_rng(seed)

    asian_high = base_price + 12.0
    asian_low  = base_price -  8.0
    mid_range  = base_price +  2.0

    SWEEP_BAR      = 144
    DISP_START     = 145
    DISP_END       = 147
    BOS_BAR        = 148
    PULLBACK_START = 149
    ENTRY_BAR      = 154
    TOTAL_BARS     = 166

    # ── extreme sweep spike ───────────────────────────────────────────────────
    sweep_wick_high = asian_high + 32.0   # 2694 — massive spike above Asian High
    bos_swing_low   = base_price - 5.0

    disp_high = asian_high - 1.0
    disp_low  = base_price - 8.0

    fib_50  = disp_low + (disp_high - disp_low) * 0.500
    fib_618 = disp_low + (disp_high - disp_low) * 0.618
    fib_786 = disp_low + (disp_high - disp_low) * 0.786

    tp_price    = base_price - 18.0
    sl_price    = sweep_wick_high + 0.5   # 2694.5 — very far from entry
    entry_price = fib_618 - 1.0

    mid = np.zeros(TOTAL_BARS)

    for i in range(48):
        t = i / 47.0
        if t < 0.45:
            mid[i] = base_price + (asian_high - base_price) * (t / 0.45)
        else:
            mid[i] = asian_high - (asian_high - mid_range) * ((t - 0.45) / 0.55)
        mid[i] += rng.normal(0, 0.25)

    for i in range(48, SWEEP_BAR):
        t = (i - 48) / (SWEEP_BAR - 48)
        target = mid_range + (asian_high - 1.5 - mid_range) * t
        mid[i] = mid[i-1] * 0.88 + target * 0.12 + rng.normal(0, 0.30)

    mid[SWEEP_BAR] = disp_high  # bar closes back near Asian High despite the spike

    step = (mid[SWEEP_BAR] - disp_low) / 3.0
    for i in range(DISP_START, DISP_END + 1):
        mid[i] = mid[i-1] - step + rng.normal(0, 0.15)

    mid[BOS_BAR] = bos_swing_low - 2.5 + rng.normal(0, 0.20)

    for i in range(PULLBACK_START, ENTRY_BAR):
        t = (i - PULLBACK_START) / (ENTRY_BAR - PULLBACK_START)
        mid[i] = mid[BOS_BAR] + (fib_618 - mid[BOS_BAR]) * t + rng.normal(0, 0.20)

    mid[ENTRY_BAR] = entry_price

    for i in range(ENTRY_BAR + 1, TOTAL_BARS):
        t = (i - ENTRY_BAR) / (TOTAL_BARS - ENTRY_BAR - 1)
        mid[i] = mid[ENTRY_BAR] - (mid[ENTRY_BAR] - tp_price) * t + rng.normal(0, 0.40)

    bars = []
    asian_high_bar_idx = int(np.argmax(mid[:48]))
    asian_low_bar_idx  = int(np.argmin(mid[:48]))

    pre_sweep_close = mid[SWEEP_BAR - 1]

    for i in range(TOTAL_BARS):
        m   = mid[i]
        vol = rng.integers(300, 1800)
        r   = max(abs(rng.normal(1.8, 0.5)), 0.5)

        if i == SWEEP_BAR:
            o = pre_sweep_close
            h = sweep_wick_high   # the massive spike
            c = disp_high
            l = c - 0.6
        elif i == DISP_START:
            o = mid[SWEEP_BAR]
            c = o - step * 0.85
            h = o + 0.4
            l = c - 0.5
            vol = rng.integers(1200, 2000)
        elif i == DISP_START + 1:
            o = mid[i-1] - 0.3
            c = o - step * 0.90
            h = o + 0.3
            l = c - 0.4
            vol = rng.integers(1000, 1800)
        elif i == DISP_START + 2:
            o = mid[i-1] - 0.2
            c = o - step * 0.80
            h = o + 0.4
            l = c - 0.5
            vol = rng.integers(900, 1600)
        elif i == BOS_BAR:
            o = mid[i-1]
            c = bos_swing_low - 2.8
            h = o + 0.5
            l = c - 0.4
            vol = rng.integers(800, 1400)
        elif i == ENTRY_BAR:
            o = fib_618 + 1.5
            c = entry_price
            h = o + 0.6
            l = c - 0.5
        else:
            o = m + rng.uniform(-r * 0.35, r * 0.35)
            c = m + rng.uniform(-r * 0.35, r * 0.35)
            h = max(o, c) + abs(rng.normal(0.25, 0.15))
            l = min(o, c) - abs(rng.normal(0.25, 0.15))

        bars.append(_make_bar(o, h, l, c, vol))

    fvg_top = bars[SWEEP_BAR]['low']
    fvg_bot = bars[DISP_START + 2]['high']
    if fvg_top < fvg_bot:
        fvg_top, fvg_bot = fvg_bot, fvg_top

    start_ny = datetime(2024, 1, 15, 20, 0, tzinfo=NY_TZ)
    for i, bar in enumerate(bars):
        ts_ny  = start_ny + timedelta(minutes=5 * i)
        ts_utc = ts_ny.astimezone(UTC_TZ)
        bar['timestamp_utc'] = pd.Timestamp(ts_utc)
        bar['timestamp_ny']  = pd.Timestamp(ts_ny)
        bar['bar_idx']       = i

    df = pd.DataFrame(bars)

    meta = SetupMeta(
        direction="BEARISH", base_price=base_price,
        asian_high=asian_high, asian_low=asian_low,
        asian_high_bar=asian_high_bar_idx, asian_low_bar=asian_low_bar_idx,
        sweep_bar=SWEEP_BAR, sweep_wick_extreme=sweep_wick_high,
        disp_start=DISP_START, disp_end=DISP_END,
        fvg_top=round(fvg_top, 2), fvg_bot=round(fvg_bot, 2),
        bos_bar=BOS_BAR, bos_level=bos_swing_low,
        pullback_start=PULLBACK_START, entry_bar=ENTRY_BAR,
        fib_50=round(fib_50, 2), fib_618=round(fib_618, 2), fib_786=round(fib_786, 2),
        sl_price=round(sl_price, 2), tp_price=round(tp_price, 2),
        entry_price=round(entry_price, 2),
    )
    return df, meta
