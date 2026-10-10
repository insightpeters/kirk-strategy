"""Fetch live M5 data via yfinance and align to Kirk's session structure.

Session convention (Kirk / engine.py):
  Bar 0  = 8:00 PM NY (Sunday or weekday open)
  Bar 48 = midnight NY  (Asian session closes)
  Bar 96 = 4:00 AM NY  (London open)

For a same-day NY session we want bars from the previous calendar day's
8 PM through the current 8 PM.  yfinance 5d / 5m gives us enough history.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import Config, SLMethod, TPMethod

NY = ZoneInfo("America/New_York")

# ── Instrument map ────────────────────────────────────────────────────────────
TICKERS: dict[str, str] = {
    "XAUUSD": "GC=F",
    "US30":   "YM=F",
    "XRPUSD": "XRP-USD",
}

# Per-instrument configs tuned for real price scales.
# XAUUSD ~$4200/oz, US30 ~$51000, XRPUSD ~$1.50
INSTRUMENT_CONFIG: dict[str, Config] = {
    "XAUUSD": Config(
        sl_method=SLMethod.STRUCTURE,
        sl_max_width_pts=50.0,
        sl_fixed_pts=15.0,
        sl_buffer_pts=0.5,
        tp_base_hit_pts=20.0,
        rr_ratio=1.5,
        # Real M5 gold data: bars rarely have 70% body/range; 1.5×ATR too strict
        displacement_body_atr_multiplier=0.6,
        displacement_body_range_ratio_min=0.45,
    ),
    "US30": Config(
        sl_method=SLMethod.STRUCTURE,
        sl_max_width_pts=300.0,
        sl_fixed_pts=80.0,
        sl_buffer_pts=5.0,
        tp_base_hit_pts=100.0,
        rr_ratio=1.5,
        displacement_body_atr_multiplier=0.6,
        displacement_body_range_ratio_min=0.45,
    ),
    "XRPUSD": Config(
        sl_method=SLMethod.STRUCTURE,
        sl_max_width_pts=0.15,
        sl_fixed_pts=0.05,
        sl_buffer_pts=0.003,
        tp_base_hit_pts=0.06,
        rr_ratio=1.5,
        displacement_body_atr_multiplier=0.6,
        displacement_body_range_ratio_min=0.45,
    ),
}


# ── Session helpers ───────────────────────────────────────────────────────────
def _session_start_utc(for_date: datetime | None = None) -> datetime:
    """Return the UTC datetime for 8 PM NY on the calendar day before for_date."""
    if for_date is None:
        for_date = datetime.now(NY)
    ny_date = for_date.astimezone(NY).date()
    # 8 PM NY on the PREVIOUS calendar day
    session_open_ny = datetime(ny_date.year, ny_date.month, ny_date.day,
                               20, 0, 0, tzinfo=NY) - timedelta(days=1)
    return session_open_ny.astimezone(timezone.utc)


def _bar_index_for_midnight(session_start_utc: datetime) -> int:
    """Number of 5-min bars from session_start until midnight NY."""
    midnight_ny = (session_start_utc.astimezone(NY)
                   + timedelta(hours=4))   # 8 PM + 4h = midnight
    delta = midnight_ny - session_start_utc.astimezone(NY)
    return int(delta.total_seconds() / 300)   # 48


# ── Main fetcher ──────────────────────────────────────────────────────────────
def fetch_session_bars(
    instrument: str,
    for_date: datetime | None = None,
) -> tuple[pd.DataFrame, int]:
    """Return (df, asian_end_bar) for one trading session.

    df columns: open, high, low, close, volume  (all float, lowercase)
    asian_end_bar: bar index where the Asian session closes (midnight NY)

    Returns an empty DataFrame on failure.
    """
    ticker_sym = TICKERS.get(instrument.upper())
    if ticker_sym is None:
        raise ValueError(f"Unknown instrument: {instrument!r}. Valid: {list(TICKERS)}")

    session_start = _session_start_utc(for_date)

    try:
        raw = yf.Ticker(ticker_sym).history(period="5d", interval="5m", auto_adjust=True)
    except Exception as exc:
        print(f"[data_loader] yfinance error for {ticker_sym}: {exc}")
        return pd.DataFrame(), 48

    if raw.empty:
        return pd.DataFrame(), 48

    # Normalise index to UTC
    raw.index = raw.index.tz_convert("UTC")
    raw.columns = [c.lower() for c in raw.columns]

    # Slice: from session_start up to session_start + 24h
    session_end = session_start + timedelta(hours=24)
    df = raw.loc[
        (raw.index >= pd.Timestamp(session_start)) &
        (raw.index < pd.Timestamp(session_end))
    ].copy()

    if df.empty:
        # Fallback: just take the most recent 288 bars (one 24h session)
        df = raw.tail(288).copy()

    timestamps = df.index                              # UTC DatetimeIndex
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = range(len(df))
    df.insert(0, "datetime_utc", timestamps.values)   # numpy datetime64[ns, UTC]

    asian_end_bar = _bar_index_for_midnight(session_start)
    return df, asian_end_bar


def fetch_all_sessions(
    instrument: str,
    n_days: int = 7,
) -> list[tuple[str, pd.DataFrame, int]]:
    """Return up to 5 completed trading sessions from the last n_days calendar days.

    Each entry: (label, df, asian_end_bar)
    label is the NY date the session OPENS, e.g. "Mon Sep 30".
    Sessions with fewer than 48 bars (incomplete Asian window) are skipped.
    """
    ticker_sym = TICKERS.get(instrument.upper())
    if ticker_sym is None:
        return []

    try:
        raw = yf.Ticker(ticker_sym).history(period="7d", interval="5m", auto_adjust=True)
    except Exception:
        return []

    if raw.empty:
        return []

    raw.index = raw.index.tz_convert("UTC")
    raw.columns = [c.lower() for c in raw.columns]

    now = datetime.now(NY)
    sessions: list[tuple[str, pd.DataFrame, int]] = []

    for days_back in range(n_days, 0, -1):
        for_date = now - timedelta(days=days_back - 1)
        session_start = _session_start_utc(for_date)
        session_end   = session_start + timedelta(hours=24)

        slice_ = raw.loc[
            (raw.index >= pd.Timestamp(session_start)) &
            (raw.index <  pd.Timestamp(session_end))
        ].copy()

        if len(slice_) < 50:       # skip sessions with almost no data (weekends)
            continue

        timestamps = slice_.index
        df_s = slice_[["open", "high", "low", "close", "volume"]].copy()
        df_s.index = range(len(df_s))
        df_s.insert(0, "datetime_utc", timestamps.values)

        asian_end_bar = _bar_index_for_midnight(session_start)
        # Label by the NY calendar date the session trades into (open night + 1 day)
        label = (session_start.astimezone(NY) + timedelta(days=1)).strftime("%a %b %d")
        sessions.append((label, df_s, asian_end_bar))

        if len(sessions) >= 5:
            break

    return sessions


def market_is_open(instrument: str = "XAUUSD", now: datetime | None = None) -> bool:
    """Return True if the given instrument is currently trading."""
    if instrument.upper() == "XRPUSD":
        return True   # crypto trades 24/7
    now_ny = (now or datetime.now(NY)).astimezone(NY)
    wd = now_ny.weekday()           # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    h  = now_ny.hour + now_ny.minute / 60
    if wd == 5:                     # Saturday: always closed
        return False
    if wd == 6:                     # Sunday: opens 6 PM NY
        return h >= 18.0
    if wd == 4:                     # Friday: closes 5 PM NY
        return h < 17.0
    return True                     # Mon–Thu: open (brief 5 PM daily break ignored)


def current_session_label(for_date: datetime | None = None) -> str:
    """Return a human-readable session name for the current NY time."""
    now_ny = (for_date or datetime.now(NY)).astimezone(NY)
    h = now_ny.hour + now_ny.minute / 60
    if 20 <= h or h < 0:
        return "Asian Open"
    if 0 <= h < 4:
        return "Asian Mid"
    if 4 <= h < 8:
        return "London"
    if 8 <= h < 17:
        return "New York"
    return "Pre-Asian"
