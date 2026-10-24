"""Data fetching and key-level computation for the HOD/LOD reversal module.

yfinance provides 5m bars for up to 60 days and 15m bars for up to 60 days.
We use 5m for the trigger FSM and 15m for context overlays.

Key levels computed per Kirk's specification:
    yesterday_high / yesterday_low   — prior session H/L
    asian_high     / asian_low       — 17:00–00:00 NY (22:00–05:00 UTC)
    london_high    / london_low      — 03:00–08:00 NY (08:00–13:00 UTC)
    session_high   / session_low     — rolling session H/L up to now
"""
from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

XAUUSD_SYMBOL = "GC=F"   # Gold continuous futures — closest to spot for retail use


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_gold_bars(interval: str = "5m", days: int = 3) -> pd.DataFrame:
    """Fetch XAUUSD bars from yfinance.

    Returns a DataFrame indexed by UTC datetime with columns:
        open, high, low, close, volume
    Raises ValueError if yfinance returns nothing.
    """
    import yfinance as yf

    end   = datetime.datetime.now(UTC)
    start = end - datetime.timedelta(days=days)
    df = yf.download(
        XAUUSD_SYMBOL,
        start=start.strftime("%Y-%m-%d"),
        end=(end + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"yfinance returned empty data for {XAUUSD_SYMBOL} ({interval})")

    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _session_window(
    df: pd.DataFrame,
    start_ny: datetime.time,
    end_ny: datetime.time,
    date_ny: datetime.date,
) -> pd.DataFrame:
    """Slice bars that fall in [start_ny, end_ny) for the given NY date."""
    tz_idx = df.index.tz_convert(NY)
    mask = (
        (tz_idx.date == date_ny) &
        (tz_idx.time >= start_ny) &
        (tz_idx.time < end_ny)
    )
    return df[mask]


def _session_hl(df: pd.DataFrame) -> tuple[float | None, float | None]:
    if df.empty:
        return None, None
    return float(df["high"].max()), float(df["low"].min())


# ── Key-level computation ─────────────────────────────────────────────────────

def compute_key_levels(
    df: pd.DataFrame,
    now_utc: datetime.datetime | None = None,
) -> dict[str, float]:
    """Compute key levels relative to *now_utc* (defaults to current time).

    Asian session  : 17:00–00:00 NY previous calendar day
    London session : 03:00–08:00 NY today
    Session high/low: rolling today from 17:00 NY (previous day) to now
    Yesterday H/L  : full day (00:00–24:00 NY) two calendar days ago

    Returns a dict with only those keys that have data (never puts None in).
    """
    if now_utc is None:
        now_utc = datetime.datetime.now(UTC)

    now_ny   = now_utc.astimezone(NY)
    today_ny = now_ny.date()

    # "Yesterday" in NY terms — full prior calendar day
    prev_day = today_ny - datetime.timedelta(days=1)
    day_before = today_ny - datetime.timedelta(days=2)

    levels: dict[str, float] = {}

    # Yesterday H/L — full prior calendar day bars
    yest_bars = _session_window(df,
        datetime.time(0, 0), datetime.time(23, 59, 59), prev_day)
    yh, yl = _session_hl(yest_bars)
    if yh is not None:
        levels["yesterday_high"] = yh
        levels["yesterday_low"]  = yl  # type: ignore[assignment]

    # Asian session — 17:00 NY (prev day) → 00:00 NY (today)
    asian_bars = _session_window(df,
        datetime.time(17, 0), datetime.time(23, 59, 59), prev_day)
    ah, al = _session_hl(asian_bars)
    if ah is not None:
        levels["asian_high"] = ah
        levels["asian_low"]  = al  # type: ignore[assignment]

    # London session — 03:00–08:00 NY today
    lon_bars = _session_window(df,
        datetime.time(3, 0), datetime.time(8, 0), today_ny)
    lh, ll = _session_hl(lon_bars)
    if lh is not None:
        levels["london_high"] = lh
        levels["london_low"]  = ll  # type: ignore[assignment]

    # Rolling session high/low — 17:00 NY yesterday to now
    sess_bars = _session_window(df,
        datetime.time(17, 0), now_ny.time(), prev_day)
    # also include bars from today 00:00 → now
    sess_bars_today = _session_window(df,
        datetime.time(0, 0), now_ny.time(), today_ny)
    all_sess = pd.concat([sess_bars, sess_bars_today]).sort_index()
    if not all_sess.empty:
        all_sess = all_sess[all_sess.index <= now_utc]
    sh, sl = _session_hl(all_sess)
    if sh is not None:
        levels["session_high"] = sh
        levels["session_low"]  = sl  # type: ignore[assignment]

    return levels


# ── Historical session slice for backtesting ──────────────────────────────────

def slice_session_for_date(
    df: pd.DataFrame,
    date_ny: datetime.date,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return (session_bars, key_levels) for a full trading day.

    session_bars: 17:00 NY prev → 17:00 NY date (next day open)
    key_levels:   computed as if it is end-of-session for date_ny
    """
    end_utc = datetime.datetime(
        date_ny.year, date_ny.month, date_ny.day,
        17, 0, tzinfo=NY,
    ).astimezone(UTC)

    kl = compute_key_levels(df, now_utc=end_utc)

    prev_day = date_ny - datetime.timedelta(days=1)
    sess_bars = _session_window(df,
        datetime.time(17, 0), datetime.time(23, 59, 59), prev_day)
    today_bars = _session_window(df,
        datetime.time(0, 0), datetime.time(17, 0), date_ny)
    session = pd.concat([sess_bars, today_bars]).sort_index()
    session = session.reset_index()
    if "Datetime" in session.columns:
        session = session.rename(columns={"Datetime": "datetime"})
    elif "index" in session.columns:
        session = session.rename(columns={"index": "datetime"})
    session = session.set_index("datetime")

    return session, kl
