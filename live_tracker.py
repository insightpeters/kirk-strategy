"""Kirk Strategy Live Signals — $10,000 paper portfolio tracker.

Run:
    streamlit run prototype/live_tracker.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    INSTRUMENT_CONFIG, fetch_session_bars, fetch_all_sessions,
    current_session_label, market_is_open,
)
from engine import EngineResult, compute_ema, run_engine

NY = ZoneInfo("America/New_York")
INSTRUMENTS = ["XAUUSD", "US30", "XRPUSD"]

ACCOUNT_SIZE = 10_000.0
RISK_PCT     = 0.01          # 1% risk per trade = $100

UNIT_LABELS = {"XAUUSD": "oz", "US30": "units", "XRPUSD": "XRP"}

st.set_page_config(
    page_title="Kirk Signals",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Signal card ── */
.sig-card {
    border-radius:12px; padding:20px 24px; margin-bottom:12px;
}
.sig-status {
    font-size:1.7em; font-weight:800; letter-spacing:.04em; margin-bottom:4px;
}
.sig-sub {
    font-size:.95em; color:rgba(255,255,255,.65); margin-bottom:14px;
}
.sig-levels {
    display:flex; gap:20px; flex-wrap:wrap; margin-top:12px;
}
.sig-level {
    text-align:center;
}
.sig-level-label { font-size:.72em; opacity:.6; text-transform:uppercase; }
.sig-level-value { font-size:1.25em; font-weight:700; }

/* signal card colours */
.card-wait     { background:#1a1f2e; border:1px solid #374151; }
.card-watch    { background:#1a1a2e; border:1px solid #4338CA; }
.card-forming  { background:#292005; border:2px solid #D97706; }
.card-short    { background:#2d0a0a; border:2px solid #DC2626; }
.card-long     { background:#022c16; border:2px solid #059669; }
.card-tp       { background:#022c16; border:2px solid #10B981; }
.card-sl       { background:#2d0a0a; border:2px solid #EF4444; }
.card-veto     { background:#1c1001; border:1px solid #92400E; }

/* ── Portfolio panel ── */
.port-card {
    background:#111827; border-radius:10px; padding:16px 20px; height:100%;
}
.port-title { font-size:.8em; color:#6B7280; text-transform:uppercase; margin-bottom:10px; }
.port-row { display:flex; justify-content:space-between; padding:4px 0;
            border-bottom:1px solid #1f2937; font-size:.92em; }
.port-row:last-child { border-bottom:none; }
.port-key { color:#9CA3AF; }
.port-val { font-weight:600; }

/* ── Sequence tracker ── */
.seq-wrap { display:flex; align-items:flex-start; gap:0; margin:10px 0; }
.seq-step { flex:1; text-align:center; position:relative; }
.seq-step:not(:last-child)::after {
    content:''; position:absolute; top:14px; left:50%; width:100%;
    height:2px; background:#1f2937; z-index:0;
}
.seq-dot {
    width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    margin:0 auto 4px; font-size:.85em; font-weight:700;
    position:relative; z-index:1;
}
.seq-dot-done    { background:#1d4ed8; color:#fff; }
.seq-dot-active  { background:#D97706; color:#fff; box-shadow:0 0 8px #D97706; }
.seq-dot-pending { background:#1f2937; color:#4B5563; }
.seq-label { font-size:.65em; color:#6B7280; line-height:1.3; }
.seq-ts    { font-size:.6em; color:#374151; font-family:monospace; margin-top:2px; }

/* ── Colour helpers ── */
.c-green { color:#34D399; }  .c-red   { color:#F87171; }
.c-gold  { color:#FCD34D; }  .c-blue  { color:#60A5FA; }
.c-muted { color:#6B7280; }  .c-amber { color:#FCD34D; }
.bold    { font-weight:700; }

/* ── P&L number ── */
.pnl-big {
    font-size:2.8em; font-weight:800; letter-spacing:-.01em; line-height:1.1;
}
</style>
""", unsafe_allow_html=True)


# ── Data / cache ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_data(instrument: str, _ts: int) -> tuple[pd.DataFrame, int]:
    if not market_is_open():
        return pd.DataFrame(), 48   # market closed — don't show stale fallback data
    return fetch_session_bars(instrument)


def get_tick() -> int:
    return int(time.time() // 60)


# ── Portfolio math ────────────────────────────────────────────────────────────
def portfolio(result: EngineResult) -> dict:
    dollar_risk = ACCOUNT_SIZE * RISK_PCT   # $100
    if result.entry_price is None or result.sl_price is None:
        return {"dollar_risk": dollar_risk, "units": 0,
                "tp_gain": 0, "sl_loss": -dollar_risk,
                "rr": result.rr or 0, "realised": None, "unrealised": None}

    sl_width = abs(result.entry_price - result.sl_price)
    units    = dollar_risk / sl_width if sl_width > 0 else 0
    tp_pts   = abs(result.tp_price - result.entry_price) if result.tp_price else 0
    tp_gain  = units * tp_pts

    realised = None
    if result.exit_reason == "TP_HIT":
        realised = tp_gain
    elif result.exit_reason == "SL_HIT":
        realised = -dollar_risk

    return {
        "dollar_risk": dollar_risk,
        "units": round(units, 2),
        "sl_width": round(sl_width, 2),
        "tp_pts": round(tp_pts, 2),
        "tp_gain": round(tp_gain, 2),
        "sl_loss": round(-dollar_risk, 2),
        "rr": round(result.rr or 0, 2),
        "realised": round(realised, 2) if realised is not None else None,
    }


def unrealised(result: EngineResult, df: pd.DataFrame, port: dict) -> float | None:
    """Live unrealised P&L if trade is still open."""
    if result.current_state not in (7, 8) or df.empty or port["units"] == 0:
        return None
    cur = float(df["close"].iloc[-1])
    if result.trade_direction == "SHORT":
        pts = result.entry_price - cur
    else:
        pts = cur - result.entry_price
    return round(pts * port["units"], 2)


# ── Timestamp helper ──────────────────────────────────────────────────────────
def _bar_ts(df: pd.DataFrame, bar_idx: int | None) -> str:
    if bar_idx is None or df.empty or "datetime_utc" not in df.columns:
        return "—"
    if bar_idx >= len(df):
        return "—"
    raw = df["datetime_utc"].iloc[bar_idx]
    ts  = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(NY).strftime("%H:%M")


def _bar_ts_labels(df: pd.DataFrame) -> list[str]:
    if df.empty or "datetime_utc" not in df.columns:
        return [str(i) for i in range(len(df))]
    ts = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(NY)
    return ts.dt.strftime("%H:%M").tolist()


# ── Signal card ───────────────────────────────────────────────────────────────
def _signal_card(result: EngineResult, port: dict, ur: float | None,
                 instrument: str) -> None:
    s = result.current_state
    ul = UNIT_LABELS.get(instrument, "units")

    # Determine card type + headline
    if not market_is_open():
        card_cls = "card-wait"
        status   = "🌙  MARKET CLOSED"
        sub      = "Gold futures closed · reopens Sunday 6 PM NY · check the 5-day history below for last signal"
    elif result.veto_reason:
        card_cls, status, sub = "card-veto", "⊘  SETUP VETOED", result.veto_reason
    elif result.exit_reason == "TP_HIT":
        card_cls = "card-tp"
        status   = f"✅  TARGET HIT  +${port['realised']:,.2f}"
        sub      = f"Trade closed · {result.pnl_pts:+.2f} pts · {port['units']} {ul}"
    elif result.exit_reason == "SL_HIT":
        card_cls = "card-sl"
        status   = f"✗  STOP HIT  −${abs(port['realised']):,.2f}"
        sub      = f"Trade closed · {result.pnl_pts:+.2f} pts · structural stop"
    elif s in (7, 8) and result.trade_direction == "SHORT":
        live_str = f"  ·  Unrealised: {'${:+,.2f}'.format(ur)}" if ur is not None else ""
        card_cls = "card-short"
        status   = f"◀  SELL SIGNAL  ACTIVE{live_str}"
        sub      = f"SHORT {instrument} @ {result.entry_price:.2f}  ·  watching TP/SL"
    elif s in (7, 8) and result.trade_direction == "LONG":
        live_str = f"  ·  Unrealised: {'${:+,.2f}'.format(ur)}" if ur is not None else ""
        card_cls = "card-long"
        status   = f"▶  BUY SIGNAL  ACTIVE{live_str}"
        sub      = f"LONG {instrument} @ {result.entry_price:.2f}  ·  watching TP/SL"
    elif s in (5, 6):
        card_cls = "card-forming"
        status   = "⚠  SETUP FORMING — PREPARE ORDER"
        fib_lo   = result.fib_zone_low_price or 0
        fib_hi   = result.fib_zone_high_price or 0
        direction = "SELL" if result.liq_direction == "HIGH" else "BUY"
        sub      = f"{direction} if price retraces to {fib_lo:.2f}–{fib_hi:.2f} and shows a reaction bar"
    elif s == 2:
        card_cls, status = "card-watch", "⚡  SWEEP DETECTED — WATCHING DISPLACEMENT"
        sub = f"{'Bearish' if result.liq_direction=='HIGH' else 'Bullish'} sweep @ {result.sweep_extreme:.2f} · no entry yet"
    elif s in (3, 4):
        card_cls, status = "card-watch", "〰  SEQUENCE BUILDING — BOS CONFIRMED"
        sub = "Displacement done · waiting for price to retrace into entry zone"
    elif s == 1:
        card_cls, status = "card-watch", "◉  ASIAN RANGE SET — WATCHING FOR SWEEP"
        sub = f"High: {result.asian_high:.2f}  ·  Low: {result.asian_low:.2f}"
    else:
        card_cls, status = "card-wait", "— WAITING · ASIAN SESSION IN PROGRESS"
        sub = "No tradeable levels until the Asian session closes"

    # Level pills
    levels_html = ""
    if result.entry_price and result.sl_price and result.tp_price:
        sl_pts = abs(result.entry_price - result.sl_price)
        tp_pts = abs(result.tp_price - result.entry_price)
        ep_col = "#EF4444" if result.trade_direction == "SHORT" else "#10B981"
        levels_html = f"""
<div class="sig-levels">
  <div class="sig-level">
    <div class="sig-level-label">Entry</div>
    <div class="sig-level-value" style="color:{ep_col}">{result.entry_price:.2f}</div>
  </div>
  <div class="sig-level">
    <div class="sig-level-label">Stop Loss</div>
    <div class="sig-level-value" style="color:#F87171">{result.sl_price:.2f} <span style="font-size:.65em;opacity:.6">({sl_pts:.1f} pts)</span></div>
  </div>
  <div class="sig-level">
    <div class="sig-level-label">Take Profit</div>
    <div class="sig-level-value" style="color:#34D399">{result.tp_price:.2f} <span style="font-size:.65em;opacity:.6">({tp_pts:.1f} pts)</span></div>
  </div>
  <div class="sig-level">
    <div class="sig-level-label">R:R</div>
    <div class="sig-level-value" style="color:#FCD34D">{result.rr:.1f}R</div>
  </div>
</div>"""
    elif s in (5, 6) and result.fib_zone_low_price:
        fib_lo = result.fib_zone_low_price
        fib_hi = result.fib_zone_high_price
        levels_html = f"""
<div class="sig-levels">
  <div class="sig-level">
    <div class="sig-level-label">Entry Zone (Fib 50–78.6%)</div>
    <div class="sig-level-value" style="color:#FCD34D">{fib_lo:.2f} – {fib_hi:.2f}</div>
  </div>
  <div class="sig-level">
    <div class="sig-level-label">Sweep Level</div>
    <div class="sig-level-value" style="color:#A78BFA">{result.sweep_extreme:.2f}</div>
  </div>
  <div class="sig-level">
    <div class="sig-level-label">BOS Level</div>
    <div class="sig-level-value" style="color:#34D399">{result.bos_level:.2f}</div>
  </div>
</div>"""

    st.markdown(
        f'<div class="sig-card {card_cls}">'
        f'<div class="sig-status">{status}</div>'
        f'<div class="sig-sub">{sub}</div>'
        f'{levels_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Portfolio panel ───────────────────────────────────────────────────────────
def _portfolio_panel(result: EngineResult, port: dict, ur: float | None,
                     instrument: str) -> None:
    ul = UNIT_LABELS.get(instrument, "units")

    # Big P&L number
    if port["realised"] is not None:
        pnl = port["realised"]
        pnl_color = "#34D399" if pnl >= 0 else "#F87171"
        pnl_str   = f"{'${:+,.2f}'.format(pnl)}"
        pnl_label = "CLOSED P&L"
    elif ur is not None:
        pnl_color = "#34D399" if ur >= 0 else "#F87171"
        pnl_str   = f"{'${:+,.2f}'.format(ur)}"
        pnl_label = "LIVE P&L"
    else:
        pnl_color = "#4B5563"
        pnl_str   = "$0"
        pnl_label = "P&L"

    st.markdown(f"""
<div class="port-card">
  <div class="port-title">$10,000 Paper Portfolio</div>

  <div style="text-align:center;margin:10px 0 16px">
    <div style="font-size:.72em;color:#6B7280;text-transform:uppercase">{pnl_label}</div>
    <div class="pnl-big" style="color:{pnl_color}">{pnl_str}</div>
  </div>

  <div class="port-row"><span class="port-key">Account size</span>
    <span class="port-val">$10,000</span></div>
  <div class="port-row"><span class="port-key">Risk per trade (1%)</span>
    <span class="port-val">$100</span></div>
  <div class="port-row"><span class="port-key">Position size</span>
    <span class="port-val">{port['units']} {ul}</span></div>
  <div class="port-row"><span class="port-key">Stop width</span>
    <span class="port-val">{port.get('sl_width', '—')} pts</span></div>
  <div class="port-row"><span class="port-key">If TP hit</span>
    <span class="port-val" style="color:#34D399">+${port['tp_gain']:,.2f}</span></div>
  <div class="port-row"><span class="port-key">If SL hit</span>
    <span class="port-val" style="color:#F87171">−$100.00</span></div>
  <div class="port-row"><span class="port-key">R:R ratio</span>
    <span class="port-val">{port['rr']}R</span></div>
  <div class="port-row"><span class="port-key">Confluence grade</span>
    <span class="port-val" style="color:#FCD34D">
      {result.confluence.grade if result.confluence else '—'}
    </span></div>
</div>
""", unsafe_allow_html=True)


# ── Sequence tracker ──────────────────────────────────────────────────────────
def _sequence_tracker(result: EngineResult, df: pd.DataFrame, asian_end_bar: int) -> None:
    steps = [
        ("Asian", "◉", result.current_state >= 1, asian_end_bar - 1),
        ("Sweep",  "⚡", result.current_state >= 2, result.sweep_bar),
        ("Displ",  "▼",  result.current_state >= 3, result.disp_start),
        ("BOS",   "✂",  result.current_state >= 4, result.bos_bar),
        ("Zone",  "⬡",  result.current_state >= 5, result.bos_bar),
        ("Entry", "◆",  result.current_state >= 7, result.entry_bar),
        ("Exit",  "⏹",  result.current_state == 9, result.exit_bar),
    ]

    html = '<div class="seq-wrap">'
    for label, icon, done, bar_idx in steps:
        ts_str = _bar_ts(df, bar_idx) if done and bar_idx is not None else "—"
        is_active = (
            (label == "Zone"  and result.current_state in (5, 6)) or
            (label == "Entry" and result.current_state in (7, 8))
        )
        if is_active:
            dot_cls = "seq-dot-active"
        elif done:
            dot_cls = "seq-dot-done"
        else:
            dot_cls = "seq-dot-pending"
        html += (
            f'<div class="seq-step">'
            f'<div class="seq-dot {dot_cls}">{icon}</div>'
            f'<div class="seq-label">{label}</div>'
            f'<div class="seq-ts">{ts_str if done else ""}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Chart ─────────────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, result: EngineResult, instrument: str) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Waiting for data…", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=16, color="#4B5563"))
        fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=360)
        return fig

    n = len(df)
    x = list(range(n))
    ts_labels = _bar_ts_labels(df)
    tick_idxs = list(range(0, n, 12))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=instrument,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        line_width=1, whiskerwidth=0.4,
    ))

    ema = compute_ema(df["close"], 20)
    fig.add_trace(go.Scatter(x=x, y=ema, mode="lines", name="EMA 20",
                             line=dict(color="#F59E0B", width=1.2, dash="dot")))

    if result.asian_high:
        for level, color, label in [
            (result.asian_high, "#60A5FA", f"Asian H {result.asian_high:.2f}"),
            (result.asian_low,  "#A78BFA", f"Asian L {result.asian_low:.2f}"),
        ]:
            fig.add_shape(type="line", x0=0, x1=n-1, y0=level, y1=level,
                          line=dict(color=color, width=1, dash="dash"))
            fig.add_annotation(x=n-1, y=level, text=f" {label}",
                                xanchor="left", showarrow=False,
                                font=dict(size=9, color=color))

    if result.fib_zone_low_price and result.disp_end is not None:
        fig.add_shape(type="rect",
                      x0=result.disp_end, x1=n-1,
                      y0=result.fib_zone_low_price, y1=result.fib_zone_high_price,
                      fillcolor="rgba(59,130,246,0.10)",
                      line=dict(color="#3B82F6", width=1))
        fig.add_annotation(x=(result.disp_end + n) // 2,
                           y=result.fib_zone_high_price,
                           text="Entry Zone", yanchor="bottom", showarrow=False,
                           font=dict(size=9, color="#3B82F6"))

    if result.fvg_top and result.fvg_bot and result.disp_end is not None:
        fig.add_shape(type="rect",
                      x0=result.disp_end, x1=n-1,
                      y0=result.fvg_bot, y1=result.fvg_top,
                      fillcolor="rgba(234,179,8,0.08)",
                      line=dict(color="#EAB308", width=1, dash="dot"))

    if result.bos_level and result.bos_bar is not None:
        fig.add_shape(type="line",
                      x0=result.bos_bar, x1=n-1,
                      y0=result.bos_level, y1=result.bos_level,
                      line=dict(color="#34D399", width=1, dash="longdash"))
        fig.add_annotation(x=result.bos_bar, y=result.bos_level, text="BOS",
                           xanchor="right", showarrow=False,
                           font=dict(size=9, color="#34D399"))

    if result.sweep_bar is not None:
        ay = -22 if result.liq_direction == "HIGH" else 22
        fig.add_annotation(x=result.sweep_bar, y=result.sweep_extreme,
                           text="SWEEP", showarrow=True, arrowhead=2,
                           arrowcolor="#A78BFA", ax=0, ay=ay,
                           font=dict(size=9, color="#A78BFA"))

    if result.entry_price and result.entry_bar is not None:
        for px, color, label in [
            (result.entry_price, "#FBBF24", "ENTRY"),
            (result.sl_price,    "#F87171", "SL"),
            (result.tp_price,    "#34D399", "TP"),
        ]:
            if px is None: continue
            fig.add_shape(type="line", x0=result.entry_bar, x1=n-1,
                          y0=px, y1=px,
                          line=dict(color=color, width=1.2,
                                    dash="solid" if label == "ENTRY" else "dash"))
            fig.add_annotation(x=result.entry_bar, y=px, text=label,
                               xanchor="right", showarrow=False,
                               font=dict(size=9, color=color))

    if result.exit_bar is not None:
        ec  = "#34D399" if result.exit_reason == "TP_HIT" else "#F87171"
        el  = "TP ✅" if result.exit_reason == "TP_HIT" else "SL ✗"
        fig.add_annotation(x=result.exit_bar, y=result.exit_price,
                           text=el, showarrow=True, arrowhead=2, arrowcolor=ec,
                           ax=0, ay=-22, font=dict(size=10, color=ec, family="Arial Black"))

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        font=dict(color="#E5E7EB", size=11),
        height=360, margin=dict(l=50, r=140, t=20, b=30),
        xaxis=dict(
            showgrid=True, gridcolor="#1f2937",
            tickvals=tick_idxs,
            ticktext=[ts_labels[i] if i < len(ts_labels) else "" for i in tick_idxs],
            rangeslider_visible=False,
        ),
        yaxis=dict(showgrid=True, gridcolor="#1f2937"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Performance history ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_history(instrument: str, _day: str) -> list[tuple]:
    """_day (e.g. '2026-10-01') rotates cache daily."""
    return fetch_all_sessions(instrument, n_days=7)


_STOPPED_AT = {
    0: "No data",
    1: "No sweep",
    2: "No displacement",
    3: "No BOS",
    4: "No entry zone",
    5: "No retracement",
    6: "No retracement",
    7: "Trade open",
    8: "Trade open",
    9: "Closed",
}


def _history_row(label: str, df: pd.DataFrame, result: EngineResult,
                 instrument: str) -> dict:
    port = portfolio(result)
    s    = result.current_state

    if result.veto_reason:
        outcome, pnl = "Vetoed", 0.0
        stopped = "Vetoed"
    elif result.exit_reason == "TP_HIT":
        outcome, pnl = "TP Hit", port["tp_gain"]
        stopped = "TP Hit"
    elif result.exit_reason == "SL_HIT":
        outcome, pnl = "SL Hit", port["sl_loss"]
        stopped = "SL Hit"
    elif s >= 7:
        outcome, pnl = "Open", 0.0
        stopped = "Trade open"
    else:
        outcome, pnl = "No trade", 0.0
        stopped = _STOPPED_AT.get(s, f"State {s}")

    return {
        "session":   label,
        "direction": result.trade_direction or "—",
        "entry":     f"{result.entry_price:.2f}" if result.entry_price else "—",
        "sl":        f"{result.sl_price:.2f}"    if result.sl_price    else "—",
        "tp":        f"{result.tp_price:.2f}"    if result.tp_price    else "—",
        "grade":     result.confluence.grade if result.confluence else "—",
        "outcome":   outcome,
        "stopped":   stopped,
        "pnl":       round(pnl, 2),
    }


def _performance_section(instrument: str) -> None:
    today_str = datetime.now(NY).strftime("%Y-%m-%d")
    config    = INSTRUMENT_CONFIG[instrument]

    with st.spinner("Loading session history…"):
        sessions = load_history(instrument, today_str)

    if not sessions:
        st.caption("No historical session data available.")
        return

    rows = []
    for label, df_s, aeb in sessions:
        r = run_engine(df_s, None, config, len(df_s) - 1, aeb)
        rows.append(_history_row(label, df_s, r, instrument))

    # Running P&L
    running = 0.0
    running_series = []
    for row in rows:
        running += row["pnl"]
        running_series.append(running)

    # ── Equity sparkline ─────────────────────────────────────────────────────
    bar_colors = ["#34D399" if v >= 0 else "#F87171" for v in
                  [rows[i]["pnl"] for i in range(len(rows))]]
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Bar(
        x=[r["session"] for r in rows],
        y=[r["pnl"] for r in rows],
        marker_color=bar_colors,
        name="Session P&L",
        text=[f"${v:+.0f}" for v in [r["pnl"] for r in rows]],
        textposition="outside",
    ))
    fig_eq.add_trace(go.Scatter(
        x=[r["session"] for r in rows],
        y=running_series,
        mode="lines+markers",
        name="Running P&L",
        line=dict(color="#60A5FA", width=2),
        marker=dict(size=6),
        yaxis="y2",
    ))
    fig_eq.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        font=dict(color="#E5E7EB", size=11),
        height=220, margin=dict(l=50, r=50, t=20, b=30),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1f2937", title="Trade P&L ($)"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    title="Running ($)", title_font=dict(color="#60A5FA"),
                    tickfont=dict(color="#60A5FA")),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)"),
        bargap=0.3,
    )
    st.plotly_chart(fig_eq, use_container_width=True, key=f"eq_{instrument}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    trades   = [r for r in rows if r["outcome"] in ("TP Hit", "SL Hit")]
    wins     = [r for r in trades if r["outcome"] == "TP Hit"]
    losses   = [r for r in trades if r["outcome"] == "SL Hit"]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = sum(r["pnl"] for r in rows)
    pnl_color = "#34D399" if total_pnl >= 0 else "#F87171"

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Sessions", len(rows))
    s2.metric("Trades taken", len(trades))
    s3.metric("Win rate", f"{win_rate:.0f}%")
    s4.metric("Wins / Losses", f"{len(wins)} / {len(losses)}")
    s5.metric("Total P&L", f"${total_pnl:+,.2f}")

    # ── Trade table ───────────────────────────────────────────────────────────
    pnl_color_map = lambda p: "#34D399" if p > 0 else ("#F87171" if p < 0 else "#6B7280")
    outcome_icon = {
        "TP Hit":    "✅ TP Hit",
        "SL Hit":    "✗  SL Hit",
        "No trade":  "—",
        "Vetoed":    "⚠ Vetoed",
        "Open":      "📊 Open",
        "Trade open":"📊 Open",
    }

    html = """
<table style="width:100%;border-collapse:collapse;font-size:.88em">
<thead>
<tr style="color:#6B7280;border-bottom:1px solid #1f2937;text-align:left">
  <th style="padding:6px 10px">Session</th>
  <th>Dir</th>
  <th>Entry</th>
  <th>SL</th>
  <th>TP</th>
  <th>Grade</th>
  <th>Result</th>
  <th>Stopped at</th>
  <th style="text-align:right;padding-right:10px">P&amp;L ($)</th>
  <th style="text-align:right;padding-right:10px">Running</th>
</tr>
</thead>
<tbody>"""

    running2 = 0.0
    for row in rows:
        running2 += row["pnl"]
        rc  = pnl_color_map(row["pnl"])
        rrc = pnl_color_map(running2)
        pnl_str = f"+${row['pnl']:.2f}" if row["pnl"] > 0 else (f"-${abs(row['pnl']):.2f}" if row["pnl"] < 0 else "—")
        run_str = f"+${running2:.2f}" if running2 > 0 else (f"-${abs(running2):.2f}" if running2 < 0 else "$0")
        oi  = outcome_icon.get(row["outcome"], row["outcome"])
        gc  = {"A": "#34D399", "B": "#FCD34D", "C": "#F87171"}.get(row["grade"], "#6B7280")
        dc  = "#F87171" if row["direction"] == "SHORT" else ("#34D399" if row["direction"] == "LONG" else "#6B7280")
        # stopped-at: green if trade happened, amber if got close, red if early stop
        stopped = row["stopped"]
        sc = "#34D399" if row["pnl"] != 0 else ("#FCD34D" if stopped in ("No retracement", "No entry zone") else "#6B7280")
        html += f"""
<tr style="border-bottom:1px solid #1a1f2e">
  <td style="padding:7px 10px;color:#9CA3AF">{row['session']}</td>
  <td style="color:{dc};font-weight:600">{row['direction']}</td>
  <td style="color:#E5E7EB">{row['entry']}</td>
  <td style="color:#F87171">{row['sl']}</td>
  <td style="color:#34D399">{row['tp']}</td>
  <td style="color:{gc};font-weight:700">{row['grade']}</td>
  <td style="color:#D1D5DB">{oi}</td>
  <td style="color:{sc};font-size:.82em">{stopped}</td>
  <td style="text-align:right;padding-right:10px;font-weight:700;color:{rc}">{pnl_str}</td>
  <td style="text-align:right;padding-right:10px;color:{rrc}">{run_str}</td>
</tr>"""

    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


# ── Auto-refresh ──────────────────────────────────────────────────────────────
def auto_refresh_bar() -> None:
    key = "refresh_at"
    if key not in st.session_state:
        st.session_state[key] = time.time() + 60
    remaining = max(0, int(st.session_state[key] - time.time()))
    c1, c2 = st.columns([5, 1])
    with c1:
        st.progress(remaining / 60, text=f"Data refreshes in {remaining}s")
    with c2:
        if st.button("↺ Now"):
            st.cache_data.clear()
            st.session_state[key] = time.time() + 60
            st.rerun()
    if remaining == 0:
        st.cache_data.clear()
        st.session_state[key] = time.time() + 60
        st.rerun()


# ── Render one instrument tab ─────────────────────────────────────────────────
def render_tab(instrument: str, df: pd.DataFrame, result: EngineResult,
               asian_end_bar: int) -> None:
    port = portfolio(result)
    ur   = unrealised(result, df, port)
    now_ny = datetime.now(NY)

    # Header strip
    h1, h2, h3, h4 = st.columns([2, 2, 2, 2])
    with h1:
        st.metric("NY Time", now_ny.strftime("%H:%M:%S"))
    with h2:
        st.metric("Session", current_session_label())
    with h3:
        cur_price = f"{df['close'].iloc[-1]:.2f}" if not df.empty else "—"
        st.metric(instrument, cur_price)
    with h4:
        st.metric("Bars", len(df))

    st.divider()

    # Signal card + portfolio
    col_sig, col_port = st.columns([3, 2])
    with col_sig:
        _signal_card(result, port, ur, instrument)
        st.markdown("**Strategy sequence** — today's session")
        _sequence_tracker(result, df, asian_end_bar)
    with col_port:
        _portfolio_panel(result, port, ur, instrument)

    st.divider()

    # Chart
    fig = build_chart(df, result, instrument)
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{instrument}")

    # 5-day performance history
    st.divider()
    st.markdown("#### 5-Day Performance History")
    st.caption(r"Each session run through the same FSM · \$100 risk per trade · \$10,000 account")
    _performance_section(instrument)

    with st.expander("Engine config", expanded=False):
        st.json(INSTRUMENT_CONFIG[instrument].model_dump())


# ── Page ──────────────────────────────────────────────────────────────────────
st.markdown("## Kirk Strategy — Live Signals")
st.caption(
    r"Applies the 10-state sweep-reversal FSM to today's M5 session · "
    r"\$10,000 paper portfolio · 1% risk per trade (\$100) · no API key required"
)
auto_refresh_bar()

tick = get_tick()
tabs = st.tabs(INSTRUMENTS)
for tab, instrument in zip(tabs, INSTRUMENTS):
    with tab:
        with st.spinner(f"Loading {instrument}…"):
            df, asian_end_bar = load_data(instrument, tick)
        result = (
            run_engine(df, None, INSTRUMENT_CONFIG[instrument], len(df) - 1, asian_end_bar)
            if not df.empty else EngineResult()
        )
        render_tab(instrument, df, result, asian_end_bar)
