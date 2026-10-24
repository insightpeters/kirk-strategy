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
from engine import EngineResult, NarrativeEvent, compute_ema, run_engine
from reversal_engine import (
    run_reversal_engine, ReversalResult, REV_STATES, LEVEL_LABEL,
    SHORT_LEVEL_KEYS, LONG_LEVEL_KEYS,
)
from reversal_data import fetch_gold_bars, compute_key_levels

NY = ZoneInfo("America/New_York")
INSTRUMENTS = ["XAUUSD", "US30", "USOIL", "USDCAD", "GBPUSD"]

ACCOUNT_SIZE = 10_000.0
RISK_PCT     = 0.01          # 1% risk per trade = $100

UNIT_LABELS = {
    "XAUUSD": "oz",
    "US30":   "units",
    "USOIL":  "bbl",
    "USDCAD": "units",
    "GBPUSD": "units",
    "XRPUSD": "XRP",
}

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

/* ── Narrative feed ── */
.nar-msg {
    margin-bottom:8px; padding:10px 14px; border-radius:8px;
    border-left:4px solid #374151; background:#111827; font-family:monospace;
}
.nar-info   { border-left-color:#3B82F6; }
.nar-pass   { border-left-color:#10B981; }
.nar-fail   { border-left-color:#EF4444; background:#180909; }
.nar-wait   { border-left-color:#F59E0B; }
.nar-scan   { border-left-color:#4B5563; background:#0d1117; }
.nar-entry  { border-left-color:#FBBF24; background:#181200; }
.nar-exit-tp { border-left-color:#10B981; background:#051a0a; }
.nar-exit-sl { border-left-color:#EF4444; background:#180909; }
.nar-narrow  { border-left-color:#F97316; background:#1a0d00; }
.nar-time { font-size:.78em; color:#4B5563; margin-right:10px; }
.nar-icon { margin-right:6px; }
.nar-head { font-weight:700; font-size:.92em; }
.nar-detail {
    font-size:.80em; color:#6B7280; margin-top:4px; padding-left:20px;
    line-height:1.6;
}
.nar-detail::before { content:"↳ "; color:#374151; }

/* ── Gold Reversal tab ── */
.rev-card { border-radius:12px; padding:16px 20px; margin-bottom:12px; }
.rev-card-watch   { background:#0f172a; border:1px solid #374151; }
.rev-card-sweep   { background:#1f1030; border:2px solid #A78BFA; }
.rev-card-reject  { background:#1a0f00; border:2px solid #F97316; }
.rev-card-bos     { background:#061a14; border:2px solid #34D399; }
.rev-card-retest  { background:#141000; border:2px solid #FCD34D; }
.rev-card-basket  { background:#2d0a0a; border:2px solid #DC2626; }
.rev-card-partial { background:#0a1f08; border:2px solid #84CC16; }
.rev-card-closed  { background:#111827; border:1px solid #4B5563; }
.basket-tbl { width:100%; border-collapse:collapse; font-family:monospace; font-size:.84em; }
.basket-tbl th { color:#6B7280; border-bottom:1px solid #1f2937; padding:5px 8px;
                 text-align:left; font-weight:normal; text-transform:uppercase; font-size:.8em; }
.basket-tbl td { padding:5px 8px; border-bottom:1px solid #0d1117; }
.rev-level-grid { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 12px; }
.rev-lv { padding:3px 10px; border-radius:8px; font-family:monospace; font-size:.82em;
          background:#111827; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Data / cache ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_data(instrument: str, _ts: int) -> tuple[pd.DataFrame, int]:
    if not market_is_open(instrument):
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
                 instrument: str, is_live: bool = True) -> None:
    s = result.current_state
    ul = UNIT_LABELS.get(instrument, "units")

    # Determine card type + headline
    if is_live and not market_is_open(instrument):
        card_cls = "card-wait"
        status   = "🌙  MARKET CLOSED"
        closed_msg = {
            "XAUUSD": "Gold futures closed · reopens Sunday 6 PM NY",
            "US30":   "US30 futures closed · reopens Sunday 6 PM NY",
            "USOIL":  "Oil futures closed · reopens Sunday 6 PM NY",
            "USDCAD": "Forex closed · reopens Sunday 5 PM NY",
            "GBPUSD": "Forex closed · reopens Sunday 5 PM NY",
        }
        sub = closed_msg.get(instrument, "Market closed") + " · check the history below for last signal"
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
    elif s == 1 and result.range_filter_result == "NARROW_EXTENDED" and result.range_frozen_bar is None:
        card_cls = "card-forming"
        status   = "⬡  BUILDING EXTENDED RANGE — LONDON WINDOW ACTIVE"
        sub      = f"Narrow Asian range · extending high/low through London · sweeps blocked until range freezes"
    elif s == 1 and result.range_filter_result == "NARROW_EXTENDED" and result.range_frozen_bar is not None:
        card_cls, status = "card-watch", "⬡  EXTENDED RANGE FROZEN — WATCHING FOR SWEEP"
        sub = f"Combined High: {result.asian_high:.2f}  ·  Low: {result.asian_low:.2f}  (Asia + London)"
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
    """_day (e.g. '2026-10-01') rotates cache daily. Returns up to 10 sessions / 14 days."""
    return fetch_all_sessions(instrument, n_days=14)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_median_range(instrument: str, _day: str) -> float | None:
    """Median Asian-session range across available historical sessions (up to 10).
    Used as the baseline for the range quality filter.  Returns None if < 3 sessions."""
    sessions = load_history(instrument, _day)
    if len(sessions) < 3:
        return None
    ranges: list[float] = []
    for _, df_s, aeb in sessions:
        if len(df_s) >= aeb and aeb > 0:
            ab = df_s.iloc[:aeb]
            r  = float(ab["high"].max()) - float(ab["low"].min())
            if r > 0:
                ranges.append(r)
    if len(ranges) < 3:
        return None
    ranges.sort()
    mid = len(ranges) // 2
    return (ranges[mid - 1] + ranges[mid]) / 2 if len(ranges) % 2 == 0 else ranges[mid]


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
    dp   = INSTRUMENT_CONFIG[instrument].price_decimals
    fmt  = f".{dp}f"

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
        if s == 1 and result.range_filter_result == "NARROW_EXTENDED" and result.range_frozen_bar is None:
            stopped = "London forming"
        elif s == 1 and result.range_filter_result == "NARROW_EXTENDED":
            stopped = "No sweep (extended)"
        else:
            stopped = _STOPPED_AT.get(s, f"State {s}")

    return {
        "session":   label,
        "direction": result.trade_direction or "—",
        "entry":     format(result.entry_price, fmt) if result.entry_price else "—",
        "sl":        format(result.sl_price,    fmt) if result.sl_price    else "—",
        "tp":        format(result.tp_price,    fmt) if result.tp_price    else "—",
        "grade":     result.confluence.grade if result.confluence else "—",
        "outcome":   outcome,
        "stopped":   stopped,
        "pnl":       round(pnl, 2),
    }


def _performance_section(instrument: str) -> None:
    config    = INSTRUMENT_CONFIG[instrument]
    today_str = datetime.now(NY).strftime("%Y-%m-%d")
    sessions  = load_history(instrument, today_str)   # last 14 days, up to 10 sessions
    med_range = compute_median_range(instrument, today_str)

    if not sessions:
        st.caption("No historical session data available.")
        return

    cfg_m = config.model_copy(update={"median_asian_range": med_range or 0.0})
    rows = []
    for label, df_s, aeb in sessions:
        r = run_engine(df_s, None, cfg_m, len(df_s) - 1, aeb)
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


# ── Narrative feed ───────────────────────────────────────────────────────────
def _narrative_feed(result: EngineResult, df: pd.DataFrame) -> None:
    """Render the engine's step-by-step reasoning as a chat-like timeline."""
    if not result.narrative:
        st.info("Waiting for data — narrative will appear once the Asian session closes.")
        return

    _ICONS = {
        "info":    "◉",
        "pass":    "✅",
        "fail":    "✗",
        "wait":    "⏳",
        "scan":    "↺",
        "entry":   "◆",
        "exit_tp": "🏁",
        "exit_sl": "✗",
        "narrow":  "⬡",
    }
    _HEAD_COLORS = {
        "info":    "#60A5FA",
        "pass":    "#34D399",
        "fail":    "#F87171",
        "wait":    "#FCD34D",
        "scan":    "#6B7280",
        "entry":   "#FBBF24",
        "exit_tp": "#34D399",
        "exit_sl": "#F87171",
        "narrow":  "#F97316",
    }

    html_parts = []
    for ev in result.narrative:
        ts   = _bar_ts(df, ev.bar_idx)
        icon = _ICONS.get(ev.status, "·")
        col  = _HEAD_COLORS.get(ev.status, "#E5E7EB")
        css  = f"nar-{ev.status.replace('_', '-')}"
        details_html = "".join(
            f'<div class="nar-detail">{d}</div>' for d in ev.details
        )
        html_parts.append(f"""
<div class="nar-msg {css}">
  <span class="nar-time">{ts} NY</span>
  <span class="nar-icon">{icon}</span>
  <span class="nar-head" style="color:{col}">{ev.headline}</span>
  {details_html}
</div>""")

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


# ── Week session loader ───────────────────────────────────────────────────────
def _week_sessions(instrument: str) -> list[tuple[str, "pd.DataFrame", int]]:
    """Sessions from Monday of the current trading week through today.

    Mon=1 session, Tue=2, … Fri/Sat/Sun=5 (full week).
    Never crosses the Monday boundary.
    """
    now = datetime.now(NY)
    wd  = now.weekday()   # 0=Mon … 6=Sun
    # How many sessions belong to this week
    if wd <= 4:           # Mon–Fri: wd+1 sessions (Mon=1, Tue=2, …)
        n_keep = wd + 1
    else:                 # Sat or Sun: full Mon–Fri week = 5 sessions
        n_keep = 5

    today_str = now.strftime("%Y-%m-%d")
    all_hist  = load_history(instrument, today_str)   # cached hourly
    return all_hist[-n_keep:] if len(all_hist) >= n_keep else all_hist


# ── Render ───────────────────────────────────────────────────────────────────
def render_tab(instrument: str, df_live: pd.DataFrame, result_live: EngineResult,
               asian_end_bar_live: int) -> None:
    config = INSTRUMENT_CONFIG[instrument]
    now_ny = datetime.now(NY)

    # ── Header strip (always live) ────────────────────────────────────────
    h1, h2, h3, h4 = st.columns([2, 2, 2, 2])
    with h1:
        st.metric("NY Time", now_ny.strftime("%H:%M:%S"))
    with h2:
        st.metric("Session", current_session_label())
    with h3:
        cur_price = f"{df_live['close'].iloc[-1]:.2f}" if not df_live.empty else "—"
        st.metric(instrument, cur_price)
    with h4:
        st.metric("Bars", len(df_live))

    st.divider()

    # ── Session picker ────────────────────────────────────────────────────
    week = _week_sessions(instrument)
    session_labels = [lbl for lbl, _, _ in week]

    if len(session_labels) > 1:
        selected = st.radio(
            "Session",
            options=session_labels,
            index=len(session_labels) - 1,
            horizontal=True,
            key=f"sess_{instrument}",
            label_visibility="collapsed",
        )
    elif session_labels:
        selected = session_labels[0]
        st.caption(f"Session: **{selected}**")
    else:
        selected = None

    # ── Resolve data for selected session ─────────────────────────────────
    today_str    = datetime.now(NY).strftime("%Y-%m-%d")
    med_range    = compute_median_range(instrument, today_str)
    live_is_open = market_is_open(instrument) and not df_live.empty
    is_live      = live_is_open and (selected == session_labels[-1] if session_labels else False)

    if is_live:
        df            = df_live
        result        = result_live
        asian_end_bar = asian_end_bar_live
    elif selected and week:
        cfg_m = config.model_copy(update={"median_asian_range": med_range or 0.0})
        for lbl, df_s, aeb in week:
            if lbl == selected:
                df            = df_s
                result        = run_engine(df_s, None, cfg_m, len(df_s) - 1, aeb)
                asian_end_bar = aeb
                break
        else:
            df, result, asian_end_bar = df_live, result_live, asian_end_bar_live
    else:
        df, result, asian_end_bar = df_live, result_live, asian_end_bar_live

    port = portfolio(result)
    ur   = unrealised(result, df, port) if is_live else None

    # ── Two-column layout: left = chart+info, right = scrollable narrative ─
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Signal card + portfolio side-by-side
        sig_col, port_col = st.columns([3, 2])
        with sig_col:
            _signal_card(result, port, ur, instrument, is_live=is_live)
            session_label_str = selected or "current"
            st.markdown(f"**Strategy sequence** — {session_label_str}")
            _sequence_tracker(result, df, asian_end_bar)
        with port_col:
            _portfolio_panel(result, port, ur, instrument)

        st.divider()

        # Chart
        if not is_live and selected:
            st.caption(f"Historical session · {selected} · all {len(df)} bars")
        fig = build_chart(df, result, instrument)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{instrument}")

        st.divider()
        st.markdown("#### Recent Performance — Last 2 Weeks")
        st.caption(r"Up to 10 sessions · \$100 risk per trade · \$10,000 account")
        _performance_section(instrument)

        with st.expander("Engine config", expanded=False):
            st.json(INSTRUMENT_CONFIG[instrument].model_dump())

    with col_right:
        session_label_str = selected or "current"
        st.markdown(f"**Bot Thinking — {session_label_str}**")
        st.caption("Every decision point in the order it happened · scroll to read all")
        with st.container(height=820):
            _narrative_feed(result, df)


# ── Reversal tab helpers ──────────────────────────────────────────────────────
LEVEL_COLORS: dict[str, str] = {
    "yesterday_high": "#9CA3AF",
    "yesterday_low":  "#9CA3AF",
    "asian_high":     "#60A5FA",
    "asian_low":      "#A78BFA",
    "london_high":    "#F97316",
    "london_low":     "#FCD34D",
    "session_high":   "#2DD4BF",
    "session_low":    "#818CF8",
}


def _rev_bar_ts(df_5m: pd.DataFrame, bar_idx: int | None) -> str:
    if bar_idx is None:
        return "—"
    try:
        ts = pd.Timestamp(df_5m.index[bar_idx])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(NY).strftime("%H:%M")
    except (IndexError, TypeError):
        return f"bar {bar_idx}"


def build_reversal_chart(
    df_5m: pd.DataFrame,
    result: ReversalResult,
    key_levels: dict[str, float],
    reveal_idx: int,
) -> go.Figure:
    if df_5m.empty:
        fig = go.Figure()
        fig.add_annotation(text="Waiting for data…", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=16, color="#4B5563"))
        fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=420)
        return fig

    df_slice = df_5m.iloc[:reveal_idx + 1]
    n = len(df_slice)
    x = list(range(n))
    ts_labels = [pd.Timestamp(t).tz_convert(NY).strftime("%H:%M") for t in df_slice.index]
    tick_idxs = list(range(0, n, 12))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=df_slice["open"], high=df_slice["high"],
        low=df_slice["low"], close=df_slice["close"],
        name="XAUUSD",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        line_width=1, whiskerwidth=0.4,
    ))

    for key, price in key_levels.items():
        color = LEVEL_COLORS.get(key, "#6B7280")
        label = LEVEL_LABEL.get(key, key)
        fig.add_shape(type="line", x0=0, x1=n - 1, y0=price, y1=price,
                      line=dict(color=color, width=1, dash="dash"))
        fig.add_annotation(x=n - 1, y=price, text=f" {label} {price:.2f}",
                           xanchor="left", showarrow=False,
                           font=dict(size=8, color=color))

    if result.sweep_bar is not None and result.sweep_extreme is not None:
        ay = -22 if result.direction == "SHORT" else 22
        fig.add_annotation(x=result.sweep_bar, y=result.sweep_extreme,
                           text="SWEEP", showarrow=True, arrowhead=2,
                           arrowcolor="#A78BFA", ax=0, ay=ay,
                           font=dict(size=9, color="#A78BFA"))

    if result.swing_ref_bar is not None and result.swing_ref_price is not None:
        sy = result.swing_ref_price
        fig.add_shape(type="line", x0=result.swing_ref_bar, x1=n - 1,
                      y0=sy, y1=sy,
                      line=dict(color="#34D399", width=1, dash="dot"))
        fig.add_annotation(x=result.swing_ref_bar, y=sy,
                           text="Swing ref", xanchor="right", showarrow=False,
                           font=dict(size=8, color="#34D399"))

    if (result.retest_level is not None and result.retest_tolerance is not None
            and result.bos_bar is not None):
        rl, rt = result.retest_level, result.retest_tolerance
        fig.add_shape(type="rect", x0=result.bos_bar, x1=n - 1,
                      y0=rl - rt, y1=rl + rt,
                      fillcolor="rgba(251,191,36,0.10)",
                      line=dict(color="#FBBF24", width=1, dash="dot"))
        fig.add_annotation(x=(result.bos_bar + n) // 2, y=rl + rt,
                           text="Retest zone", yanchor="bottom", showarrow=False,
                           font=dict(size=8, color="#FBBF24"))
        fig.add_annotation(x=result.bos_bar, y=rl,
                           text="BOS", xanchor="left", showarrow=False,
                           font=dict(size=9, color="#10B981"))

    if result.entry_bar is not None and result.entry_price is not None:
        for px, color, label in [
            (result.entry_price, "#FBBF24", "ENTRY"),
            (result.sl_price,    "#F87171", "SL"),
            (result.tp1_price,   "#34D399", "TP1"),
            (result.tp2_price,   "#10B981", "TP2"),
        ]:
            if px is None:
                continue
            fig.add_shape(type="line", x0=result.entry_bar, x1=n - 1,
                          y0=px, y1=px,
                          line=dict(color=color, width=1.2,
                                    dash="solid" if label == "ENTRY" else "dash"))
            fig.add_annotation(x=result.entry_bar, y=px, text=label,
                               xanchor="right", showarrow=False,
                               font=dict(size=8, color=color))

    for tk in result.tickets:
        if tk.exit_bar >= 0:
            icon = "✅" if tk.status in ("TP1", "TP2") else "✗"
            col  = "#34D399" if tk.status in ("TP1", "TP2") else "#F87171"
            fig.add_annotation(x=tk.exit_bar, y=tk.exit_price,
                               text=f"{icon}{tk.status}", showarrow=True,
                               arrowhead=2, arrowcolor=col, ax=0, ay=-18,
                               font=dict(size=8, color=col))

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        font=dict(color="#E5E7EB", size=11),
        height=420, margin=dict(l=50, r=160, t=20, b=30),
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


def _reversal_signal_card(result: ReversalResult) -> None:
    s = result.current_state
    STATE_CARD = {
        0: ("rev-card-watch",   "◉  WATCHING — scanning key levels"),
        1: ("rev-card-sweep",   "⚡  SWEEP DETECTED — awaiting close-back rejection"),
        2: ("rev-card-reject",  "↓  REJECTED — awaiting displacement + BOS"),
        3: ("rev-card-bos",     "✂  BOS CONFIRMED — awaiting retest"),
        4: ("rev-card-basket",  f"◆  IN BASKET — {result.direction}"),
        5: ("rev-card-partial", "✅  PARTIAL EXIT — TP1 hit, second half running to TP2"),
        6: ("rev-card-closed",  "⏹  BASKET CLOSED"),
    }
    card_cls, status = STATE_CARD.get(s, ("rev-card-watch", f"State {s}"))

    sub = ""
    if result.skip_reason:
        sub = f"Last skip: {result.skip_reason}"
    elif s >= 4 and result.entry_price is not None:
        dir_color = "#F87171" if result.direction == "SHORT" else "#34D399"
        sub = (
            f'<span style="color:{dir_color};font-weight:700">{result.direction}</span>'
            f' @ {result.entry_price:.2f}  ·  SL {result.sl_price:.2f}'
            f'  ·  TP1 {result.tp1_price:.2f}  ·  TP2 {result.tp2_price:.2f}'
        )
    elif s == 1 and result.swept_level_price is not None:
        lbl = LEVEL_LABEL.get(result.swept_level_key or "", result.swept_level_key or "")
        sub = f"Swept {lbl} @ {result.swept_level_price:.2f}  ·  extreme: {result.sweep_extreme:.2f}"
    elif s in (2, 3) and result.swing_ref_price is not None:
        retest_str = f"  ·  retest zone: {result.retest_level:.2f}" if result.retest_level else ""
        sub = f"Swing ref: {result.swing_ref_price:.2f}{retest_str}"

    pnl_html = ""
    if result.tickets:
        total = result.realized_pnl_usd + result.open_pnl_usd
        pnl_color = "#34D399" if total >= 0 else "#F87171"
        pnl_html = (
            f'<div style="margin-top:10px">'
            f'<span style="color:{pnl_color};font-size:1.4em;font-weight:800">${total:+.2f}</span>'
            f'<span style="color:#6B7280;font-size:.8em;margin-left:8px">'
            f'realized ${result.realized_pnl_usd:+.2f} / open ${result.open_pnl_usd:+.2f}</span>'
            f'</div>'
            f'<div style="color:#9CA3AF;font-size:.82em;margin-top:6px">'
            f'{result.n_tickets} tickets · {result.total_lots:.2f} lots total'
            f' · basket risk ${result.basket_risk_usd:.2f}'
            f' · RR to TP2: {result.rr_available:.1f}R'
            f'</div>'
        )

    st.markdown(
        f'<div class="rev-card {card_cls}">'
        f'<div style="font-size:1.4em;font-weight:800;letter-spacing:.04em">{status}</div>'
        f'<div style="font-size:.9em;color:rgba(255,255,255,.65);margin-top:4px">{sub}</div>'
        f'{pnl_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _basket_table(result: ReversalResult, latest_price: float) -> None:
    if not result.tickets:
        st.caption("No basket yet.")
        return

    rows = []
    for idx, tk in enumerate(result.tickets):
        if tk.status == "OPEN":
            if result.direction == "SHORT":
                live_pnl = (tk.entry - latest_price) * tk.lots * 100
            else:
                live_pnl = (latest_price - tk.entry) * tk.lots * 100
            pnl_str = f"${live_pnl:+.2f} live"
            pnl_col = "#34D399" if live_pnl >= 0 else "#F87171"
            sc = "#FCD34D"
        else:
            pnl_str = f"${tk.pnl_usd:+.2f}"
            pnl_col = "#34D399" if tk.pnl_usd >= 0 else "#F87171"
            sc = "#34D399" if tk.status in ("TP1", "TP2") else "#F87171"

        target_color = "#34D399" if tk.tp_target == "TP1" else "#10B981"
        tp_price = tk.tp1 if tk.tp_target == "TP1" else tk.tp2
        rows.append(
            f'<tr>'
            f'<td style="color:#9CA3AF">#{idx + 1}</td>'
            f'<td style="color:{target_color};font-weight:700">{tk.tp_target}</td>'
            f'<td>{tk.lots:.2f}</td>'
            f'<td style="color:#FCD34D">{tk.entry:.2f}</td>'
            f'<td style="color:#F87171">{tk.sl:.2f}</td>'
            f'<td style="color:{target_color}">{tp_price:.2f}</td>'
            f'<td style="color:{sc};font-weight:600">{tk.status}</td>'
            f'<td style="color:{pnl_col};font-weight:700;text-align:right">{pnl_str}</td>'
            f'</tr>'
        )

    html = (
        '<table class="basket-tbl"><thead><tr>'
        '<th>#</th><th>Target</th><th>Lots</th>'
        '<th>Entry</th><th>SL</th><th>TP price</th><th>Status</th>'
        '<th style="text-align:right">P&amp;L</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _reversal_narrative_feed(result: ReversalResult, df_5m: pd.DataFrame) -> None:
    if not result.narrative:
        st.info("No narrative yet — waiting for data.")
        return

    _ICONS = {
        "info":    "◉",
        "pass":    "✅",
        "fail":    "✗",
        "wait":    "⏳",
        "scan":    "↺",
        "entry":   "◆",
        "exit_tp": "🏁",
        "exit_sl": "✗",
    }
    _HEAD_COLORS = {
        "info":    "#60A5FA",
        "pass":    "#34D399",
        "fail":    "#F87171",
        "wait":    "#FCD34D",
        "scan":    "#6B7280",
        "entry":   "#FBBF24",
        "exit_tp": "#34D399",
        "exit_sl": "#F87171",
    }

    html_parts = []
    for ev in result.narrative:
        ts      = _rev_bar_ts(df_5m, ev.bar_idx)
        icon    = _ICONS.get(ev.status, "·")
        col     = _HEAD_COLORS.get(ev.status, "#E5E7EB")
        css     = f"nar-{ev.status.replace('_', '-')}"
        details = "".join(f'<div class="nar-detail">{d}</div>' for d in ev.details)
        html_parts.append(f"""
<div class="nar-msg {css}">
  <span class="nar-time">{ts} NY</span>
  <span class="nar-icon">{icon}</span>
  <span class="nar-head" style="color:{col}">{ev.headline}</span>
  {details}
</div>""")

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def _load_gold_bars_cached(_tick: int) -> pd.DataFrame:
    try:
        return fetch_gold_bars(interval="5m", days=3)
    except Exception:
        return pd.DataFrame()


def render_reversal_tab() -> None:
    st.markdown("#### Gold HOD/LOD Reversal — Paper Trading")
    st.caption(
        "Kirk's 8-ticket basket setup · 5m bars (GC=F) · "
        "$10K account · 1% risk ($100 total basket) · TP1 @ 2R / TP2 @ liquidity"
    )

    df_5m = _load_gold_bars_cached(get_tick())

    if df_5m.empty:
        st.error("Gold data unavailable — yfinance may be rate-limited. Try refreshing in 60s.")
        return

    try:
        key_levels = compute_key_levels(df_5m)
    except Exception as e:
        st.error(f"Key level computation failed: {e}")
        key_levels = {}

    n = len(df_5m)
    reveal_idx = st.slider(
        "Replay to bar", min_value=20, max_value=n - 1, value=n - 1,
        key="rev_reveal",
        help="Drag left to replay the setup development bar by bar",
    )

    result = run_reversal_engine(
        df_5m, key_levels, reveal_idx,
        account_equity=ACCOUNT_SIZE,
        risk_pct=RISK_PCT,
        max_positions=8,
    )

    latest_price = float(df_5m["close"].iloc[reveal_idx])

    if key_levels:
        pills = "".join(
            f'<span class="rev-lv" style="color:{LEVEL_COLORS.get(k, "#6B7280")}">'
            f'{LEVEL_LABEL.get(k, k)}: {v:.2f}</span>'
            for k, v in key_levels.items()
        )
        st.markdown(f'<div class="rev-level-grid">{pills}</div>', unsafe_allow_html=True)
    else:
        st.warning("No key levels — check data has at least one prior session.")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        _reversal_signal_card(result)
        st.plotly_chart(
            build_reversal_chart(df_5m, result, key_levels, reveal_idx),
            use_container_width=True,
            key="rev_chart",
        )
        if result.n_tickets > 0:
            st.markdown("**Basket tickets**")
            _basket_table(result, latest_price)

    with col_right:
        state_label = REV_STATES.get(result.current_state, f"State {result.current_state}")
        st.markdown(f"**Engine Reasoning** — {state_label}")
        st.caption("Decision log · scroll down for most recent")
        with st.container(height=720):
            _reversal_narrative_feed(result, df_5m)


# ── Page ──────────────────────────────────────────────────────────────────────
st.markdown("## Kirk Strategy — Live Signals")
st.caption(
    r"Applies the 10-state sweep-reversal FSM to today's M5 session · "
    r"\$10,000 paper portfolio · 1% risk per trade (\$100) · no API key required"
)
auto_refresh_bar()

tick = get_tick()
today_str_main = datetime.now(NY).strftime("%Y-%m-%d")
tabs = st.tabs(["⬡ Gold HOD/LOD"] + list(INSTRUMENTS))
with tabs[0]:
    render_reversal_tab()
for tab, instrument in zip(tabs[1:], INSTRUMENTS):
    with tab:
        with st.spinner(f"Loading {instrument}…"):
            df_live, asian_end_bar_live = load_data(instrument, tick)
        med_range_live = compute_median_range(instrument, today_str_main)
        cfg_live = INSTRUMENT_CONFIG[instrument].model_copy(
            update={"median_asian_range": med_range_live or 0.0}
        )
        result_live = (
            run_engine(df_live, None, cfg_live, len(df_live) - 1, asian_end_bar_live)
            if not df_live.empty else EngineResult()
        )
        render_tab(instrument, df_live, result_live, asian_end_bar_live)
