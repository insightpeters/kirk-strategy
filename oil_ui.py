"""Oil Sweep-Break-Retest v1 — Streamlit research interface.

Run:
    streamlit run prototype/oil_ui.py

Data: yfinance CL=F — 15m and 5m bars up to 60 days.
For 12-month backtests, export from your broker and use the CSV import.

Limitation: yfinance does not provide USOUSD spot directly.
CL=F (WTI front-month futures) is used as the closest publicly available proxy.
"""
from __future__ import annotations

import datetime
import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oil_config import OilConfig, list_presets, load_preset, save_preset
from oil_engine import OilResult, OilTicket, OIL_STATES, run_oil_engine

NY  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

st.set_page_config(
    page_title="Oil SBR v1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.sbr-card { border-radius:10px; padding:14px 18px; margin-bottom:10px; }
.sbr-watch   { background:#0f172a; border:1px solid #374151; }
.sbr-active  { background:#2d0a0a; border:2px solid #DC2626; }
.sbr-closed  { background:#111827; border:1px solid #4B5563; }
.sbr-stat { text-align:center; padding:10px; background:#111827;
            border-radius:8px; margin:4px; }
.sbr-stat-val  { font-size:1.6em; font-weight:800; }
.sbr-stat-lbl  { font-size:.72em; color:#6B7280; text-transform:uppercase; }
.sbr-tbl { width:100%; border-collapse:collapse; font-family:monospace; font-size:.84em; }
.sbr-tbl th { color:#6B7280; border-bottom:1px solid #1f2937; padding:5px 8px; text-align:left; font-size:.78em; }
.sbr-tbl td { padding:5px 8px; border-bottom:1px solid #0d1117; }
.nar-msg { margin-bottom:6px; padding:9px 12px; border-radius:6px;
           border-left:4px solid #374151; background:#111827; font-family:monospace; }
.nar-info    { border-left-color:#3B82F6; }
.nar-pass    { border-left-color:#10B981; }
.nar-fail    { border-left-color:#EF4444; background:#180909; }
.nar-wait    { border-left-color:#F59E0B; }
.nar-skip    { border-left-color:#78716C; background:#0d1117; }
.nar-entry   { border-left-color:#FBBF24; background:#181200; }
.nar-exit-tp { border-left-color:#10B981; background:#051a0a; }
.nar-exit-sl { border-left-color:#EF4444; background:#180909; }
.nar-time { font-size:.76em; color:#4B5563; margin-right:8px; }
.nar-tf   { font-size:.68em; color:#374151; margin-right:6px; }
.nar-head { font-weight:700; font-size:.88em; }
.nar-detail { font-size:.76em; color:#6B7280; margin-top:3px; padding-left:18px; line-height:1.5; }
.nar-detail::before { content:"↳ "; color:#374151; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_yf_bars(interval: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    end   = datetime.datetime.now(UTC)
    start = end - datetime.timedelta(days=days)
    df = yf.download(
        "CL=F",
        start=start.strftime("%Y-%m-%d"),
        end=(end + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        return pd.DataFrame()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def load_csv_bars(uploaded) -> pd.DataFrame:
    """Parse a broker-exported CSV.  Expects columns: datetime, open, high, low, close (or similar)."""
    try:
        df = pd.read_csv(uploaded, parse_dates=True)
        # Try to find a datetime-like column
        dt_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if dt_candidates:
            df = df.rename(columns={dt_candidates[0]: "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.set_index("datetime")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                st.error(f"CSV missing column '{col}'")
                return pd.DataFrame()
        return df[["open", "high", "low", "close"]].sort_index()
    except Exception as e:
        st.error(f"CSV parse failed: {e}")
        return pd.DataFrame()


# ── Session date helpers ──────────────────────────────────────────────────────

def get_session_dates(df_15m: pd.DataFrame, cfg: OilConfig) -> list[datetime.date]:
    """Return unique NY dates that have at least 1 bar in the entry window."""
    if df_15m.empty:
        return []
    ny_idx = df_15m.index.tz_convert(NY)
    # Entry window is entry_start_hour – entry_end_hour
    entry_mask = (ny_idx.hour >= cfg.entry_start_hour) & (ny_idx.hour < cfg.entry_end_hour)
    dates = sorted(set(ny_idx[entry_mask].date))
    return dates


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    cfg: OilConfig,
    holdout_days: int = 0,
) -> list[tuple[datetime.date, OilResult]]:
    """Run engine over all session dates.  Returns (date, result) pairs."""
    dates = get_session_dates(df_15m, cfg)
    if not dates:
        return []

    results: list[tuple[datetime.date, OilResult]] = []
    daily_baskets: dict[datetime.date, int]  = {}
    daily_losers:  dict[datetime.date, int]  = {}
    daily_pnl:     dict[datetime.date, float] = {}

    for date_ny in dates:
        d_baskets = daily_baskets.get(date_ny, 0)
        d_losers  = daily_losers.get(date_ny, 0)
        d_pnl     = daily_pnl.get(date_ny, 0.0)

        # Daily limits check
        if d_baskets >= cfg.max_baskets_per_day:
            r = OilResult(); r.skip_reason = "daily basket limit reached"
            results.append((date_ny, r)); continue
        if d_losers >= cfg.max_losers_per_day:
            r = OilResult(); r.skip_reason = "daily loser limit reached"
            results.append((date_ny, r)); continue
        if d_pnl < -(cfg.account_equity * cfg.max_daily_loss_pct):
            r = OilResult(); r.skip_reason = "daily loss limit reached"
            results.append((date_ny, r)); continue

        result = run_oil_engine(df_15m, df_5m, date_ny, cfg)
        results.append((date_ny, result))

        if result.current_state == 4:  # had a trade
            daily_baskets[date_ny] = d_baskets + 1
            if result.net_pnl_usd < 0:
                daily_losers[date_ny] = d_losers + 1
            daily_pnl[date_ny] = d_pnl + result.net_pnl_usd

    return results


# ── Stats computation ─────────────────────────────────────────────────────────

def compute_stats(results: list[tuple[datetime.date, OilResult]], cost_mult: float = 1.0) -> dict:
    traded = [(d, r) for d, r in results if r.current_state == 4]
    if not traded:
        return {"n_setups": 0, "n_traded": 0}

    n_setups   = len(results)
    n_traded   = len(traded)
    n_rejected = sum(1 for _, r in results if r.skip_reason)

    net_pnls = [r.gross_pnl_usd - r.costs_usd * cost_mult for _, r in traded]
    wins   = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]
    win_rate = len(wins) / n_traded if n_traded else 0

    total_net = sum(net_pnls)
    avg_win   = sum(wins)   / len(wins)   if wins   else 0
    avg_loss  = sum(losses) / len(losses) if losses else 0
    pf = abs(sum(wins) / sum(losses)) if losses else float("inf")

    # Drawdown
    running = 0.0; peak = 0.0; max_dd = 0.0
    for p in net_pnls:
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Longest losing streak
    streak = 0; max_streak = 0
    for p in net_pnls:
        if p <= 0:
            streak += 1; max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "n_setups": n_setups, "n_traded": n_traded, "n_rejected": n_rejected,
        "win_rate": win_rate, "total_net": total_net,
        "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": pf,
        "max_drawdown": max_dd, "max_losing_streak": max_streak,
        "net_pnls": net_pnls,
    }


def monthly_breakdown(results: list[tuple[datetime.date, OilResult]], cost_mult: float = 1.0) -> pd.DataFrame:
    rows = []
    for date_ny, r in results:
        if r.current_state != 4:
            continue
        net = r.gross_pnl_usd - r.costs_usd * cost_mult
        rows.append({"month": date_ny.strftime("%Y-%m"), "net": net,
                     "win": 1 if net > 0 else 0, "loss": 1 if net <= 0 else 0})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby("month").agg(
        trades=("net", "count"), net_pnl=("net", "sum"),
        wins=("win", "sum"), losses=("loss", "sum"),
    ).reset_index()
    df["win_rate"] = (df["wins"] / df["trades"] * 100).round(1)
    df["net_pnl"]  = df["net_pnl"].round(2)
    return df


def rejected_table(results: list[tuple[datetime.date, OilResult]]) -> pd.DataFrame:
    rows = []
    for date_ny, r in results:
        if r.skip_reason:
            rows.append({"date": str(date_ny), "reason": r.skip_reason,
                         "swept_level": r.swept_key or "—",
                         "got_to_state": OIL_STATES.get(r.current_state, str(r.current_state))})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Chart ────────────────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "Yesterday H": "#9CA3AF",
    "Session H":   "#60A5FA",
    "Extended H":  "#F97316",
}


def build_oil_chart(df_15m: pd.DataFrame, df_5m: pd.DataFrame,
                    result: OilResult, date_ny: datetime.date, cfg: OilConfig) -> go.Figure:
    from oil_engine import _session_bounds
    bounds = _session_bounds(date_ny, cfg)

    chart_start = bounds["asian_start"]
    chart_end   = bounds["eod"]

    slice_15 = df_15m[(df_15m.index >= chart_start) & (df_15m.index <= chart_end)]
    slice_5  = df_5m[ (df_5m.index  >= chart_start) & (df_5m.index  <= chart_end)]

    # Use 5m as base for better resolution in entry zone
    base_df = slice_5 if not slice_5.empty else slice_15
    n = len(base_df)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=450)
        return fig

    ts_labels = [pd.Timestamp(t).tz_convert(NY).strftime("%H:%M") for t in base_df.index]
    tick_idxs = list(range(0, n, 12))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=list(range(n)),
        open=base_df["open"], high=base_df["high"],
        low=base_df["low"],   close=base_df["close"],
        name="USOIL",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        line_width=1, whiskerwidth=0.4,
    ))

    def _bar_idx(ts_utc) -> int | None:
        tgt = pd.Timestamp(ts_utc)
        diffs = abs(base_df.index - tgt)
        if diffs.min().total_seconds() > 900:
            return None
        return int(diffs.argmin())

    # Key levels
    for lk, lv in result.eligible_highs.items():
        color = LEVEL_COLORS.get(lk, "#6B7280")
        fig.add_shape(type="line", x0=0, x1=n - 1, y0=lv, y1=lv,
                      line=dict(color=color, width=1, dash="dash"))
        fig.add_annotation(x=n - 1, y=lv, text=f" {lk} {lv:.3f}",
                           xanchor="left", showarrow=False, font=dict(size=8, color=color))

    # Sweep annotation
    if result.sweep_ts and result.sweep_extreme:
        # find bar index by converting sweep_ts back (approximate)
        sweep_level = result.swept_level or 0
        fig.add_shape(type="line", x0=0, x1=n - 1, y0=sweep_level, y1=sweep_level,
                      line=dict(color="#A78BFA", width=1.5, dash="dashdot"))
        fig.add_annotation(x=n // 3, y=sweep_level,
                           text=f"Sweep level {sweep_level:.3f}", showarrow=False,
                           yanchor="bottom", font=dict(size=8, color="#A78BFA"))
        fig.add_annotation(x=n // 3, y=result.sweep_extreme,
                           text=f"SWEEP extreme {result.sweep_extreme:.3f}",
                           showarrow=True, arrowhead=2, arrowcolor="#A78BFA",
                           ax=0, ay=-22, font=dict(size=8, color="#A78BFA"))

    # Swing ref
    if result.swing_ref_price:
        fig.add_shape(type="line", x0=0, x1=n - 1,
                      y0=result.swing_ref_price, y1=result.swing_ref_price,
                      line=dict(color="#34D399", width=1, dash="dot"))
        fig.add_annotation(x=2, y=result.swing_ref_price, text="Swing ref",
                           xanchor="left", showarrow=False, font=dict(size=8, color="#34D399"))

    # Retrace zone
    if result.retrace_lo and result.retrace_hi and result.disp_low:
        fig.add_shape(type="rect", x0=n // 2, x1=n - 1,
                      y0=result.retrace_lo, y1=result.retrace_hi,
                      fillcolor="rgba(251,191,36,0.12)",
                      line=dict(color="#FBBF24", width=1, dash="dot"))
        fig.add_annotation(x=(n // 2 + n) // 2, y=result.retrace_hi,
                           text=f"Retrace 50–75% {result.retrace_lo:.3f}–{result.retrace_hi:.3f}",
                           yanchor="bottom", showarrow=False, font=dict(size=8, color="#FBBF24"))

    # Entry / SL / TP lines
    if result.entry_price and result.tickets:
        for px, color, label in [
            (result.entry_price, "#FBBF24", "ENTRY"),
            (result.sl_price,    "#F87171", "SL"),
        ] + [(t.tp, "#34D399" if t.tp_r <= 2 else "#10B981", f"T{t.idx+1} {t.tp_r}R")
             for t in result.tickets]:
            if px is None:
                continue
            fig.add_shape(type="line", x0=n // 2, x1=n - 1, y0=px, y1=px,
                          line=dict(color=color, width=1.2 if label in ("ENTRY", "SL") else 0.8,
                                    dash="solid" if label == "ENTRY" else "dash"))
            fig.add_annotation(x=n - 1, y=px, text=f" {label}",
                               xanchor="left", showarrow=False,
                               font=dict(size=8, color=color))

        # Basket risk box
        if result.sl_price:
            fig.add_shape(type="rect", x0=n // 2, x1=n - 1,
                          y0=result.entry_price, y1=result.sl_price,
                          fillcolor="rgba(239,68,68,0.06)",
                          line=dict(color="rgba(239,68,68,0.3)", width=0))

    # Ticket exits
    for t in result.tickets:
        if t.exit_ts and t.exit_price:
            col = "#34D399" if t.status == "TP" else "#F87171"
            icon = "✅" if t.status == "TP" else ("🕐" if t.status == "EOD" else "✗")
            # Can't map exit_ts string → bar index precisely; skip annotation if unclear
            # (would need exit bar index stored on ticket for precision)

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        font=dict(color="#E5E7EB", size=11),
        height=450, margin=dict(l=50, r=180, t=24, b=30),
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


# ── Narrative feed ────────────────────────────────────────────────────────────

def _oil_narrative(result: OilResult) -> None:
    if not result.narrative:
        st.info("No narrative — no events generated for this session.")
        return
    _ICONS = {"info":"◉","pass":"✅","fail":"✗","wait":"⏳","skip":"⊘",
               "entry":"◆","exit_tp":"🏁","exit_sl":"✗"}
    _COLORS = {"info":"#60A5FA","pass":"#34D399","fail":"#F87171","wait":"#FCD34D",
                "skip":"#78716C","entry":"#FBBF24","exit_tp":"#34D399","exit_sl":"#F87171"}
    parts = []
    for ev in result.narrative:
        icon = _ICONS.get(ev.status, "·")
        col  = _COLORS.get(ev.status, "#E5E7EB")
        css  = f"nar-{ev.status.replace('_','-')}"
        dets = "".join(f'<div class="nar-detail">{d}</div>' for d in ev.details)
        parts.append(f"""
<div class="nar-msg {css}">
  <span class="nar-time">{ev.ts}</span>
  <span class="nar-tf">[{ev.tf}]</span>
  <span class="nar-head" style="color:{col}">{icon} {ev.headline}</span>
  {dets}
</div>""")
    st.markdown("\n".join(parts), unsafe_allow_html=True)


# ── Basket table ──────────────────────────────────────────────────────────────

def _basket_table(result: OilResult, latest_price: float | None = None) -> None:
    if not result.tickets:
        st.caption("No basket."); return
    rows = []
    for t in result.tickets:
        if t.status == "OPEN" and latest_price:
            live_pnl = (t.entry - latest_price) * t.lots * 1000
            pnl_s = f"${live_pnl:+.2f} live"
            pc = "#34D399" if live_pnl >= 0 else "#F87171"
            sc = "#FCD34D"
        else:
            pnl_s = f"${t.pnl_gross:+.2f}"
            pc = "#34D399" if t.pnl_gross >= 0 else "#F87171"
            sc = "#34D399" if t.status == "TP" else ("#F87171" if t.status in ("SL","RUNNER_SL") else "#6B7280")
        tc = "#34D399" if t.tp_r <= 2.0 else "#10B981"
        rows.append(
            f'<tr><td style="color:#9CA3AF">#{t.idx+1}</td>'
            f'<td style="color:{tc};font-weight:700">{t.tp_r}R</td>'
            f'<td>{"★" if t.runner else ""}{t.lots:.2f}</td>'
            f'<td style="color:#FCD34D">{t.entry:.3f}</td>'
            f'<td style="color:#F87171">{t.sl:.3f}</td>'
            f'<td style="color:{tc}">{t.tp:.3f}</td>'
            f'<td style="color:{sc}">{t.status}</td>'
            f'<td style="color:#9CA3AF;font-size:.8em">{t.exit_ts or "—"}</td>'
            f'<td style="color:{pc};font-weight:700;text-align:right">{pnl_s}</td></tr>'
        )
    html = ('<table class="sbr-tbl"><thead><tr>'
            '<th>#</th><th>Target</th><th>Lots</th>'
            '<th>Entry</th><th>SL</th><th>TP</th>'
            '<th>Status</th><th>Time</th><th style="text-align:right">P&amp;L</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')
    st.markdown(html, unsafe_allow_html=True)


# ── Stat card grid ────────────────────────────────────────────────────────────

def _stat_grid(stats: dict, label: str = "") -> None:
    if stats.get("n_traded", 0) == 0:
        st.warning("No completed trades to report.")
        return
    if label:
        st.caption(label)
    c = st.columns(5)
    metrics = [
        ("Setups", stats["n_setups"]),
        ("Traded", stats["n_traded"]),
        (f"Win rate", f"{stats['win_rate']:.0%}"),
        ("Total net", f"${stats['total_net']:+.2f}"),
        ("Max drawdown", f"${stats['max_drawdown']:.2f}"),
    ]
    for col, (lbl, val) in zip(c, metrics):
        col.metric(lbl, val)
    c2 = st.columns(5)
    metrics2 = [
        ("Avg win", f"${stats['avg_win']:+.2f}"),
        ("Avg loss", f"${stats['avg_loss']:+.2f}"),
        ("Profit factor", f"{stats['profit_factor']:.2f}" if stats["profit_factor"] != float("inf") else "∞"),
        ("Losing streak", stats["max_losing_streak"]),
        ("Rejected", stats["n_rejected"]),
    ]
    for col, (lbl, val) in zip(c2, metrics2):
        col.metric(lbl, val)


# ── Equity curve ──────────────────────────────────────────────────────────────

def _equity_curve(net_pnls: list[float]) -> go.Figure:
    cumulative = [sum(net_pnls[:i+1]) for i in range(len(net_pnls))]
    bar_colors = ["#34D399" if p >= 0 else "#F87171" for p in net_pnls]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(net_pnls))), y=net_pnls,
                         marker_color=bar_colors, name="Trade P&L"))
    fig.add_trace(go.Scatter(x=list(range(len(net_pnls))), y=cumulative,
                             mode="lines+markers", name="Running P&L",
                             line=dict(color="#60A5FA", width=2),
                             marker=dict(size=5), yaxis="y2"))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        font=dict(color="#E5E7EB", size=11),
        height=220, margin=dict(l=50, r=50, t=20, b=30),
        yaxis=dict(showgrid=True, gridcolor="#1f2937", title="Trade P&L ($)"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    title="Running ($)", title_font=dict(color="#60A5FA"),
                    tickfont=dict(color="#60A5FA")),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)"),
        bargap=0.3,
    )
    return fig


# ── Single vs 8-ticket comparison ────────────────────────────────────────────

def _compare_table(results_8: list[tuple[datetime.date, OilResult]],
                   results_1: list[tuple[datetime.date, OilResult]]) -> None:
    s8 = compute_stats(results_8)
    s1 = compute_stats(results_1)
    if s8.get("n_traded", 0) == 0:
        st.info("No trades to compare."); return

    rows = [
        ("Setups", s1.get("n_setups",0), s8.get("n_setups",0)),
        ("Traded", s1.get("n_traded",0), s8.get("n_traded",0)),
        ("Win rate", f"{s1.get('win_rate',0):.0%}", f"{s8.get('win_rate',0):.0%}"),
        ("Total net", f"${s1.get('total_net',0):+.2f}", f"${s8.get('total_net',0):+.2f}"),
        ("Profit factor", f"{s1.get('profit_factor',0):.2f}", f"{s8.get('profit_factor',0):.2f}"),
        ("Max drawdown", f"${s1.get('max_drawdown',0):.2f}", f"${s8.get('max_drawdown',0):.2f}"),
    ]
    html = ('<table class="sbr-tbl"><thead><tr>'
            '<th>Metric</th><th>1 Ticket</th><th>8 Tickets</th></tr></thead><tbody>')
    for row in rows:
        html += f'<tr><td style="color:#9CA3AF">{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


# ── Sidebar: parameters ───────────────────────────────────────────────────────

def _sidebar_params() -> OilConfig:
    st.sidebar.markdown("### Oil SBR v1")

    presets = list_presets()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sel = st.selectbox("Load preset", ["(default)"] + presets, key="oil_preset_sel")
    with col2:
        pname = st.text_input("Save name", "v1_baseline", key="oil_preset_name")

    loaded_cfg = None
    if sel != "(default)" and st.sidebar.button("Load ▲", key="oil_load"):
        try:
            loaded_cfg = load_preset(sel)
            st.sidebar.success(f"Loaded '{sel}'")
        except Exception as e:
            st.sidebar.error(str(e))

    # Build defaults from loaded preset or factory defaults
    base = loaded_cfg or OilConfig()

    st.sidebar.divider()

    with st.sidebar.expander("Session & Range", expanded=True):
        range_mult = st.slider("Range min ATR mult", 0.5, 3.0, base.range_min_atr_mult, 0.1,
                               help="Asian/extended range must be ≥ this × ATR")
        entry_end = st.slider("Entry window end (NY hour)", 9, 16, base.entry_end_hour)
        eod_h = st.slider("EOD close hour (NY)", 14, 17, base.eod_hour)

    with st.sidebar.expander("Sweep"):
        sweep_mult = st.slider("Sweep ATR mult", 0.05, 0.50, base.sweep_atr_mult, 0.01,
                               help="Min sweep = max(mult×ATR, 2×spread)")
        sweep_bars = st.slider("Reclaim bars (15m)", 1, 6, base.sweep_close_bars)

    with st.sidebar.expander("BOS & Displacement"):
        bos_bars   = st.slider("BOS window (15m bars)", 1, 8, base.bos_bars)
        disp_body  = st.slider("Displacement body ATR mult", 0.3, 2.0, base.disp_body_atr_mult, 0.05)
        disp_ratio = st.slider("Displacement body/range %", 0.30, 0.90, base.disp_body_range_pct, 0.05)
        disp_close = st.slider("Displacement close top %", 0.10, 0.50, base.disp_close_top_pct, 0.05)

    with st.sidebar.expander("Retracement (5m)"):
        ret_lo   = st.slider("Zone low %", 0.30, 0.70, base.retrace_pct_low, 0.05)
        ret_hi   = st.slider("Zone high %", 0.50, 0.90, base.retrace_pct_high, 0.05)
        ret_bars = st.slider("Retrace window (5m bars)", 3, 12, base.retrace_bars_5m)

    with st.sidebar.expander("Stop & Risk"):
        sl_buf  = st.slider("SL buffer ATR mult", 0.05, 0.50, base.sl_buffer_atr_mult, 0.05)
        sl_max  = st.slider("Max SL ATR mult", 1.0, 4.0, base.max_sl_atr_mult, 0.25)
        spread  = st.number_input("Spread (pts)", 0.0, 1.0, base.spread_pts, 0.005)
        max_tix = st.slider("Max tickets", 1, 8, base.max_tickets)

    with st.sidebar.expander("Exits"):
        tp1 = st.slider("TP1 R", 0.5, 3.0, base.tp1_r, 0.25)
        tp2 = st.slider("TP2 R", 1.0, 4.0, base.tp2_r, 0.25)
        tp3 = st.slider("TP3 R", 1.5, 6.0, base.tp3_r, 0.25)
        tp4 = st.slider("TP4 R", 2.0, 8.0, base.tp4_r, 0.25)
        runner = st.checkbox("Runner mode (trail last 2 after 2R)", base.use_runner)

    with st.sidebar.expander("Costs & News"):
        slip      = st.number_input("Slippage (pts)", 0.0, 0.5, base.slippage_pts, 0.005)
        commiss   = st.number_input("Commission $/lot/side", 0.0, 20.0, base.commission_per_lot, 0.5)
        eia_block = st.checkbox("Block EIA Wednesday 10:30 AM ET", base.use_eia_block)
        blk_before= st.slider("News block before (min)", 5, 60, base.news_block_before_min)
        blk_after = st.slider("News block after (min)", 10, 90, base.news_block_after_min)

    cfg = OilConfig(
        spread_pts=spread, range_min_atr_mult=range_mult,
        sweep_atr_mult=sweep_mult, sweep_close_bars=sweep_bars,
        bos_bars=bos_bars,
        disp_body_atr_mult=disp_body, disp_body_range_pct=disp_ratio,
        disp_close_top_pct=disp_close,
        retrace_pct_low=ret_lo, retrace_pct_high=ret_hi, retrace_bars_5m=ret_bars,
        sl_buffer_atr_mult=sl_buf, max_sl_atr_mult=sl_max,
        entry_end_hour=entry_end, eod_hour=eod_h,
        tp1_r=tp1, tp2_r=tp2, tp3_r=tp3, tp4_r=tp4,
        use_runner=runner, max_tickets=max_tix,
        slippage_pts=slip, commission_per_lot=commiss,
        use_eia_block=eia_block,
        news_block_before_min=blk_before, news_block_after_min=blk_after,
    )

    st.sidebar.divider()
    if st.sidebar.button("💾 Save preset", key="oil_save"):
        save_preset(pname, cfg)
        st.sidebar.success(f"Saved '{pname}'")

    return cfg


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("## Oil Sweep-Break-Retest v1")
st.caption(
    "Dual-timeframe sell setup (15m setup · 5m entry) · 8-ticket basket · "
    "Data: yfinance CL=F proxy or broker CSV · 1% risk / $100 per basket"
)

cfg = _sidebar_params()

tab_bt, tab_live, tab_import = st.tabs(["Backtest", "Today (live)", "Import CSV"])

# ── Backtest tab ──────────────────────────────────────────────────────────────
with tab_bt:
    st.markdown("#### Backtest — yfinance data (up to 60 days)")
    st.info(
        "yfinance provides ~60 days of 15m/5m data for CL=F (WTI crude futures). "
        "For a 12-month test, upload broker-exported CSV in the **Import CSV** tab."
    )

    col_bt1, col_bt2 = st.columns([2, 1])
    with col_bt1:
        days_back = st.slider("Data lookback (days)", 7, 59, 45)
    with col_bt2:
        run_btn = st.button("▶ Run backtest", type="primary", use_container_width=True)

    if run_btn or st.session_state.get("bt_results"):
        if run_btn:
            with st.spinner("Downloading data and running backtest…"):
                df_15 = load_yf_bars("15m", days_back)
                df_5  = load_yf_bars("5m",  days_back)
                if df_15.empty or df_5.empty:
                    st.error("Failed to fetch data. Check your internet connection.")
                    st.stop()
                results_8 = run_backtest(df_15, df_5, cfg)
                # 1-ticket comparison: max_tickets=1
                cfg_1 = cfg.model_copy(update={"max_tickets": 1})
                results_1 = run_backtest(df_15, df_5, cfg_1)
                st.session_state["bt_results"]   = results_8
                st.session_state["bt_results_1"] = results_1
                st.session_state["bt_df15"]      = df_15
                st.session_state["bt_df5"]       = df_5
            st.success("Done.")

        results_8 = st.session_state.get("bt_results", [])
        results_1 = st.session_state.get("bt_results_1", [])
        df_15     = st.session_state.get("bt_df15", pd.DataFrame())
        df_5      = st.session_state.get("bt_df5",  pd.DataFrame())

        if not results_8:
            st.warning("No sessions found. Try a longer lookback.")
            st.stop()

        st.divider()

        # ── Stats tabs
        stab_base, stab_2x, stab_compare = st.tabs(["Baseline", "2× Costs", "1 vs 8 Tickets"])

        with stab_base:
            stats = compute_stats(results_8)
            _stat_grid(stats, "8-ticket basket · baseline costs")
            if stats.get("net_pnls"):
                st.plotly_chart(_equity_curve(stats["net_pnls"]), use_container_width=True,
                                key="eq_base")

        with stab_2x:
            stats2 = compute_stats(results_8, cost_mult=cfg.worst_case_cost_mult
                                    if hasattr(cfg, "worst_case_cost_mult") else 2.0)
            _stat_grid(stats2, "8-ticket basket · 2× costs sensitivity")
            if stats2.get("net_pnls"):
                st.plotly_chart(_equity_curve(stats2["net_pnls"]), use_container_width=True,
                                key="eq_2x")

        with stab_compare:
            st.markdown("**Total exposure identical — only ticket count differs**")
            _compare_table(results_8, results_1)

        # ── Monthly breakdown
        st.divider()
        st.markdown("#### Monthly breakdown")
        mdf = monthly_breakdown(results_8)
        if not mdf.empty:
            st.dataframe(mdf.style.applymap(
                lambda v: "color:#34D399" if isinstance(v, (int, float)) and v > 0
                          else ("color:#F87171" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["net_pnl"]
            ), use_container_width=True, hide_index=True)

        # ── Session picker + chart
        st.divider()
        st.markdown("#### Session inspector")
        session_dates = [d for d, r in results_8]
        if session_dates:
            sel_date = st.selectbox(
                "Select date", session_dates,
                index=len(session_dates) - 1,
                format_func=lambda d: str(d),
                key="bt_sel_date",
            )
            sel_result = next((r for d, r in results_8 if d == sel_date), None)
            if sel_result:
                col_ch, col_nar = st.columns([3, 2])
                with col_ch:
                    st.plotly_chart(
                        build_oil_chart(df_15, df_5, sel_result, sel_date, cfg),
                        use_container_width=True, key="bt_chart",
                    )
                    if sel_result.n_tickets > 0:
                        st.markdown("**Basket**")
                        _basket_table(sel_result)
                with col_nar:
                    state_lbl = OIL_STATES.get(sel_result.current_state, "?")
                    st.markdown(f"**Engine log — {state_lbl}**")
                    if sel_result.skip_reason:
                        st.warning(f"Skip: {sel_result.skip_reason}")
                    with st.container(height=520):
                        _oil_narrative(sel_result)

        # ── Rejected setups
        st.divider()
        st.markdown("#### All rejected setups")
        rdf = rejected_table(results_8)
        if not rdf.empty:
            st.dataframe(rdf, use_container_width=True, hide_index=True)
        else:
            st.caption("No rejected setups in this dataset.")

# ── Live / Today tab ──────────────────────────────────────────────────────────
with tab_live:
    st.markdown("#### Today's session — live (5-min auto-refresh)")
    if st.button("↺ Refresh data", key="live_refresh"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching live data…"):
        df_15_live = load_yf_bars("15m", 5)
        df_5_live  = load_yf_bars("5m",  3)

    if df_15_live.empty:
        st.error("No live data.")
    else:
        today_ny = datetime.datetime.now(NY).date()
        result_live = run_oil_engine(df_15_live, df_5_live, today_ny, cfg)

        state_lbl = OIL_STATES.get(result_live.current_state, "?")
        card_cls = ("sbr-active" if result_live.current_state == 4
                    else ("sbr-closed" if result_live.current_state == 5
                          else "sbr-watch"))
        st.markdown(
            f'<div class="sbr-card {card_cls}">'
            f'<div style="font-size:1.3em;font-weight:800">{state_lbl}</div>'
            f'<div style="color:#9CA3AF;font-size:.88em">'
            f'{result_live.skip_reason or ("Entry @ " + str(result_live.entry_price) if result_live.entry_price else "Watching levels")}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if result_live.eligible_highs:
            pills = "  ·  ".join(f"{k}: {v:.3f}" for k, v in result_live.eligible_highs.items())
            st.caption(f"Eligible levels — {pills}")

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.plotly_chart(
                build_oil_chart(df_15_live, df_5_live, result_live, today_ny, cfg),
                use_container_width=True, key="live_chart",
            )
            if result_live.n_tickets > 0:
                latest = float(df_5_live["close"].iloc[-1]) if not df_5_live.empty else None
                st.markdown("**Basket**")
                _basket_table(result_live, latest)
        with col_r:
            st.markdown(f"**Engine log**")
            with st.container(height=500):
                _oil_narrative(result_live)

# ── CSV Import tab ────────────────────────────────────────────────────────────
with tab_import:
    st.markdown("#### Import broker-exported CSV for extended backtesting")
    st.markdown("""
**Format required:**
- One row per bar — sorted oldest → newest
- Required columns: `datetime` (or `date`/`time`), `open`, `high`, `low`, `close`
- Datetime must be UTC-aware **or** you can specify NY offset below
- Two separate files: one for 15m bars, one for 5m bars

This allows backtesting across 12+ months using your broker's own USOUSD data.
    """)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        f15 = st.file_uploader("15m bars CSV", type="csv", key="csv_15m")
    with col_up2:
        f5  = st.file_uploader("5m bars CSV",  type="csv", key="csv_5m")

    if f15 and f5:
        df_15_csv = load_csv_bars(f15)
        df_5_csv  = load_csv_bars(f5)
        st.success(f"Loaded {len(df_15_csv)} × 15m bars, {len(df_5_csv)} × 5m bars")
        st.write(f"Range: {df_15_csv.index[0]} → {df_15_csv.index[-1]}")

        if st.button("▶ Run CSV backtest", type="primary", key="csv_run"):
            with st.spinner("Running backtest on imported data…"):
                csv_results = run_backtest(df_15_csv, df_5_csv, cfg)
                st.session_state["csv_results"] = csv_results
                st.session_state["csv_df15"] = df_15_csv
                st.session_state["csv_df5"]  = df_5_csv

        if st.session_state.get("csv_results"):
            csv_res = st.session_state["csv_results"]
            stats_csv = compute_stats(csv_res)
            _stat_grid(stats_csv, "CSV data — baseline costs")

            mdf_csv = monthly_breakdown(csv_res)
            if not mdf_csv.empty:
                st.markdown("#### Monthly breakdown")
                st.dataframe(mdf_csv, use_container_width=True, hide_index=True)

            rdf_csv = rejected_table(csv_res)
            if not rdf_csv.empty:
                st.markdown("#### Rejected setups")
                st.dataframe(rdf_csv, use_container_width=True, hide_index=True)
