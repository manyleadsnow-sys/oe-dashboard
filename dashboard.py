"""
Owner Earnings Dashboard — Streamlit App
Run with: streamlit run dashboard.py
"""

import streamlit as st
import json
import os
import subprocess
import threading
from datetime import datetime, date
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OE Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, .mono { font-family: 'IBM Plex Mono', monospace; }

/* Header */
.oe-header {
    background: linear-gradient(135deg, #0a0a0f 0%, #111827 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.oe-title { font-size: 2rem; font-weight: 800; color: #e2f0ff; margin: 0; }
.oe-subtitle { font-size: 0.85rem; color: #6b8fb5; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }

/* Metric card */
.metric-card {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1d4ed8, #7c3aed);
}
.metric-label {
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #4b6a8a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #c9dff5;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-diff-pos { color: #22c55e; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; }
.metric-diff-neg { color: #ef4444; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; }
.metric-diff-neu { color: #94a3b8; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; }

/* Ticker badge */
.ticker-badge {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 10px 16px;
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #93c5fd;
    margin: 4px;
    cursor: pointer;
}
.ticker-badge:hover { border-color: #3b82f6; color: #fff; }

/* Section header */
.section-header {
    border-left: 3px solid #1d4ed8;
    padding-left: 12px;
    margin: 20px 0 12px 0;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #93c5fd;
    font-family: 'IBM Plex Mono', monospace;
}

/* Bear market row */
.bear-row {
    background: #0d1117;
    border: 1px solid #2d1b1b;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #94a3b8;
}
.bear-drawdown { color: #ef4444; font-weight: 600; }

/* Error badge */
.error-badge {
    background: #1a0a0a;
    border: 1px solid #5c1a1a;
    border-radius: 6px;
    padding: 6px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #f87171;
    margin: 4px 0;
}

/* Streamlit overrides */
.stSelectbox > div { background: #0d1117 !important; }
div[data-testid="stSidebar"] { background: #080c12 !important; }
div[data-testid="stSidebar"] * { color: #93c5fd; }
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 20px;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = os.path.join(os.path.dirname(__file__), "oe_data.json")

SECTORS = {
    "AI & Technology":          ["AAPL","GOOG","META","MSFT","NVDA","PLTR","TSLA"],
    "Communication Services":   ["EA","NFLX","TMUS"],
    "Consumer Discretionary":   ["AMZN","CMG","CPRT","GRMN","LEN","MCD","ORLY","POOL","ROST","TSCO","ULTA"],
    "Consumer Staples":         ["BG","COST","HSY","KO","PEP","PG","PM","STZ","SYY","WMT"],
    "Energy":                   ["BKR","CVX","EOG","EPD","EXE","FANG","SLB","TPL","VLO","XOM"],
    "Financials":               ["BRK-B","ACGL","AIZ","AJG","AON","ARES","AXP","BAC","BLK","BRO",
                                  "C","CB","CBOE","CBRE","CINF","CME","CPAY","EG","ERIE","FICO",
                                  "GS","IBKR","ICE","JPM","KKR","MA","MCO","MSCI","NDAQ","PGR",
                                  "RJF","SPGI","TRV","V","VRSK","WFC","WRB"],
    "Health Care":              ["A","BSX","CI","ABT","COO","HCA","IDXX","IQV","ISRG","JNJ",
                                  "LLY","MCK","MRK","MTD","REGN","RMD","SYK","TECH","VRTX","WAT","WST","ZTS"],
    "Industrials":              ["WM","MO","ADP","AXON","CAT","CTAS","DE","EME","EMR","ETN",
                                  "FAST","FIX","GD","GE","GWW","HON","HWM","LMT","NOC","ODFL",
                                  "OTIS","PH","PWR","ROK","ROL","ROP","TDG","TT"],
    "Information Technology":   ["ACN","ADI","ADSK","AMAT","AMD","ANET","APH","CDNS","CSCO","FTNT",
                                  "IT","KLAC","LRCX","MCHP","MPWR","MSI","NXPI","ON","PTC","Q",
                                  "SNPS","TEL","TER","TTD","TXN","TYL","VRSN","WDAY"],
    "Materials":                ["APD","AVY","CRH","ECL","FSLR","LIN","MLM","NUE","SHW","STLD","VMC"],
    "Real Estate":              ["AMT","CSGP","EXR","PSA","SBAC","VICI"],
    "Utilities":                ["AEP","AWK","CEG","D","DUK","ETR","NEE","NRG","PEG","SO","SRE","VST","XEL"],
}

METRIC_LABELS = {
    "oe":          ("Owner Earnings", "$B"),
    "oe_yield":    ("OE Yield", "%"),
    "oe_multiple": ("OE Multiple (EV/OE)", "x"),
    "oe_growth":   ("OE Growth Rate (10yr CAGR)", "%"),
    "oe_peg":      ("OE-PEG Ratio", "x"),
    "epv":         ("Earnings Power Value", "$/share"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE) as f:
        return json.load(f)


def fmt_val(v, unit):
    if v is None:
        return "—"
    if unit == "$B":
        return f"${v:.2f}B"
    if unit == "%":
        return f"{v:.2f}%"
    if unit == "x":
        return f"{v:.2f}x"
    if unit == "$/share":
        return f"${v:.2f}"
    return str(v)


def diff_html(d):
    if d is None:
        return '<span class="metric-diff-neu">—</span>'
    sign = "▲" if d > 0 else "▼"
    cls  = "metric-diff-pos" if d > 0 else "metric-diff-neg"
    return f'<span class="{cls}">{sign} {abs(d):.1f}%</span>'


def metric_card(label, unit, current_val, peak_val, diff):
    curr_fmt = fmt_val(current_val, unit)
    peak_fmt = fmt_val(peak_val, unit)
    d_html   = diff_html(diff)
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{curr_fmt}</div>
      <div style="display:flex; gap:16px; margin-top:6px; font-size:0.75rem; font-family:'IBM Plex Mono',monospace; color:#4b6a8a;">
        <span>PEAK&nbsp;<span style="color:#93c5fd">{peak_fmt}</span></span>
        <span>vs PEAK&nbsp;{d_html}</span>
      </div>
    </div>"""


def render_ticker(symbol, data):
    if data.get("error") and not data.get("current"):
        st.markdown(f'<div class="error-badge">⚠ {symbol}: {data["error"]}</div>', unsafe_allow_html=True)
        return

    curr = data.get("current", {})
    peak = data.get("peak_since_oct2022", {})
    diff = data.get("vs_peak_diff", {})

    # Header row
    price_now   = curr.get("price", "—")
    price_peak  = peak.get("price", "—")
    peak_date   = peak.get("date", "—")
    updated     = data.get("last_updated", "")[:10]

    st.markdown(f"""
    <div style="background:#080c12; border:1px solid #1e2d3d; border-radius:10px; padding:14px 20px; margin-bottom:16px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <span style="font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#60a5fa;">{symbol}</span>
          <span style="margin-left:12px; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#4b6a8a;">{data.get('sector','')}</span>
        </div>
        <div style="text-align:right; font-family:'IBM Plex Mono',monospace; font-size:0.78rem;">
          <span style="color:#94a3b8;">Now <span style="color:#e2f0ff;">${price_now}</span></span>
          &nbsp;|&nbsp;
          <span style="color:#94a3b8;">Peak <span style="color:#fbbf24;">${price_peak}</span> on {peak_date}</span>
          &nbsp;|&nbsp;
          <span style="color:#374151;">Updated {updated}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    metric_items = list(METRIC_LABELS.items())
    for idx, (key, (label, unit)) in enumerate(metric_items):
        with cols[idx % 3]:
            st.markdown(metric_card(
                label, unit,
                curr.get(key), peak.get(key), diff.get(key)
            ), unsafe_allow_html=True)

    # Bear markets
    bears = data.get("bear_markets", [])
    if bears:
        with st.expander(f"🐻  Bear Market Periods ({len(bears)} detected)", expanded=False):
            for b in bears:
                td  = b.get("trough_date", "—")
                pd_ = b.get("peak_date", "—")
                dd  = b.get("drawdown_pct", 0)
                tp  = b.get("price", "—")
                pp  = b.get("peak_price", "—")
                bd  = b.get("vs_bear_peak_diff", {})

                st.markdown(f"""
                <div class="bear-row">
                  <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span>📅 Peak <b>{pd_}</b> @ <b>${pp}</b> → Trough <b>{td}</b> @ <b>${tp}</b></span>
                    <span class="bear-drawdown">▼ {abs(dd):.1f}% drawdown</span>
                  </div>
                  <div style="display:flex; flex-wrap:wrap; gap:10px;">
                """, unsafe_allow_html=True)

                parts = []
                for key, (label, unit) in METRIC_LABELS.items():
                    val  = fmt_val(b.get(key), unit)
                    d_v  = bd.get(key)
                    d_s  = diff_html(d_v)
                    parts.append(f'<span style="color:#6b8fb5;">{label}: <span style="color:#c9dff5;">{val}</span> {d_s}</span>')
                st.markdown(" &nbsp;|&nbsp; ".join(parts) + "</div></div>", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown("## ⚙️ Controls")

    data = load_data()
    last_run = "Never"
    if data:
        dates = [v.get("last_updated","") for v in data.values() if v.get("last_updated")]
        if dates:
            last_run = max(dates)[:16].replace("T", " ")

    st.sidebar.markdown(f"""
    <div style="background:#0d1117; border:1px solid #1e2d3d; border-radius:8px; padding:12px; margin-bottom:16px;">
      <div style="font-size:0.7rem; color:#4b6a8a; font-family:'IBM Plex Mono',monospace; text-transform:uppercase;">Last Calculated</div>
      <div style="font-size:0.85rem; color:#93c5fd; font-family:'IBM Plex Mono',monospace;">{last_run}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🔄  Recalculate Now"):
        with st.spinner("Running calculations for all tickers… (this takes ~5–10 min)"):
            try:
                subprocess.run(
                    ["python", os.path.join(os.path.dirname(__file__), "calculator.py")],
                    check=True, capture_output=True, timeout=1200
                )
                st.sidebar.success("Done! Refresh the page.")
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.sidebar.error(f"Error: {e.stderr.decode()[:200]}")
            except subprocess.TimeoutExpired:
                st.sidebar.error("Timed out after 20 min.")

    st.sidebar.markdown("---")

    # Sector filter
    all_sectors = ["All Sectors"] + list(SECTORS.keys())
    selected_sector = st.sidebar.selectbox("Filter by Sector", all_sectors)

    # Search
    search = st.sidebar.text_input("Search Ticker", "").strip().upper()

    # Metric highlight
    metric_keys = ["All"] + [v[0] for v in METRIC_LABELS.values()]
    highlight = st.sidebar.selectbox("Highlight Metric", metric_keys)

    # Sort
    sort_by = st.sidebar.selectbox("Sort Tickers By", [
        "Ticker (A-Z)", "OE Multiple (low→high)", "OE Yield (high→low)",
        "OE Growth (high→low)", "OE-PEG (low→high)"
    ])

    return data, selected_sector, search, highlight, sort_by


def get_filtered_tickers(selected_sector, search):
    if selected_sector == "All Sectors":
        tickers = [t for lst in SECTORS.values() for t in lst]
    else:
        tickers = SECTORS.get(selected_sector, [])
    if search:
        tickers = [t for t in tickers if search in t]
    return tickers


def sort_tickers(tickers, data, sort_by):
    def key_fn(sym):
        d = data.get(sym, {}).get("current", {})
        if sort_by == "OE Multiple (low→high)":
            return d.get("oe_multiple") or 9999
        if sort_by == "OE Yield (high→low)":
            return -(d.get("oe_yield") or 0)
        if sort_by == "OE Growth (high→low)":
            return -(d.get("oe_growth") or 0)
        if sort_by == "OE-PEG (low→high)":
            return d.get("oe_peg") or 9999
        return sym  # A-Z
    return sorted(tickers, key=key_fn)


# ── Summary table ──────────────────────────────────────────────────────────────

def render_summary_table(tickers, data):
    rows = []
    for sym in tickers:
        d = data.get(sym, {})
        curr = d.get("current", {})
        peak = d.get("peak_since_oct2022", {})
        diff = d.get("vs_peak_diff", {})
        row = {
            "Ticker":       sym,
            "Sector":       d.get("sector", "—"),
            "Price":        curr.get("price"),
            "Peak Price":   peak.get("price"),
            "OE ($B)":      curr.get("oe"),
            "OE Yield %":   curr.get("oe_yield"),
            "OE Multiple":  curr.get("oe_multiple"),
            "OE Growth %":  curr.get("oe_growth"),
            "OE-PEG":       curr.get("oe_peg"),
            "EPV $/sh":     curr.get("epv"),
            "Δ OE%":        diff.get("oe"),
            "Δ Yield%":     diff.get("oe_yield"),
            "Δ Multiple%":  diff.get("oe_multiple"),
            "Δ Growth%":    diff.get("oe_growth"),
            "Δ PEG%":       diff.get("oe_peg"),
            "Δ EPV%":       diff.get("epv"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    def color_diff(val):
        if pd.isna(val) or val is None:
            return "color: #4b6a8a"
        return "color: #22c55e" if val > 0 else "color: #ef4444"

    diff_cols = ["Δ OE%","Δ Yield%","Δ Multiple%","Δ Growth%","Δ PEG%","Δ EPV%"]
    styled = df.style.applymap(color_diff, subset=diff_cols).format({
        "Price": "${:.2f}", "Peak Price": "${:.2f}",
        "OE ($B)": "${:.2f}B", "OE Yield %": "{:.2f}%",
        "OE Multiple": "{:.2f}x", "OE Growth %": "{:.2f}%",
        "OE-PEG": "{:.2f}x", "EPV $/sh": "${:.2f}",
        **{c: "{:+.1f}%" for c in diff_cols}
    }, na_rep="—")

    st.dataframe(styled, use_container_width=True, height=420)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Header
    today = date.today().strftime("%A, %B %d %Y")
    st.markdown(f"""
    <div class="oe-header">
      <div>
        <div class="oe-title">📊 Owner Earnings Dashboard</div>
        <div class="oe-subtitle">6 metrics · {sum(len(v) for v in SECTORS.values())} tickers · industry-adjusted WACC · auto bear detection</div>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:#4b6a8a;">{today}</div>
    </div>
    """, unsafe_allow_html=True)

    data, selected_sector, search, highlight, sort_by = sidebar()

    if not data:
        st.warning("No data yet. Click **Recalculate Now** in the sidebar to run the first calculation.")
        return

    tickers = get_filtered_tickers(selected_sector, search)
    tickers = sort_tickers(tickers, data, sort_by)

    # View toggle
    view = st.radio("View", ["Cards", "Summary Table"], horizontal=True, label_visibility="collapsed")

    st.markdown("---")

    if view == "Summary Table":
        render_summary_table(tickers, data)
    else:
        # Cards grouped by sector
        if selected_sector != "All Sectors":
            for sym in tickers:
                render_ticker(sym, data.get(sym, {"error": "No data"}))
        else:
            for sector, syms in SECTORS.items():
                filtered = [s for s in syms if s in tickers]
                if not filtered:
                    continue
                st.markdown(f'<div class="section-header">{sector}</div>', unsafe_allow_html=True)
                for sym in filtered:
                    with st.expander(f"**{sym}**  —  {data.get(sym,{}).get('current',{}).get('price','—')}  |  OE Multiple: {data.get(sym,{}).get('current',{}).get('oe_multiple','—')}", expanded=False):
                        render_ticker(sym, data.get(sym, {"error": "No data"}))

    # Footer
    errors = [sym for sym in tickers if data.get(sym, {}).get("error")]
    if errors:
        with st.expander(f"⚠ {len(errors)} tickers with errors"):
            for e in errors:
                st.markdown(f'<div class="error-badge">{e}: {data[e]["error"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
