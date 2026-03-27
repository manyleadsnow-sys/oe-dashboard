"""
Owner Earnings Dashboard — Streamlit App
"""
import streamlit as st
import json
import os
import subprocess
import pandas as pd
from datetime import date

st.set_page_config(page_title="OE Dashboard", page_icon="📊", layout="wide")

DATA_FILE = os.path.join(os.path.dirname(__file__), "oe_data.json")

def load_data():
    if not os.path.exists(DATA_FILE): return {}
    with open(DATA_FILE) as f: return json.load(f)

def render_summary_table(tickers, data):
    rows = []
    for sym in tickers:
        d = data.get(sym, {})
        curr = d.get("current", {})
        
        row = {
            "Ticker":       sym,
            "Price":        curr.get("price"),
            "OE Yield %":   curr.get("oe_yield"),
            "OE Multiple":  curr.get("oe_multiple"),
            "OE Growth %":  curr.get("oe_growth"),
            "OE-PEG":       curr.get("oe_peg"),
            "Crisis Floor": d.get("crisis_floor_multiple"),
            "DCA Signal":   d.get("dca_signal", "—")
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    styled = df.style.format({
        "Price": "${:.2f}", 
        "OE Yield %": "{:.2f}%",
        "OE Multiple": "{:.2f}x", 
        "OE Growth %": "{:.2f}%",
        "OE-PEG": "{:.2f}x",
        "Crisis Floor": "{:.2f}x"
    }, na_rep="—")

    st.dataframe(styled, use_container_width=True, height=600)

def main():
    st.title("📊 Owner Earnings Dashboard - DCA View")
    data = load_data()
    
    if not data:
        st.warning("No data. Run `python calculator.py` first.")
        return

    tickers = list(data.keys())
    render_summary_table(tickers, data)

if __name__ == "__main__":
    main()