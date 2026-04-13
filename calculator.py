import os
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import requests
import warnings
warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
EDGAR_HEADERS = {
    "User-Agent": "YourName your@email.com",
    "Accept-Encoding": "gzip, deflate",
}

# Added SBC concepts to the priority list to ensure strict OE compliance
SBC_CONCEPTS = [
    "ShareBasedCompensation", 
    "StockBasedCompensation", 
    "AllocatedShareBasedCompensationExpense"
]

# ── Refined EDGAR Data Layer ──────────────────────────────────────────────────

def extract_edgar_financials(facts):
    """
    Refined extraction to explicitly handle SBC and Maintenance CapEx 
    as per the "Hedge Fund" level precision requirements.
    """
    # 1. Net Income
    ni_ttm = get_ttm(facts, "NetIncomeLoss", "ProfitLoss")
    
    # 2. D&A (Cash Flow Statement based)
    da_ttm = get_ttm(facts, "DepreciationDepletionAndAmortization", "DepreciationAndAmortization")
    
    # 3. SBC (Critical Fix: Subtracting non-cash compensation)
    sbc_ttm = get_ttm(facts, *SBC_CONCEPTS) or 0
    
    # 4. CapEx (Maintenance Proxy)
    # Using PaymentsToAcquirePropertyPlantAndEquipment as the primary proxy
    capex_ttm = get_ttm(facts, "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForCapitalImprovements")

    # Owner Earnings Formula: OE = NI + D&A - SBC - CapEx
    # This aligns with your specific correction to include SBC as a cash outflow
    oe_ttm = None
    if ni_ttm is not None:
        oe_ttm = ni_ttm + (da_ttm or 0) - sbc_ttm - abs(capex_ttm or 0)

    # Historical Series for Z-Scores
    ni_a = annual_values(facts, "NetIncomeLoss")
    da_a = annual_values(facts, "DepreciationDepletionAndAmortization")
    sbc_a = annual_values(facts, *SBC_CONCEPTS)
    capex_a = annual_values(facts, "PaymentsToAcquirePropertyPlantAndEquipment")

    ni_d = {e[0]: e[1] for e in ni_a}
    da_d = {e[0]: e[1] for e in da_a}
    sbc_d = {e[0]: e[1] for e in sbc_a}
    capex_d = {e[0]: e[1] for e in capex_a}

    oe_by_fiscal_end = {}
    for end_date in sorted(ni_d.keys()):
        oe_by_fiscal_end[end_date] = (
            ni_d[end_date] 
            + da_d.get(end_date, 0) 
            - sbc_d.get(end_date, 0) 
            - abs(capex_d.get(end_date, 0))
        )

    # Shares: Use Weighted Average for period-matching accuracy
    shares_a = annual_values(facts, "WeightedAverageNumberOfSharesOutstandingBasic", unit="shares")
    shares_ttm = shares_a[0][1] if shares_a else None

    return {
        "oe_ttm": oe_ttm,
        "oe_by_fiscal_end": oe_by_fiscal_end,
        "shares_ttm": shares_ttm,
        "fetched_at": datetime.now().isoformat(),
    }

# ── Valuation Metrics Logic ───────────────────────────────────────────────────

def metrics_at_price(oe, mc, cagr_decimal, shares):
    """
    Standardized Valuation Block.
    Ensures that 1/Multiple == Yield and calculates OE-PEG.
    """
    if oe is None or oe <= 0 or not mc:
        return {"oe_negative": True}

    oe_yield = (oe / mc) * 100
    oe_multiple = mc / oe
    
    # OE-PEG uses your Trailing OE CAGR
    oe_peg = None
    if cagr_decimal and cagr_decimal > 0:
        growth_pct = cagr_decimal * 100
        oe_peg = oe_multiple / growth_pct

    oeps = oe / shares if shares else None

    return {
        "oe_bn": round(oe / 1e9, 4),
        "oeps": round(oeps, 2) if oeps else None,
        "oe_yield": round(oe_yield, 4),
        "oe_multiple": round(oe_multiple, 4),
        "oe_growth": round(cagr_decimal * 100, 2) if cagr_decimal else None,
        "oe_peg": round(oe_peg, 4) if oe_peg else None,
        "oe_negative": False
    }

# ── Helper for Z-Score Calculation ───────────────────────────────────────────

def calculate_oe_z_score(current_oe, history_dict, window_years):
    """
    Statistical Discount Block: 
    Validates current OE against the historical distribution.
    """
    if not history_dict or current_oe is None:
        return None
    
    cutoff = (datetime.now() - timedelta(days=window_years * 365)).strftime("%Y-%m-%d")
    relevant_vals = [v for d, v in history_dict.items() if d >= cutoff]
    
    if len(relevant_vals) < 3:
        return None
        
    arr = np.array(relevant_vals)
    return round((current_oe - arr.mean()) / arr.std(), 2)