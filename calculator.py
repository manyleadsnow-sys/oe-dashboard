"""
Owner Earnings Dashboard - Core Calculation Engine
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import requests
import warnings
warnings.filterwarnings("ignore")

PEAK_START_DATE = "2022-10-12"

EDGAR_HEADERS   = {
    "User-Agent": "Gustavo Gonzalez gusqweenglish@gmail.com", 
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_CACHE     = "edgar_cache.json"
OE_DATA         = "oe_data.json"

TICKERS = [
    "AAPL","GOOG","META","MSFT","NVDA","PLTR","TSLA",
    "EA","NFLX","TMUS",
    "AMZN","CMG","CPRT","GRMN","LEN","MCD","ORLY","POOL","ROST","TSCO","ULTA",
    "BG","COST","HSY","KO","PEP","PG","PM","STZ","SYY","WMT",
    "BKR","CVX","EOG","EPD","EXE","FANG","SLB","TPL","VLO","XOM",
    "BRK-B","ACGL","AIZ","AJG","AON","ARES","AXP","BAC","BLK","BRO",
    "C","CB","CBOE","CBRE","CINF","CME","CPAY","EG","ERIE","FICO",
    "GS","IBKR","ICE","JPM","KKR","MA","MCO","MSCI","NDAQ","PGR",
    "RJF","SPGI","TRV","V","VRSK","WFC","WRB",
    "A","BSX","CI","ABT","COO","HCA","IDXX","IQV","ISRG","JNJ",
    "LLY","MCK","MRK","MTD","REGN","RMD","SYK","TECH","VRTX","WAT","WST","ZTS",
    "WM","MO","ADP","AXON","CAT","CTAS","DE","EME","EMR","ETN",
    "FAST","FIX","GD","GE","GWW","HON","HWM","LMT","NOC","ODFL",
    "OTIS","PH","PWR","ROK","ROL","ROP","TDG","TT",
    "ACN","ADI","ADSK","AMAT","AMD","ANET","APH","CDNS","CSCO","FTNT",
    "IT","KLAC","LRCX","MCHP","MPWR","MSI","NXPI","ON","PTC","Q",
    "SNPS","TEL","TER","TTD","TXN","TYL","VRSN","WDAY",
    "APD","AVY","CRH","ECL","FSLR","LIN","MLM","NUE","SHW","STLD","VMC",
    "AMT","CSGP","EXR","PSA","SBAC","VICI",
    "AEP","AWK","CEG","D","DUK","ETR","NEE","NRG","PEG","SO","SRE","VST","XEL",
]

INDUSTRY_WACC = {
    "Technology":            0.090,
    "Communication Services":0.085,
    "Consumer Cyclical":     0.085,
    "Consumer Defensive":    0.075,
    "Energy":                0.095,
    "Financial Services":    0.090,
    "Healthcare":            0.085,
    "Industrials":           0.085,
    "Real Estate":           0.075,
    "Basic Materials":       0.090,
    "Utilities":             0.065,
    "default":               0.085,
}

HARDCODED_CIKS = {}
DYNAMIC_CIK_MAP = {}

# ── Bug 5 fix: track whether live yield was fetched successfully ──────────────
GLOBAL_10Y_YIELD      = 4.2   # fallback
GLOBAL_10Y_YIELD_LIVE = False  # flag so the UI can warn when using fallback
# ─────────────────────────────────────────────────────────────────────────────

def get_wacc(sector):
    return INDUSTRY_WACC.get(sector, INDUSTRY_WACC["default"])

def get_cik(ticker):
    t = ticker.upper().replace(".", "-")
    return DYNAMIC_CIK_MAP.get(t) or HARDCODED_CIKS.get(t)

# ══════════════════════════════════════════════════════════════════════════════
# EDGAR DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_edgar_facts(cik):
    url = EDGAR_FACTS_URL.format(cik=cik)
    for attempt in range(3):
        try:
            r = requests.get(url, headers=EDGAR_HEADERS, timeout=90)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(30 * (attempt + 1))
            else:
                time.sleep(5 * (attempt + 1))
        except:
            time.sleep(5)
    return None

def find_concept_series(facts, *concepts, unit="USD"):
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    all_valid = []
    for concept in concepts:
        if concept not in us_gaap: continue
        entries = us_gaap[concept].get("units", {}).get(unit, [])
        valid = [e for e in entries if "end" in e and "val" in e]
        all_valid.extend(valid)
    all_valid.sort(key=lambda x: (x["end"], x.get("filed", "")), reverse=True)
    return all_valid

def dedup_by_end(entries, form_types=("10-Q", "10-K")):
    filtered = [e for e in entries if e.get("form", "") in form_types]
    seen = {}
    for e in filtered:
        key = e["end"]
        if key not in seen or e.get("filed", "") > seen[key].get("filed", ""):
            seen[key] = e
    return sorted(seen.values(), key=lambda x: x["end"], reverse=True)

def get_ttm(facts, *concepts):
    entries = find_concept_series(facts, *concepts)
    if not entries: return None
    k_entries = [e for e in entries if e.get("form") == "10-K" and "start" in e and "end" in e]
    if not k_entries: return None
    k_entries.sort(key=lambda x: x["end"], reverse=True)
    latest_k = k_entries[0]

    q_entries = [e for e in entries if e.get("form") == "10-Q" and "start" in e and "end" in e and e["end"] > latest_k["end"]]
    if not q_entries: return latest_k["val"]

    q_entries.sort(key=lambda x: x["end"], reverse=True)
    latest_q_end = q_entries[0]["end"]
    latest_q_ytd = max(
        [e for e in q_entries if e["end"] == latest_q_end],
        key=lambda x: (datetime.strptime(x["end"], "%Y-%m-%d") - datetime.strptime(x["start"], "%Y-%m-%d")).days
    )

    expected_prior_end   = datetime.strptime(latest_q_end, "%Y-%m-%d") - timedelta(days=365)
    expected_prior_start = datetime.strptime(latest_q_ytd["start"], "%Y-%m-%d") - timedelta(days=365)

    prior_ytd_val = 0
    for e in entries:
        if e.get("form") == "10-Q" and "start" in e and "end" in e:
            ed = datetime.strptime(e["end"],   "%Y-%m-%d")
            sd = datetime.strptime(e["start"], "%Y-%m-%d")
            if abs((ed - expected_prior_end).days) <= 25 and abs((sd - expected_prior_start).days) <= 25:
                prior_ytd_val = e["val"]
                break

    return latest_k["val"] + latest_q_ytd["val"] - prior_ytd_val

def annual_values(facts, *concepts, n=15):
    entries = find_concept_series(facts, *concepts)
    annual_entries = []
    for e in entries:
        if e.get("form") == "10-K" and "start" in e and "end" in e:
            try:
                sd = datetime.strptime(e["start"], "%Y-%m-%d")
                ed = datetime.strptime(e["end"],   "%Y-%m-%d")
                if 350 <= (ed - sd).days <= 380:
                    annual_entries.append(e)
            except:
                pass
    annual = dedup_by_end(annual_entries, ("10-K",))
    return [(e["end"], e["val"]) for e in annual[:n]]

# ── Bug 4 fix: also pull historical share counts from EDGAR ───────────────────
def annual_shares(facts, n=15):
    """Return {fiscal_year_str: share_count} from EDGAR for historical OEPS."""
    entries = find_concept_series(
        facts,
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
        unit="shares",
    )
    annual_entries = []
    for e in entries:
        if e.get("form") in ("10-K", "10-Q") and "end" in e:
            try:
                annual_entries.append(e)
            except:
                pass
    # dedup by year: keep the filing closest to fiscal year-end
    by_year = {}
    for e in annual_entries:
        yr = e["end"][:4]
        if yr not in by_year or e.get("filed", "") > by_year[yr].get("filed", ""):
            by_year[yr] = e
    return {yr: e["val"] for yr, e in by_year.items()}
# ─────────────────────────────────────────────────────────────────────────────

def extract_edgar_financials(facts):
    ni_ttm    = get_ttm(facts, "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
                        "ProfitLoss", "NetIncomeLossAllocatedToParent")
    da_ttm    = get_ttm(facts, "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                        "Depreciation", "DepreciationAmortizationAndAccretionNet",
                        "AmortizationOfIntangibleAssets")
    capex_ttm = get_ttm(facts, "PaymentsToAcquirePropertyPlantAndEquipment",
                        "PaymentsForCapitalImprovements", "PaymentsToAcquireProductiveAssets",
                        "PaymentsToAcquireBusinessesAndPropertyPlantAndEquipment")

    oe_ttm = None
    if ni_ttm is not None:
        da_val    = da_ttm    if da_ttm    else 0
        capex_val = abs(capex_ttm) if capex_ttm else 0
        oe_ttm    = ni_ttm + da_val - capex_val

    ni_a    = annual_values(facts, "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
                            "ProfitLoss", "NetIncomeLossAllocatedToParent")
    da_a    = annual_values(facts, "DepreciationDepletionAndAmortization",
                            "DepreciationAndAmortization", "Depreciation")
    capex_a = annual_values(facts, "PaymentsToAcquirePropertyPlantAndEquipment",
                            "PaymentsForCapitalImprovements", "PaymentsToAcquireProductiveAssets")

    ni_d    = {e[0][:4]: e[1] for e in ni_a}
    da_d    = {e[0][:4]: e[1] for e in da_a}
    capex_d = {e[0][:4]: e[1] for e in capex_a}
    years   = sorted(set(ni_d.keys()))

    oe_annual = {}
    for yr in years:
        cx_val = abs(capex_d[yr]) if yr in capex_d else 0
        da_val = da_d[yr]         if yr in da_d    else 0
        oe_annual[str(yr)] = ni_d[yr] + da_val - cx_val

    ebit_a  = annual_values(facts, "OperatingIncomeLoss",
                            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest")
    avg_ebit = float(np.mean([v for _, v in ebit_a[:5]])) if ebit_a else None

    # Bug 4 fix: store historical share counts
    shares_annual = annual_shares(facts)

    return {
        "oe_ttm":        oe_ttm,
        "oe_annual":     oe_annual,
        "shares_annual": shares_annual,   # ← new
        "avg_ebit":      avg_ebit,
        "tax_rate":      0.21,
        "fetched_at":    datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════════════
# METRICS LAYER
# ══════════════════════════════════════════════════════════════════════════════

def compute_oe_growth_10yr(oe_annual: dict):
    if len(oe_annual) < 2: return None
    years  = sorted(oe_annual.keys())
    oldest = oe_annual[years[0]]
    newest = oe_annual[years[-1]]
    n      = int(years[-1]) - int(years[0])
    if n <= 0 or oldest <= 0 or newest <= 0: return None
    return (newest / oldest) ** (1.0 / n) - 1

def compute_epv_per_share(avg_ebit, tax_rate, wacc, shares):
    if avg_ebit is None or shares is None or shares <= 0: return None
    return avg_ebit * (1 - tax_rate) / wacc / shares

CRISIS_PERIODS = [
    {"name": "April 2025 Crash",      "start": "2025-03-01", "end": "2025-05-31"},
    {"name": "2022 Bear Market",       "start": "2022-01-01", "end": "2022-12-31"},
    {"name": "2020 COVID-19 Crash",    "start": "2020-02-01", "end": "2020-04-30"},
    {"name": "2018 Crypto/Rate Selloff","start": "2018-01-01","end": "2019-01-31"},
    {"name": "2015-2016 Selloff",      "start": "2015-01-01", "end": "2016-06-30"},
]

def detect_macro_crises(price_series):
    crises = []
    if price_series.empty: return crises
    for c in CRISIS_PERIODS:
        mask   = (price_series.index >= pd.Timestamp(c["start"])) & \
                 (price_series.index <= pd.Timestamp(c["end"]))
        window = price_series[mask]
        if len(window) < 2: continue
        peak_date        = window.idxmax()
        peak_price       = window.max()
        post_peak_window = window[window.index >= peak_date]
        if post_peak_window.empty: continue
        trough_date  = post_peak_window.idxmin()
        trough_price = post_peak_window.min()
        drawdown     = (trough_price - peak_price) / peak_price * 100
        if drawdown < -10:
            crises.append({
                "crisis_name":  c["name"],
                "peak_date":    peak_date,
                "trough_date":  trough_date,
                "peak_price":   float(peak_price),
                "trough_price": float(trough_price),
                "drawdown_pct": float(drawdown),
            })
    return crises

def pct_diff(a, b):
    if a is None or b is None or b == 0: return None
    return round((a - b) / abs(b) * 100, 2)

def metrics_at_price(oe, ev, mc, oe_growth, epv_per_share, shares):
    oe_yield    = (oe / mc * 100)                        if (mc and mc != 0 and oe)                      else None
    oe_multiple = (ev / oe)                              if (oe and oe != 0 and ev)                      else None
    oe_peg      = (oe_multiple / (oe_growth * 100))      if (oe_multiple and oe_growth and oe_growth != 0) else None
    oeps        = (oe / shares)                          if (shares and shares != 0 and oe)              else None
    return {
        "oe":          round(oe / 1e9, 4)          if oe           is not None else None,
        "oeps":        round(oeps, 2)              if oeps         is not None else None,
        "oe_yield":    round(oe_yield, 4)          if oe_yield     is not None else None,
        "oe_multiple": round(oe_multiple, 4)       if oe_multiple  is not None else None,
        "oe_growth":   round(oe_growth * 100, 4)   if oe_growth    is not None else None,
        "oe_peg":      round(oe_peg, 4)            if oe_peg       is not None else None,
        "epv":         round(epv_per_share, 4)     if epv_per_share is not None else None,
    }

def diff_block(curr, peak):
    return {k: pct_diff(curr.get(k), peak.get(k))
            for k in ("oe", "oeps", "oe_yield", "oe_multiple", "oe_growth", "oe_peg", "epv")}

# ══════════════════════════════════════════════════════════════════════════════
# CALCULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_ticker_result(symbol, financials, yf_info, hist):
    result = {
        "ticker":           symbol,
        "company_name":     yf_info.get("shortName", symbol),
        "error":            None,
        "sector":           yf_info.get("sector", "default"),
        "data_source":      "SEC EDGAR + Yahoo Finance",
        "current":          {},
        "peak_since_oct2022": {},
        "bear_markets":     [],
        "discount_metrics": {
            "z_score_5y":       None,
            "z_score_10y":      None,
            "erp_spread":       None,
            "erp_yield_used":   None,   # Bug 5 fix: record which yield was used
            "erp_yield_live":   GLOBAL_10Y_YIELD_LIVE,
            "premium_to_floor": None,
        },
        "last_updated":      datetime.now().isoformat(),
        "edgar_fetched_at":  financials.get("fetched_at", ""),
        "crisis_floor_multiple": None,
        "dca_signal":        "—",
    }

    if hist.empty:
        result["error"] = "No price data"
        return result

    current_price = float(hist.iloc[-1])
    hist_post     = hist[hist.index >= pd.Timestamp(PEAK_START_DATE)]
    peak_price    = float(hist_post.max()) if not hist_post.empty else current_price
    peak_date     = hist_post.idxmax()    if not hist_post.empty else hist.index[-1]

    # Bug 4 fix: prefer current shares from yfinance, with EDGAR annual fallback
    shares_current = yf_info.get("sharesOutstanding") or yf_info.get("impliedSharesOutstanding")
    shares_annual  = financials.get("shares_annual", {})   # {year_str: count}

    net_debt = (yf_info.get("totalDebt") or 0) - (yf_info.get("totalCash") or 0)
    wacc     = get_wacc(result["sector"])

    oe_ttm    = financials.get("oe_ttm")
    oe_annual = financials.get("oe_annual", {})
    avg_ebit  = financials.get("avg_ebit")
    tax_rate  = financials.get("tax_rate", 0.21)

    oe_growth_current = compute_oe_growth_10yr(oe_annual)
    epv_per_share     = compute_epv_per_share(avg_ebit, tax_rate, wacc, shares_current)

    def ev_mc(price, shares):
        if not shares: return None, None
        mc = price * shares
        return mc + net_debt, mc

    # ── Bug 4 helper: pick best share count for a given date ─────────────────
    def shares_at(dt):
        """Return the share count that was in effect at date dt.
        Prefer the EDGAR annual figure whose fiscal year-end <= dt.year,
        fall back to the current yfinance figure."""
        if dt is None:
            return shares_current
        target_year = dt.year if hasattr(dt, "year") else int(str(dt)[:4])
        available = sorted([int(y) for y in shares_annual.keys() if int(y) <= target_year])
        if available:
            return shares_annual[str(available[-1])]
        return shares_current
    # ─────────────────────────────────────────────────────────────────────────

    # ── Bug 1 & 2 fix: proper fiscal-year anchoring + robust growth CAGR ─────
    def get_hist_metrics(dt):
        """Return (oe_for_period, growth_cagr) anchored to the most recent
        fiscal year whose year-end is <= dt.  Peak and trough on the same
        calendar date therefore get different OE values when they straddle a
        fiscal year boundary, and never blindly share the same figure."""
        if not dt or not oe_annual:
            return oe_ttm, oe_growth_current

        target_year = dt.year if hasattr(dt, "year") else int(str(dt)[:4])

        # Bug 1 fix: find the latest fiscal year whose year-end <= target_year
        available_years = sorted([int(y) for y in oe_annual.keys() if int(y) <= target_year])
        if not available_years:
            return oe_ttm, oe_growth_current

        end_y = available_years[-1]
        h_oe  = oe_annual.get(str(end_y))

        # Bug 2 fix: try progressively shorter spans so we always get a
        # period-specific CAGR rather than always returning oe_growth_current.
        h_growth = None
        for span in [5, 4, 3, 2]:
            start_y = end_y - span
            if str(start_y) in oe_annual:
                old = oe_annual[str(start_y)]
                new = oe_annual[str(end_y)]
                if old > 0 and new > 0:
                    h_growth = (new / old) ** (1.0 / span) - 1
                    break

        if h_growth is None:
            h_growth = oe_growth_current   # true last resort only

        return h_oe, h_growth
    # ─────────────────────────────────────────────────────────────────────────

    # Current metrics (always use current shares & TTM OE)
    ev_c, mc_c = ev_mc(current_price, shares_current)
    if oe_ttm and mc_c:
        m_c = metrics_at_price(oe_ttm, ev_c, mc_c, oe_growth_current, epv_per_share, shares_current)
    else:
        m_c = {}
    m_c["price"] = round(current_price, 2)
    result["current"] = m_c

    # Peak-since-Oct-2022 metrics — use OE & shares anchored to the peak date
    ev_p, mc_p     = ev_mc(peak_price, shares_at(peak_date))
    oe_p22, gr_p22 = get_hist_metrics(peak_date)
    epv_peak       = compute_epv_per_share(avg_ebit, tax_rate, wacc, shares_at(peak_date))
    if oe_p22 and mc_p:
        m_p = metrics_at_price(oe_p22, ev_p, mc_p, gr_p22, epv_peak, shares_at(peak_date))
    else:
        m_p = {}
    m_p["price"] = round(peak_price, 2)
    m_p["date"]  = str(peak_date.date())
    result["peak_since_oct2022"] = m_p

    if result["current"] and result["peak_since_oct2022"]:
        result["vs_peak_diff"] = diff_block(result["current"], result["peak_since_oct2022"])

    # Crisis bear-market blocks
    for bear in detect_macro_crises(hist):
        sh_b = shares_at(bear["trough_date"])
        sh_p = shares_at(bear["peak_date"])

        ev_b, mc_b = ev_mc(bear["trough_price"], sh_b)
        ev_p2, mc_p2 = ev_mc(bear["peak_price"],   sh_p)

        oe_b, gr_b = get_hist_metrics(bear["trough_date"])
        oe_p2, gr_p2 = get_hist_metrics(bear["peak_date"])

        epv_b = compute_epv_per_share(avg_ebit, tax_rate, wacc, sh_b)
        epv_p = compute_epv_per_share(avg_ebit, tax_rate, wacc, sh_p)

        if oe_b and mc_b:
            mb = metrics_at_price(oe_b, ev_b, mc_b, gr_b, epv_b, sh_b)
        else:
            mb = {}

        mb["price"]       = round(bear["trough_price"], 2)
        mb["peak_price"]  = round(bear["peak_price"], 2)
        mb["crisis_name"] = bear["crisis_name"]
        mb["peak_date"]   = str(bear["peak_date"].date())   if hasattr(bear["peak_date"],   "date") else str(bear["peak_date"])
        mb["trough_date"] = str(bear["trough_date"].date()) if hasattr(bear["trough_date"], "date") else str(bear["trough_date"])
        mb["drawdown_pct"]= round(bear["drawdown_pct"], 2)

        if oe_p2 and mc_p2:
            mp = metrics_at_price(oe_p2, ev_p2, mc_p2, gr_p2, epv_p, sh_p)
        else:
            mp = {}

        mb["peak_metrics"] = mp
        result["bear_markets"].append(mb)

    # ── Bug 5 fix: record the yield used and flag stale fallback ─────────────
    if result["current"].get("oe_yield"):
        result["discount_metrics"]["erp_spread"]     = round(result["current"]["oe_yield"] - GLOBAL_10Y_YIELD, 2)
        result["discount_metrics"]["erp_yield_used"] = round(GLOBAL_10Y_YIELD, 2)
        result["discount_metrics"]["erp_yield_live"] = GLOBAL_10Y_YIELD_LIVE
    # ─────────────────────────────────────────────────────────────────────────

    # ── Bug 1 fix (Z-score): use properly anchored OE per date, not a simple
    #    year-string lookup that shares the same value for peak and trough ────
    if oe_annual and shares_current and result["current"].get("oe_multiple"):
        historical_oeps_series = pd.Series(index=hist.index, dtype=float)
        for dt in hist.index:
            # find the most recent fiscal year-end <= dt.year
            target_y   = dt.year
            avail_yrs  = sorted([int(y) for y in oe_annual.keys() if int(y) <= target_y])
            if avail_yrs:
                best_y  = str(avail_yrs[-1])
                hist_oe = oe_annual[best_y]
                # Bug 4 fix: use historical shares for that year
                hist_sh = shares_annual.get(best_y) or shares_current
            else:
                hist_oe = oe_ttm
                hist_sh = shares_current
            historical_oeps_series[dt] = hist_oe / hist_sh if hist_sh else np.nan

        # Avoid division by zero / inf in multiples
        historical_oeps_series = historical_oeps_series.replace(0, np.nan)
        hist_multiples = hist / historical_oeps_series
        hist_multiples = hist_multiples.replace([np.inf, -np.inf], np.nan).dropna()

        curr_m = result["current"]["oe_multiple"]

        lookback_5y = hist_multiples[hist_multiples.index > (datetime.now() - timedelta(days=5*365))]
        if not lookback_5y.empty and lookback_5y.std() > 0:
            result["discount_metrics"]["z_score_5y"] = round(
                (curr_m - lookback_5y.mean()) / lookback_5y.std(), 2)

        lookback_10y = hist_multiples[hist_multiples.index > (datetime.now() - timedelta(days=10*365))]
        if not lookback_10y.empty and lookback_10y.std() > 0:
            result["discount_metrics"]["z_score_10y"] = round(
                (curr_m - lookback_10y.mean()) / lookback_10y.std(), 2)
    # ─────────────────────────────────────────────────────────────────────────

    historical_trough_multiples = [b.get("oe_multiple") for b in result["bear_markets"]
                                   if b.get("oe_multiple") is not None]
    if historical_trough_multiples and result["current"].get("oe_multiple"):
        min_floor = min(historical_trough_multiples)
        result["crisis_floor_multiple"] = round(min_floor, 2)
        if min_floor > 0:
            premium = (result["current"]["oe_multiple"] / min_floor - 1) * 100
            result["discount_metrics"]["premium_to_floor"] = round(premium, 1)

    z5   = result["discount_metrics"].get("z_score_5y")
    prem = result["discount_metrics"].get("premium_to_floor")

    if   (z5 is not None and z5 <= -1.5) or (prem is not None and prem <= 10):
        result["dca_signal"] = "🟢 STRONG DCA"
    elif (z5 is not None and z5 <= -0.5) or (prem is not None and prem <= 25):
        result["dca_signal"] = "🟡 ACCUMULATE"
    else:
        result["dca_signal"] = "🔴 WAIT"

    return result

def load_edgar_cache():
    try:
        with open(EDGAR_CACHE) as f: return json.load(f)
    except FileNotFoundError:
        return {}

def save_edgar_cache(cache):
    with open(EDGAR_CACHE, "w") as f:
        json.dump(cache, f, indent=2, default=str)
    print(f"EDGAR cache saved: {len(cache)} tickers")

def refresh_edgar_cache(tickers):
    cache = load_edgar_cache()
    total = len(tickers)
    ok, failed = 0, []
    for i, sym in enumerate(tickers):
        cik = get_cik(sym)
        if cik is None:
            failed.append(sym)
            continue
        print(f"  [{i+1}/{total}] {sym} (CIK {cik})...", end=" ", flush=True)
        facts = fetch_edgar_facts(cik)
        if facts is None:
            print("FAILED")
            failed.append(sym)
            continue
        try:
            financials = extract_edgar_financials(facts)
            cache[sym] = financials
            ok += 1
            print(f"OK (OE TTM: ${financials['oe_ttm']/1e9:.2f}B)" if financials["oe_ttm"] else "OK (OE: n/a)")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(sym)
        time.sleep(0.15)
    save_edgar_cache(cache)
    print(f"\nEDGAR refresh complete: {ok} ok, {len(failed)} failed")
    return cache

def run_prices_only(tickers, edgar_cache):
    global GLOBAL_10Y_YIELD, GLOBAL_10Y_YIELD_LIVE
    # Bug 5 fix: track success/failure of the live yield fetch
    try:
        tnx = yf.Ticker("^TNX")
        fetched = tnx.history(period="5d")["Close"].dropna()
        if fetched.empty:
            raise ValueError("Empty TNX series")
        GLOBAL_10Y_YIELD      = float(fetched.iloc[-1])
        GLOBAL_10Y_YIELD_LIVE = True
        print(f"Current 10Y Treasury Yield: {GLOBAL_10Y_YIELD:.2f}% (live)")
    except Exception as e:
        GLOBAL_10Y_YIELD_LIVE = False
        print(f"WARNING: Failed to fetch 10Y Treasury ({e}). Using fallback: {GLOBAL_10Y_YIELD:.2f}%")

    results = {}
    total   = len(tickers)
    for i, sym in enumerate(tickers):
        print(f"  [{i+1}/{total}] {sym}", end=" ", flush=True)
        financials = edgar_cache.get(sym)
        if not financials:
            results[sym] = {
                "ticker": sym, "error": "No EDGAR cache", "sector": None,
                "current": {}, "peak_since_oct2022": {}, "bear_markets": [],
                "last_updated": datetime.now().isoformat(),
            }
            print("(no cache)")
            continue

        for attempt in range(3):
            try:
                time.sleep(0.5 + attempt)
                t    = yf.Ticker(sym)
                info = t.info or {}
                hist = t.history(period="max", interval="1d")["Close"].dropna()
                if hist.empty: raise ValueError("Empty price history")
                hist.index = hist.index.tz_convert(None) if hist.index.tz else hist.index.tz_localize(None)
                results[sym] = compute_ticker_result(sym, financials, info, hist)
                price = results[sym].get("current", {}).get("price", "n/a")
                print(f"${price}")
                break
            except Exception as e:
                if attempt == 2:
                    results[sym] = {
                        "ticker": sym, "error": f"Price fetch failed: {str(e)}", "sector": None,
                        "current": {}, "peak_since_oct2022": {}, "bear_markets": [],
                        "last_updated": datetime.now().isoformat(),
                    }
                    print(f"ERROR: {e}")
                else:
                    time.sleep(2)
    return results

def run_edgar_and_prices(tickers):
    print("Phase 1: Refreshing EDGAR financial data...")
    edgar_cache = refresh_edgar_cache(tickers)
    print("\nPhase 2: Fetching prices and computing metrics...")
    return run_prices_only(tickers, edgar_cache)

def save_results(results, path=OE_DATA):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} tickers -> {path}")

if __name__ == "__main__":
    run_mode = os.environ.get("RUN_MODE", "prices_only").strip().lower()
    print(f"{'='*60}")
    print(f"OE Dashboard Calculator - Mode: {run_mode}")
    print(f"{'='*60}")

    print("Fetching official SEC CIK mapping...")
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=EDGAR_HEADERS, timeout=10)
        for k, v in r.json().items():
            DYNAMIC_CIK_MAP[v["ticker"]] = str(v["cik_str"]).zfill(10)
    except Exception as e:
        print(f"Failed to fetch SEC mapping: {e}")

    if run_mode == "edgar_and_prices":
        results = run_edgar_and_prices(TICKERS)
    else:
        cache = load_edgar_cache()
        if len(cache) == 0:
            print("No cache found — switching to full EDGAR refresh...")
            results = run_edgar_and_prices(TICKERS)
        else:
            results = run_prices_only(TICKERS, cache)

    save_results(results)
