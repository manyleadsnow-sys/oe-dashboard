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
    "A","BSX","ABT","COO","HCA","IDXX","IQV","ISRG","JNJ",
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
GLOBAL_10Y_YIELD = 4.2  # Fallback

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
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP {r.status_code} for CIK {cik}")
                time.sleep(5 * (attempt + 1))
        except requests.exceptions.Timeout:
            print(f"    Timeout for CIK {cik}, attempt {attempt+1}/3")
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"    Error for CIK {cik}: {e}")
            time.sleep(5)
    return None

def find_concept_series(facts, *concepts, unit="USD"):
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        if concept not in us_gaap:
            continue
        entries = us_gaap[concept].get("units", {}).get(unit, [])
        if not entries:
            continue
        valid = [e for e in entries if "end" in e and "val" in e]
        valid.sort(key=lambda x: (x["end"], x.get("filed", "")), reverse=True)
        return valid
    return []

def dedup_by_end(entries, form_types=("10-Q", "10-K")):
    filtered = [e for e in entries if e.get("form", "") in form_types]
    seen = {}
    for e in filtered:
        key = e["end"]
        if key not in seen or e.get("filed", "") > seen[key].get("filed", ""):
            seen[key] = e
    return sorted(seen.values(), key=lambda x: x["end"], reverse=True)

def quarterly_ttm(facts, *concepts):
    entries = find_concept_series(facts, *concepts)
    valid_quarters = []
    for e in entries:
        if "start" in e and "end" in e:
            try:
                start_date = datetime.strptime(e["start"], "%Y-%m-%d")
                end_date = datetime.strptime(e["end"], "%Y-%m-%d")
                days = (end_date - start_date).days
                if 75 <= days <= 110:
                    valid_quarters.append(e)
            except ValueError:
                continue
    quarterly = dedup_by_end(valid_quarters, ("10-Q", "10-K"))
    vals = [e["val"] for e in quarterly[:4]]
    if len(vals) < 2: return None
    return sum(vals)

def annual_values(facts, *concepts, n=11):
    entries = find_concept_series(facts, *concepts)
    annual  = dedup_by_end(entries, ("10-K",))
    return [(e["end"], e["val"]) for e in annual[:n]]

def bs_values(facts, *concepts, n=6):
    entries = find_concept_series(facts, *concepts)
    deduped = dedup_by_end(entries, ("10-Q", "10-K"))
    return [(e["end"], e["val"]) for e in deduped[:n]]

def extract_edgar_financials(facts):
    ni_ttm = quarterly_ttm(facts, "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss", "NetIncomeLossAllocatedToParent")
    da_ttm = quarterly_ttm(facts, "DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation", "DepreciationAmortizationAndAccretionNet", "AmortizationOfIntangibleAssets")
    capex_ttm = quarterly_ttm(facts, "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForCapitalImprovements", "PaymentsToAcquireProductiveAssets", "PaymentsToAcquireBusinessesAndPropertyPlantAndEquipment")

    ca_s   = bs_values(facts, "AssetsCurrent")
    cl_s   = bs_values(facts, "LiabilitiesCurrent")
    csh_s  = bs_values(facts, "CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments")
    std_s  = bs_values(facts, "DebtCurrent", "ShortTermBorrowings", "LongTermDebtCurrent")

    def wc_at(idx):
        if idx >= len(ca_s) or idx >= len(cl_s): return None
        ca  = ca_s[idx][1]
        cl  = cl_s[idx][1]
        csh = csh_s[idx][1] if idx < len(csh_s) else 0
        std = std_s[idx][1] if idx < len(std_s) else 0
        return (ca - csh) - (cl - std)

    wc_curr  = wc_at(0)
    wc_prev  = wc_at(4)
    delta_wc = (wc_curr - wc_prev) if (wc_curr is not None and wc_prev is not None) else 0.0

    oe_ttm = None
    if ni_ttm is not None:
        da_val = da_ttm if da_ttm else 0
        capex_val = abs(capex_ttm) if capex_ttm else 0
        oe_ttm = ni_ttm + da_val - capex_val - delta_wc

    ni_a    = annual_values(facts, "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLossAllocatedToParent")
    da_a    = annual_values(facts, "DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation")
    capex_a = annual_values(facts, "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForCapitalImprovements", "PaymentsToAcquireProductiveAssets")

    ni_d    = {e[0][:4]: e[1] for e in ni_a}
    da_d    = {e[0][:4]: e[1] for e in da_a}
    capex_d = {e[0][:4]: e[1] for e in capex_a}
    years   = sorted(set(ni_d.keys())) 
    
    oe_annual = {}
    for yr in years:
        cx_val = abs(capex_d[yr]) if yr in capex_d else 0
        da_val = da_d[yr] if yr in da_d else 0
        oe_annual[int(yr)] = ni_d[yr] + da_val - cx_val

    ebit_a   = annual_values(facts, "OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest")
    pretax_a = annual_values(facts, "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic")
    tax_a    = annual_values(facts, "IncomeTaxExpenseBenefit", "CurrentIncomeTaxExpenseBenefit")

    avg_ebit = float(np.mean([v for _, v in ebit_a[:5]])) if ebit_a else None
    tax_rate = 0.21
    if pretax_a and tax_a:
        pt_d = {e[0]: e[1] for e in pretax_a[:4]}
        tx_d = {e[0]: e[1] for e in tax_a[:4]}
        rates = []
        for dt in pt_d:
            if dt in tx_d and pt_d[dt] != 0:
                r = tx_d[dt] / pt_d[dt]
                if 0 < r < 0.60: rates.append(r)
        if rates: tax_rate = float(np.mean(rates))

    return {
        "oe_ttm":    oe_ttm,
        "oe_annual": oe_annual,
        "avg_ebit":  avg_ebit,
        "tax_rate":  tax_rate,
        "fetched_at": datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════════════
# METRICS LAYER
# ══════════════════════════════════════════════════════════════════════════════

def compute_oe_growth_10yr(oe_annual: dict):
    if len(oe_annual) < 2: return None
    years  = sorted(oe_annual.keys())
    oldest = oe_annual[years[0]]
    newest = oe_annual[years[-1]]
    n      = years[-1] - years[0]
    if n <= 0 or oldest <= 0 or newest <= 0: return None
    return (newest / oldest) ** (1.0 / n) - 1

def compute_epv_per_share(avg_ebit, tax_rate, wacc, shares):
    if avg_ebit is None or shares is None or shares <= 0: return None
    return avg_ebit * (1 - tax_rate) / wacc / shares

CRISIS_PERIODS = [
    {"name": "April 2025 Crash", "start": "2025-03-01", "end": "2025-05-31"},
    {"name": "2022 Bear Market (Jan 2022 - Oct 2022)", "start": "2022-01-01", "end": "2022-12-31"},
    {"name": "2020 COVID-19 Crash & Bear Market (Feb 2020 - April 2020)", "start": "2020-02-01", "end": "2020-04-30"},
    {"name": "2018 Crypto/Rate Selloff", "start": "2018-01-01", "end": "2019-01-31"},
    {"name": "2015-2016 Selloff", "start": "2015-01-01", "end": "2016-06-30"}
]

def detect_macro_crises(price_series):
    crises = []
    if price_series.empty: return crises
    
    for c in CRISIS_PERIODS:
        mask = (price_series.index >= pd.Timestamp(c["start"])) & (price_series.index <= pd.Timestamp(c["end"]))
        window = price_series[mask]
        
        if len(window) < 2: continue
        
        peak_date = window.idxmax()
        peak_price = window.max()
        
        post_peak_window = window[window.index >= peak_date]
        if post_peak_window.empty: continue
        
        trough_date = post_peak_window.idxmin()
        trough_price = post_peak_window.min()
        
        drawdown = (trough_price - peak_price) / peak_price * 100
        
        if drawdown < -10:
            crises.append({
                "crisis_name": c["name"],
                "peak_date": peak_date,
                "trough_date": trough_date,
                "peak_price": float(peak_price),
                "trough_price": float(trough_price),
                "drawdown_pct": float(drawdown)
            })
            
    return crises

def pct_diff(a, b):
    if a is None or b is None or b == 0: return None
    return round((a - b) / abs(b) * 100, 2)

def metrics_at_price(oe_ttm, ev, mc, oe_growth, epv_per_share, shares):
    oe_yield    = (oe_ttm / mc * 100)               if (mc and mc != 0 and oe_ttm)             else None
    oe_multiple = (ev / oe_ttm)                      if (oe_ttm and oe_ttm != 0 and ev)         else None
    oe_peg      = (oe_multiple / (oe_growth * 100))  if (oe_multiple and oe_growth and oe_growth != 0) else None
    oeps        = (oe_ttm / shares)                  if (shares and shares != 0 and oe_ttm)     else None
    return {
        "oe":          round(oe_ttm / 1e9, 4)      if oe_ttm        is not None else None,
        "oeps":        round(oeps, 2)              if oeps          is not None else None,
        "oe_yield":    round(oe_yield, 4)           if oe_yield      is not None else None,
        "oe_multiple": round(oe_multiple, 4)        if oe_multiple   is not None else None,
        "oe_growth":   round(oe_growth * 100, 4)    if oe_growth     is not None else None,
        "oe_peg":      round(oe_peg, 4)             if oe_peg        is not None else None,
        "epv":         round(epv_per_share, 4)      if epv_per_share is not None else None,
    }

def diff_block(curr, peak):
    return {k: pct_diff(curr.get(k), peak.get(k)) for k in ("oe","oeps","oe_yield","oe_multiple","oe_growth","oe_peg","epv")}

# ══════════════════════════════════════════════════════════════════════════════
# CACHE AND INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

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
            print(f"OK (OE TTM: ${financials['oe_ttm']/1e9:.2f}B)" if financials['oe_ttm'] else "OK (OE: n/a)")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(sym)

        time.sleep(0.15) 

    save_edgar_cache(cache)
    print(f"\nEDGAR refresh complete: {ok} ok, {len(failed)} failed")
    return cache

# ══════════════════════════════════════════════════════════════════════════════
# PRICE + METRICS CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ticker_result(symbol, financials, yf_info, hist):
    result = {
        "ticker":             symbol,
        "company_name":       yf_info.get("shortName", symbol),
        "error":              None,
        "sector":             yf_info.get("sector", "default"),
        "data_source":        "SEC EDGAR + Yahoo Finance",
        "current":            {},
        "peak_since_oct2022": {},
        "bear_markets":       [],
        "discount_metrics":   {"z_score_5y": None, "z_score_10y": None, "erp_spread": None, "premium_to_floor": None},
        "last_updated":       datetime.now().isoformat(),
        "edgar_fetched_at":   financials.get("fetched_at", ""),
        "crisis_floor_multiple": None,
        "dca_signal": "—"
    }

    if hist.empty:
        result["error"] = "No price data"
        return result

    current_price = float(hist.iloc[-1])
    hist_post     = hist[hist.index >= pd.Timestamp(PEAK_START_DATE)]

    if hist_post.empty:
        result["error"] = "No price data since Oct 2022"
        return result

    peak_price = float(hist_post.max())
    peak_date  = hist_post.idxmax()

    shares   = yf_info.get("sharesOutstanding") or yf_info.get("impliedSharesOutstanding")
    net_debt = (yf_info.get("totalDebt") or 0) - (yf_info.get("totalCash") or 0)
    sector   = yf_info.get("sector", "default")
    wacc     = get_wacc(sector)

    oe_ttm        = financials.get("oe_ttm")
    oe_annual     = financials.get("oe_annual", {})
    avg_ebit      = financials.get("avg_ebit")
    tax_rate      = financials.get("tax_rate", 0.21)

    oe_growth     = compute_oe_growth_10yr(oe_annual)
    epv_per_share = compute_epv_per_share(avg_ebit, tax_rate, wacc, shares)

    def ev_mc(price):
        if not shares: return None, None
        mc = price * shares
        return mc + net_debt, mc

    ev_c, mc_c = ev_mc(current_price)
    
    if oe_ttm and mc_c:
        m_c = metrics_at_price(oe_ttm, ev_c, mc_c, oe_growth, epv_per_share, shares)
    else:
        m_c = {}
    m_c["price"] = round(current_price, 2)
    result["current"] = m_c

    ev_p, mc_p = ev_mc(peak_price)
    if oe_ttm and mc_p:
        m_p = metrics_at_price(oe_ttm, ev_p, mc_p, oe_growth, epv_per_share, shares)
    else:
        m_p = {}
    m_p["price"] = round(peak_price, 2)
    m_p["date"]  = str(peak_date.date())
    result["peak_since_oct2022"] = m_p

    if result["current"] and result["peak_since_oct2022"]:
        result["vs_peak_diff"] = diff_block(result["current"], result["peak_since_oct2022"])

    for bear in detect_macro_crises(hist):
        bp = bear["trough_price"]
        pp = bear["peak_price"]
        ev_b, mc_b = ev_mc(bp)
        ev_p, mc_p = ev_mc(pp)
        
        if oe_ttm and mc_b:
            mb = metrics_at_price(oe_ttm, ev_b, mc_b, oe_growth, epv_per_share, shares)
        else:
            mb = {}
            
        mb["price"]        = round(bp, 2)
        mb["peak_price"]   = round(pp, 2)
        mb["crisis_name"]  = bear["crisis_name"]
        mb["peak_date"]    = str(bear["peak_date"].date()) if hasattr(bear["peak_date"], "date") else str(bear["peak_date"])
        mb["trough_date"]  = str(bear["trough_date"].date()) if hasattr(bear["trough_date"], "date") else str(bear["trough_date"])
        mb["drawdown_pct"] = round(bear["drawdown_pct"], 2)
        
        if oe_ttm and mc_p:
            mp = metrics_at_price(oe_ttm, ev_p, mc_p, oe_growth, epv_per_share, shares)
        else:
            mp = {}
            
        mb["peak_metrics"] = mp
        result["bear_markets"].append(mb)

    # ══════════════════════════════════════════════════════════════════════════════
    # DISCOUNT METRICS ENGINE
    # ══════════════════════════════════════════════════════════════════════════════
    if result["current"].get("oe_yield"):
        result["discount_metrics"]["erp_spread"] = round(result["current"]["oe_yield"] - GLOBAL_10Y_YIELD, 2)

    if oe_ttm and shares and result["current"].get("oe_multiple"):
        oeps = oe_ttm / shares
        hist_multiples = hist / oeps
        curr_m = result["current"]["oe_multiple"]
        
        # 5-Year Z-Score
        lookback_5y = hist_multiples[hist_multiples.index > (datetime.now() - timedelta(days=5*365))]
        if not lookback_5y.empty and lookback_5y.std() > 0:
            result["discount_metrics"]["z_score_5y"] = round((curr_m - lookback_5y.mean()) / lookback_5y.std(), 2)
            
        # 10-Year Z-Score
        lookback_10y = hist_multiples[hist_multiples.index > (datetime.now() - timedelta(days=10*365))]
        if not lookback_10y.empty and lookback_10y.std() > 0:
            result["discount_metrics"]["z_score_10y"] = round((curr_m - lookback_10y.mean()) / lookback_10y.std(), 2)

    historical_trough_multiples = [b.get("oe_multiple") for b in result["bear_markets"] if b.get("oe_multiple") is not None]
    if historical_trough_multiples and result["current"].get("oe_multiple"):
        min_floor = min(historical_trough_multiples)
        result["crisis_floor_multiple"] = round(min_floor, 2)
        
        if min_floor > 0:
            premium = (result["current"]["oe_multiple"] / min_floor - 1) * 100
            result["discount_metrics"]["premium_to_floor"] = round(premium, 1)

    # DCA SIGNAL LOGIC 
    z5 = result["discount_metrics"].get("z_score_5y")
    prem = result["discount_metrics"].get("premium_to_floor")
    
    if (z5 is not None and z5 <= -1.5) or (prem is not None and prem <= 10):
        result["dca_signal"] = "🟢 STRONG DCA"
    elif (z5 is not None and z5 <= -0.5) or (prem is not None and prem <= 25):
        result["dca_signal"] = "🟡 ACCUMULATE"
    else:
        result["dca_signal"] = "🔴 WAIT" 

    return result

def run_prices_only(tickers, edgar_cache):
    global GLOBAL_10Y_YIELD
    try:
        tnx = yf.Ticker("^TNX")
        GLOBAL_10Y_YIELD = tnx.history(period="1d")["Close"].iloc[-1]
        print(f"Current 10Y Treasury Yield: {GLOBAL_10Y_YIELD:.2f}%")
    except: 
        print(f"Failed to fetch 10Y Treasury. Using fallback: {GLOBAL_10Y_YIELD}%")

    results  = {}
    total    = len(tickers)
    no_cache = []

    for i, sym in enumerate(tickers):
        print(f"  [{i+1}/{total}] {sym}", end=" ", flush=True)
        financials = edgar_cache.get(sym)
        
        if not financials:
            no_cache.append(sym)
            results[sym] = {"ticker": sym, "error": "No EDGAR cache", "sector": None, "current": {}, "peak_since_oct2022": {}, "bear_markets": [], "last_updated": datetime.now().isoformat()}
            print("(no cache)")
            continue

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                time.sleep(0.5 + attempt) 
                
                t = yf.Ticker(sym)
                info = t.info or {}
                hist = t.history(period="max", interval="1d")["Close"].dropna()
                
                if hist.empty:
                     raise ValueError("Empty price history")
                     
                hist.index = hist.index.tz_convert(None) if hist.index.tz else hist.index.tz_localize(None)

                results[sym] = compute_ticker_result(sym, financials, info, hist)
                price = results[sym].get("current", {}).get("price", "n/a")
                print(f"${price}")
                break 
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    results[sym] = {"ticker": sym, "error": f"Price fetch failed: {str(e)}", "sector": None, "current": {}, "peak_since_oct2022": {}, "bear_markets": [], "last_updated": datetime.now().isoformat()}
                    print(f"ERROR: {e}")

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
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=EDGAR_HEADERS, timeout=10)
        for k, v in r.json().items():
            DYNAMIC_CIK_MAP[v['ticker']] = str(v['cik_str']).zfill(10)
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