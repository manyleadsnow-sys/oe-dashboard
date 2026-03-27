"""
Owner Earnings Dashboard - Core Calculation Engine
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time
import requests
import warnings
warnings.filterwarnings("ignore")

PEAK_START_DATE = "2022-10-12"

# CRITICAL: Replace with your actual name and email if this is not yours
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

HARDCODED_CIKS = {
    "AAPL":"0000320193","GOOG":"0001652044","META":"0001326801","MSFT":"0000789019",
    "NVDA":"0001045810","PLTR":"0001321655","TSLA":"0001318605","EA":"0000712515",
    "NFLX":"0001065280","TMUS":"0001283699","AMZN":"0001018724","CMG":"0001058090",
    "CPRT":"0000723254","GRMN":"0001121788","LEN":"0000060667","MCD":"0000063754",
    "ORLY":"0000898173","POOL":"0000945841","ROST":"0000745732","TSCO":"0000916365",
    "ULTA":"0001403568","BG":"0001144519","COST":"0000909832","HSY":"0000047111",
    "KO":"0000021344","PEP":"0000077476","PG":"0000080424","PM":"0001413159",
    "STZ":"0000016160","SYY":"0000086312","WMT":"0000104169","BKR":"0001701605",
    "CVX":"0000093410","EOG":"0000821189","EPD":"0001061219","EXE":"0001168165",
    "FANG":"0001539838","SLB":"0000087347","TPL":"0000097476","VLO":"0001035002",
    "XOM":"0000034088","BRK-B":"0001067983","ACGL":"0000947484","AIZ":"0001267238",
    "AJG":"0000354190","AON":"0000315293","ARES":"0001555280","AXP":"0000004962",
    "BAC":"0000070858","BLK":"0001364742","BRO":"0000014846","C":"0000831001",
    "CB":"0000896159","CBOE":"0001374310","CBRE":"0001138118","CINF":"0000020286",
    "CME":"0001156375","CPAY":"0001175922","EG":"0000049697","ERIE":"0000049697",
    "FICO":"0000814547","GS":"0000886982","IBKR":"0001381197","ICE":"0001571123",
    "JPM":"0000019617","KKR":"0001404912","MA":"0001141391","MCO":"0001059556",
    "MSCI":"0001408198","NDAQ":"0001120193","PGR":"0000080661","RJF":"0000720005",
    "SPGI":"0000064040","TRV":"0000086312","V":"0001403161","VRSK":"0001442145",
    "WFC":"0000072971","WRB":"0000011544","A":"0001090872","BSX":"0000885725",
    "CI":"0001739940","ABT":"0000001800","COO":"0000723254","HCA":"0000860730",
    "IDXX":"0000874716","IQV":"0001478454","ISRG":"0001035267","JNJ":"0000200406",
    "LLY":"0000059478","MCK":"0000927653","MRK":"0000310158","MTD":"0001037586",
    "REGN":"0000872589","RMD":"0000943819","SYK":"0000310764","TECH":"0000849547",
    "VRTX":"0000875320","WAT":"0001000230","WST":"0000105770","ZTS":"0001555280",
    "WM":"0000823768","MO":"0000764038","ADP":"0000012927","AXON":"0001069183",
    "CAT":"0000018230","CTAS":"0000723254","DE":"0000315189","EME":"0000093859",
    "EMR":"0000032604","ETN":"0001551182","FAST":"0000815556","FIX":"0000766704",
    "GD":"0000040533","GE":"0000040987","GWW":"0000277135","HON":"0000773840",
    "HWM":"0000004281","LMT":"0000936468","NOC":"0001133421","ODFL":"0000878927",
    "OTIS":"0001781335","PH":"0000076334","PWR":"0001050606","ROK":"0001024795",
    "ROL":"0000085408","ROP":"0000882184","TDG":"0001260221","TT":"0001466258",
    "ACN":"0001467373","ADI":"0000006845","ADSK":"0000796343","AMAT":"0000796343",
    "AMD":"0000002488","ANET":"0001313925","APH":"0000820081","CDNS":"0000813672",
    "CSCO":"0000858877","FTNT":"0001262039","IT":"0000749251","KLAC":"0000319201",
    "LRCX":"0000707549","MCHP":"0000827054","MPWR":"0001280452","MSI":"0000068505",
    "NXPI":"0001413447","ON":"0000863894","PTC":"0000857005","Q":"0001479290",
    "SNPS":"0000883241","TEL":"0001385157","TER":"0000097210","TTD":"0001671933",
    "TXN":"0000097476","TYL":"0000860731","VRSN":"0001014473","WDAY":"0001327811",
    "APD":"0000002969","AVY":"0000008818","CRH":"0001370946","ECL":"0000031462",
    "FSLR":"0001274494","LIN":"0001707092","MLM":"0000916789","NUE":"0000073309",
    "SHW":"0000089089","STLD":"0001022671","VMC":"0001396033","AMT":"0001053507",
    "CSGP":"0001467373","EXR":"0001289490","PSA":"0001393311","SBAC":"0001034669",
    "VICI":"0001695678","AEP":"0000004904","AWK":"0001410636","CEG":"0001168165",
    "D":"0000715957","DUK":"0001326160","ETR":"0000049600","NEE":"0000753308",
    "NRG":"0001013871","PEG":"0000081033","SO":"0000092122","SRE":"0001032778",
    "VST":"0001692819","XEL":"0000072741",
}

def get_wacc(sector):
    return INDUSTRY_WACC.get(sector, INDUSTRY_WACC["default"])

def get_cik(ticker):
    return HARDCODED_CIKS.get(ticker.upper().replace(".", "-"))

# ══════════════════════════════════════════════════════════════════════════════
# EDGAR DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_edgar_facts(cik):
    """Fetch XBRL facts from EDGAR with retries."""
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
    """Sum last 4 true quarters, filtering out Year-To-Date (YTD) cumulative overlaps."""
    entries = find_concept_series(facts, *concepts)
    valid_quarters = []
    
    for e in entries:
        if "start" in e and "end" in e:
            try:
                start_date = datetime.strptime(e["start"], "%Y-%m-%d")
                end_date = datetime.strptime(e["end"], "%Y-%m-%d")
                days = (end_date - start_date).days
                if 80 <= days <= 105:
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
    ni_ttm = quarterly_ttm(facts, "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss")
    da_ttm = quarterly_ttm(facts, "DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation", "DepreciationAmortizationAndAccretionNet")
    capex_ttm = quarterly_ttm(facts, "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForCapitalImprovements", "PaymentsToAcquireProductiveAssets")

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
    if ni_ttm is not None and da_ttm is not None and capex_ttm is not None:
        oe_ttm = ni_ttm + da_ttm - abs(capex_ttm) - delta_wc

    ni_a    = annual_values(facts, "NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic")
    da_a    = annual_values(facts, "DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation")
    capex_a = annual_values(facts, "PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForCapitalImprovements", "PaymentsToAcquireProductiveAssets")

    ni_d    = {e[0][:4]: e[1] for e in ni_a}
    da_d    = {e[0][:4]: e[1] for e in da_a}
    capex_d = {e[0][:4]: e[1] for e in capex_a}
    years   = sorted(set(ni_d) & set(da_d) & set(capex_d))
    oe_annual = {int(yr): ni_d[yr] + da_d[yr] - abs(capex_d[yr]) for yr in years}

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

def detect_bear_markets(price_series, threshold=0.20, top_n=10):
    if price_series.empty: return []
    rolling_max = price_series.expanding().max()
    drawdown    = (price_series - rolling_max) / rolling_max
    bears, in_bear = [], False
    peak_date = peak_price = trough_date = trough_price = None

    for dt, dd in drawdown.items():
        price    = price_series[dt]
        roll_max = rolling_max[dt]
        if not in_bear and dd <= -threshold:
            in_bear      = True
            mask         = price_series[:dt] == roll_max
            peak_date    = mask[mask].index[-1] if mask.any() else dt
            peak_price   = roll_max
            trough_date  = dt
            trough_price = price
        elif in_bear:
            if price < trough_price:
                trough_date  = dt
                trough_price = price
            elif dd > -threshold / 2:
                bears.append({"peak_date": peak_date, "trough_date": trough_date,
                               "peak_price": float(peak_price), "trough_price": float(trough_price),
                               "drawdown_pct": float((trough_price - peak_price) / peak_price * 100)})
                in_bear = False

    if in_bear and trough_date is not None:
        bears.append({"peak_date": peak_date, "trough_date": trough_date,
                       "peak_price": float(peak_price), "trough_price": float(trough_price),
                       "drawdown_pct": float((trough_price - peak_price) / peak_price * 100)})
    bears.sort(key=lambda x: x["drawdown_pct"])
    return bears[:top_n]

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
# EDGAR CACHE LAYER
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
        m = metrics_at_price(oe_ttm, ev_c, mc_c, oe_growth, epv_per_share, shares)
        m["price"] = round(current_price, 2)
        result["current"] = m

    ev_p, mc_p = ev_mc(peak_price)
    if oe_ttm and mc_p:
        m = metrics_at_price(oe_ttm, ev_p, mc_p, oe_growth, epv_per_share, shares)
        m["price"] = round(peak_price, 2)
        m["date"]  = str(peak_date.date())
        result["peak_since_oct2022"] = m

    if result["current"] and result["peak_since_oct2022"]:
        result["vs_peak_diff"] = diff_block(result["current"], result["peak_since_oct2022"])

    # POST-2015 BEAR MARKET FILTER
    hist_since_2015 = hist[hist.index >= pd.Timestamp("2015-01-01")]

    for bear in detect_bear_markets(hist_since_2015):
        bp = bear["trough_price"]
        pp = bear["peak_price"]
        
        ev_b, mc_b = ev_mc(bp)
        ev_p, mc_p = ev_mc(pp)
        
        if not (oe_ttm and mc_b and mc_p): continue
        
        # Calculate trough metrics
        mb = metrics_at_price(oe_ttm, ev_b, mc_b, oe_growth, epv_per_share, shares)
        mb["price"]        = round(bp, 2)
        mb["peak_price"]   = round(pp, 2)
        mb["peak_date"]    = str(bear["peak_date"].date()) if hasattr(bear["peak_date"], "date") else str(bear["peak_date"])
        mb["trough_date"]  = str(bear["trough_date"].date()) if hasattr(bear["trough_date"], "date") else str(bear["trough_date"])
        mb["drawdown_pct"] = round(bear["drawdown_pct"], 2)
        
        # Calculate peak metrics immediately preceding the crash
        mp = metrics_at_price(oe_ttm, ev_p, mc_p, oe_growth, epv_per_share, shares)
        mb["peak_metrics"] = mp
        
        result["bear_markets"].append(mb)

    historical_trough_multiples = [b.get("oe_multiple") for b in result["bear_markets"] if b.get("oe_multiple") is not None]
    
    if historical_trough_multiples and result["current"].get("oe_multiple"):
        crisis_floor_multiple = min(historical_trough_multiples)
        current_multiple = result["current"]["oe_multiple"]
        
        result["crisis_floor_multiple"] = round(crisis_floor_multiple, 2)
        
        if crisis_floor_multiple > 0:
            premium_to_floor = (current_multiple - crisis_floor_multiple) / crisis_floor_multiple
            
            if premium_to_floor <= 0.15: 
                result["dca_signal"] = "🟢 STRONG DCA"
            elif premium_to_floor <= 0.30: 
                result["dca_signal"] = "🟡 ACCUMULATE"
            else:
                result["dca_signal"] = "🔴 WAIT"
        else:
            result["dca_signal"] = "🔴 WAIT" 

    return result

def run_prices_only(tickers, edgar_cache):
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

        try:
            t    = yf.Ticker(sym)
            info = t.info or {}
            hist = t.history(period="max", interval="1d")["Close"].dropna()
            hist.index = hist.index.tz_convert(None) if hist.index.tz else hist.index.tz_localize(None)

            results[sym] = compute_ticker_result(sym, financials, info, hist)
            price = results[sym].get("current", {}).get("price", "n/a")
            print(f"${price}")
        except Exception as e:
            results[sym] = {"ticker": sym, "error": str(e), "sector": None, "current": {}, "peak_since_oct2022": {}, "bear_markets": [], "last_updated": datetime.now().isoformat()}
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