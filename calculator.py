"""
Owner Earnings Dashboard - Core Calculation Engine
Financials: SEC EDGAR (XBRL API) — authoritative source from 10-Q/10-K filings
Prices:     Yahoo Finance (yfinance) — daily price history
"""

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
EDGAR_HEADERS   = {"User-Agent": "OEDashboard research/1.0 admin@oedashboard.com",
                    "Accept-Encoding": "gzip, deflate"}
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/{cik}.json"
TICKER_CIK_URL  = "https://www.sec.gov/files/company_tickers.json"

# Hardcoded CIK map — avoids dependency on SEC's ticker list endpoint
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
    "AEP","AWK","CEG","D","DUK","ETR","NEE","NRG","PEG","SO","SRE","VST","XEL"
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

def get_wacc(sector):
    return INDUSTRY_WACC.get(sector, INDUSTRY_WACC["default"])


# ── EDGAR CIK lookup ───────────────────────────────────────────────────────────

_LIVE_CIK_MAP = {}

def load_live_cik_map():
    """Try to load live CIK map from SEC as a supplement to hardcoded values."""
    global _LIVE_CIK_MAP
    if _LIVE_CIK_MAP:
        return
    try:
        r = requests.get(TICKER_CIK_URL, headers=EDGAR_HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            for entry in data.values():
                tk  = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                _LIVE_CIK_MAP[tk] = cik
            if "BRK-B" not in _LIVE_CIK_MAP and "BRK.B" in _LIVE_CIK_MAP:
                _LIVE_CIK_MAP["BRK-B"] = _LIVE_CIK_MAP["BRK.B"]
            print(f"  Live CIK map loaded: {len(_LIVE_CIK_MAP)} entries")
        else:
            print(f"  Live CIK map unavailable (HTTP {r.status_code}), using hardcoded values")
    except Exception as e:
        print(f"  Live CIK map unavailable ({e}), using hardcoded values")


def get_cik(ticker):
    tk = ticker.upper().replace(".", "-")
    # Hardcoded map first (reliable), then live map as supplement
    if tk in HARDCODED_CIKS:
        return HARDCODED_CIKS[tk]
    return _LIVE_CIK_MAP.get(tk)


# ── EDGAR facts fetcher ────────────────────────────────────────────────────────

def fetch_edgar_facts(cik):
    """Fetch all XBRL company facts from EDGAR with retries. Returns dict or None."""
    url = EDGAR_FACTS_URL.format(cik=cik)
    for attempt in range(4):
        try:
            r = requests.get(url, headers=EDGAR_HEADERS, timeout=90)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  EDGAR rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif r.status_code == 403:
                print(f"  EDGAR 403 for CIK {cik} — check User-Agent header")
                return None
            else:
                print(f"  EDGAR HTTP {r.status_code} for CIK {cik}")
                time.sleep(5 * (attempt + 1))
        except requests.exceptions.Timeout:
            print(f"  EDGAR timeout for CIK {cik}, attempt {attempt+1}/4")
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"  EDGAR error for CIK {cik}: {e}")
            time.sleep(5)
    return None


def get_concept(facts, *concept_names, unit="USD", form_filter=None):
    """
    Extract a time-series of values for a GAAP concept from EDGAR facts.
    Returns a list of dicts: {end, val, form, filed} sorted by end date.
    concept_names: try in order (first found wins).
    form_filter: e.g. "10-K" or "10-Q" — None means both.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concept_names:
        if concept not in us_gaap:
            continue
        units_data = us_gaap[concept].get("units", {})
        if unit not in units_data:
            continue
        entries = units_data[unit]
        result  = []
        for e in entries:
            form  = e.get("form", "")
            if form_filter and form_filter not in form:
                continue
            # Skip amended forms with different period lengths (use accn dedup)
            if "end" not in e or "val" not in e:
                continue
            result.append({
                "end":   e["end"],
                "val":   e["val"],
                "form":  form,
                "filed": e.get("filed", ""),
                "accn":  e.get("accn", ""),
                "fp":    e.get("fp", ""),
                "fy":    e.get("fy", ""),
            })
        if result:
            # Sort by end date descending
            result.sort(key=lambda x: x["end"], reverse=True)
            return result
    return []


def quarterly_series(entries, n=8):
    """
    Extract n most-recent quarterly values from EDGAR entries.
    Quarterly = 10-Q forms. Returns list of (end_date, value) sorted newest first.
    """
    quarterly = [e for e in entries if "10-Q" in e.get("form","")]
    # Deduplicate by end date (keep most recently filed)
    seen = {}
    for e in quarterly:
        key = e["end"]
        if key not in seen or e["filed"] > seen[key]["filed"]:
            seen[key] = e
    deduped = sorted(seen.values(), key=lambda x: x["end"], reverse=True)
    return [(e["end"], e["val"]) for e in deduped[:n]]


def annual_series(entries, n=11):
    """
    Extract n most-recent annual values from EDGAR 10-K entries.
    Returns list of (end_date, value) sorted newest first.
    """
    annual = [e for e in entries if e.get("form","") == "10-K"]
    seen   = {}
    for e in annual:
        key = e["end"]
        if key not in seen or e["filed"] > seen[key]["filed"]:
            seen[key] = e
    deduped = sorted(seen.values(), key=lambda x: x["end"], reverse=True)
    return [(e["end"], e["val"]) for e in deduped[:n]]


def ttm_from_quarters(series):
    """Sum last 4 quarterly values = TTM."""
    vals = [v for _, v in series[:4]]
    if len(vals) < 2:
        return None
    return sum(vals)


def ttm_from_annual_minus_quarter(annual_series, quarterly_series):
    """
    TTM = Most recent annual + most recent quarter YTD adjustment.
    Alternative when quarterly data is sparse.
    Falls back to most recent annual.
    """
    if annual_series:
        return annual_series[0][1]
    return None


# ── Balance sheet point-in-time ────────────────────────────────────────────────

def latest_bs_value(facts, *concepts):
    """Get the most recent balance sheet value for any of the given concepts."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        if concept not in us_gaap:
            continue
        entries = us_gaap[concept].get("units", {}).get("USD", [])
        if not entries:
            continue
        # Filter to annual or quarterly, sort by end date
        valid = [e for e in entries if e.get("form","") in ("10-K","10-Q")]
        if not valid:
            valid = entries
        valid.sort(key=lambda x: (x.get("end",""), x.get("filed","")), reverse=True)
        return valid[0]["val"]
    return None


def bs_series(facts, *concepts, n=8):
    """Get a time series of balance sheet values (for WC calculation)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        if concept not in us_gaap:
            continue
        entries = us_gaap[concept].get("units", {}).get("USD", [])
        valid   = [e for e in entries if e.get("form","") in ("10-K","10-Q")]
        seen    = {}
        for e in valid:
            key = e["end"]
            if key not in seen or e.get("filed","") > seen[key].get("filed",""):
                seen[key] = e
        deduped = sorted(seen.values(), key=lambda x: x["end"], reverse=True)
        return [(e["end"], e["val"]) for e in deduped[:n]]
    return []


# ── OE calculation from EDGAR ──────────────────────────────────────────────────

def compute_oe_ttm_edgar(facts):
    """
    OE (TTM) = Net Income + D&A - |CapEx| +/- Delta Working Capital
    All from EDGAR XBRL facts.
    """
    # Net Income (TTM from quarters)
    ni_entries = get_concept(facts,
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
        "NetIncome")
    ni_q  = quarterly_series(ni_entries)
    ni_ttm = ttm_from_quarters(ni_q)

    # D&A (TTM from quarters)
    da_entries = get_concept(facts,
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
        "DepreciationAmortizationAndAccretionNet",
        "AmortizationOfIntangibleAssets")
    da_q   = quarterly_series(da_entries)
    da_ttm = ttm_from_quarters(da_q)

    # CapEx (TTM from quarters) — stored as negative in cash flow
    capex_entries = get_concept(facts,
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "CapitalExpendituresIncurredButNotYetPaid",
        "PaymentsToAcquireProductiveAssets")
    capex_q   = quarterly_series(capex_entries)
    capex_ttm = ttm_from_quarters(capex_q)

    if ni_ttm is None or da_ttm is None or capex_ttm is None:
        return None

    capex_val = abs(capex_ttm)

    # Working Capital Change (most recent quarter vs ~4 quarters ago)
    ca_s   = bs_series(facts, "AssetsCurrent")
    cl_s   = bs_series(facts, "LiabilitiesCurrent")
    csh_s  = bs_series(facts,
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments",
                "Cash")
    std_s  = bs_series(facts,
                "DebtCurrent",
                "ShortTermBorrowings",
                "LongTermDebtCurrent",
                "NotesPayableCurrent")

    def wc_at(series_dict, idx):
        ca  = series_dict["ca"][idx][1]  if idx < len(series_dict["ca"])  else None
        cl  = series_dict["cl"][idx][1]  if idx < len(series_dict["cl"])  else None
        csh = series_dict["csh"][idx][1] if idx < len(series_dict["csh"]) else 0
        std = series_dict["std"][idx][1] if idx < len(series_dict["std"]) else 0
        if ca is None or cl is None:
            return None
        return (ca - csh) - (cl - std)

    sd = {"ca": ca_s, "cl": cl_s, "csh": csh_s, "std": std_s}
    wc_curr = wc_at(sd, 0)
    wc_prev = wc_at(sd, 4)
    delta_wc = (wc_curr - wc_prev) if (wc_curr is not None and wc_prev is not None) else 0.0

    return ni_ttm + da_ttm - capex_val - delta_wc


def compute_oe_annual_series_edgar(facts):
    """Annual OE series for 10-year CAGR calculation."""
    ni_entries    = get_concept(facts, "NetIncomeLoss",
                                "NetIncomeLossAvailableToCommonStockholdersBasic",
                                "ProfitLoss")
    da_entries    = get_concept(facts, "DepreciationDepletionAndAmortization",
                                "DepreciationAndAmortization", "Depreciation")
    capex_entries = get_concept(facts, "PaymentsToAcquirePropertyPlantAndEquipment",
                                "PaymentsForCapitalImprovements",
                                "PaymentsToAcquireProductiveAssets")

    ni_a    = annual_series(ni_entries,    n=11)
    da_a    = annual_series(da_entries,    n=11)
    capex_a = annual_series(capex_entries, n=11)

    # Build year-indexed dict
    def to_dict(series):
        return {e[0][:4]: e[1] for e in series}  # key by year string

    ni_d    = to_dict(ni_a)
    da_d    = to_dict(da_a)
    capex_d = to_dict(capex_a)

    years = sorted(set(ni_d) & set(da_d) & set(capex_d))
    result = {}
    for yr in years:
        oe = ni_d[yr] + da_d[yr] - abs(capex_d[yr])
        result[int(yr)] = oe

    return pd.Series(result).sort_index()


def compute_oe_growth_10yr(oe_series):
    if len(oe_series) < 2:
        return None
    years  = oe_series.index.tolist()
    oldest = oe_series.iloc[0]
    newest = oe_series.iloc[-1]
    n      = years[-1] - years[0]
    if n <= 0 or oldest <= 0 or newest <= 0:
        return None
    return (newest / oldest) ** (1.0 / n) - 1


def compute_epv_edgar(facts, sector, shares):
    """EPV = Avg Normalised EBIT x (1 - tax_rate) / WACC, per share."""
    try:
        ebit_entries = get_concept(facts,
            "OperatingIncomeLoss",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "OperatingIncome")
        ebit_a = annual_series(ebit_entries, n=5)
        if not ebit_a:
            return None
        avg_ebit = float(np.mean([v for _, v in ebit_a]))

        # Tax rate from pretax income and tax provision
        pretax_entries = get_concept(facts,
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic")
        tax_entries = get_concept(facts,
            "IncomeTaxExpenseBenefit",
            "CurrentIncomeTaxExpenseBenefit")

        pretax_a = annual_series(pretax_entries, n=4)
        tax_a    = annual_series(tax_entries,    n=4)

        tax_rate = 0.21
        if pretax_a and tax_a:
            rates = []
            pt_d  = {e[0]: e[1] for e in pretax_a}
            tx_d  = {e[0]: e[1] for e in tax_a}
            for dt in pt_d:
                if dt in tx_d and pt_d[dt] != 0:
                    r = tx_d[dt] / pt_d[dt]
                    if 0 < r < 0.60:
                        rates.append(r)
            if rates:
                tax_rate = float(np.mean(rates))

        wacc      = get_wacc(sector)
        epv_total = avg_ebit * (1 - tax_rate) / wacc
        if shares and shares > 0:
            return epv_total / shares
        return None
    except Exception:
        return None


# ── Bear market detection (price only) ────────────────────────────────────────

def detect_bear_markets(price_series, threshold=0.20, top_n=10):
    if price_series.empty:
        return []
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


# ── Metric helpers ─────────────────────────────────────────────────────────────

def pct_diff(a, b):
    if a is None or b is None or b == 0:
        return None
    return round((a - b) / abs(b) * 100, 2)


def metrics_at_price(oe_ttm, ev, mc, oe_growth, epv_per_share):
    oe_yield    = (oe_ttm / mc * 100)              if (mc and mc != 0 and oe_ttm) else None
    oe_multiple = (ev / oe_ttm)                    if (oe_ttm and oe_ttm != 0 and ev) else None
    oe_peg      = (oe_multiple / (oe_growth * 100)) if (oe_multiple and oe_growth and oe_growth != 0) else None
    return {
        "oe":          round(oe_ttm / 1e9, 4)    if oe_ttm        is not None else None,
        "oe_yield":    round(oe_yield, 4)         if oe_yield      is not None else None,
        "oe_multiple": round(oe_multiple, 4)      if oe_multiple   is not None else None,
        "oe_growth":   round(oe_growth * 100, 4)  if oe_growth     is not None else None,
        "oe_peg":      round(oe_peg, 4)           if oe_peg        is not None else None,
        "epv":         round(epv_per_share, 4)    if epv_per_share is not None else None,
    }


def diff_block(curr, peak):
    return {k: pct_diff(curr.get(k), peak.get(k))
            for k in ("oe","oe_yield","oe_multiple","oe_growth","oe_peg","epv")}


# ── Main per-ticker fetch ──────────────────────────────────────────────────────

def fetch_ticker_data(symbol):
    result = {
        "ticker":             symbol,
        "error":              None,
        "sector":             None,
        "data_source":        "SEC EDGAR + Yahoo Finance",
        "current":            {},
        "peak_since_oct2022": {},
        "bear_markets":       [],
        "last_updated":       datetime.now().isoformat(),
    }
    try:
        # ── Yahoo Finance: price + sector/shares/debt ──────────────────
        t    = yf.Ticker(symbol)
        info = t.info or {}

        sector = info.get("sector", "default")
        result["sector"] = sector
        shares   = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        net_debt = (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)

        hist = t.history(period="max", interval="1d")["Close"].dropna()
        hist.index = hist.index.tz_convert(None) if hist.index.tz is not None else hist.index.tz_localize(None)

        if hist.empty:
            result["error"] = "No price data"; return result

        current_price = float(hist.iloc[-1])
        hist_post     = hist[hist.index >= pd.Timestamp(PEAK_START_DATE)]

        if hist_post.empty:
            result["error"] = "No price data since Oct 2022"; return result

        peak_price = float(hist_post.max())
        peak_date  = hist_post.idxmax()

        # ── SEC EDGAR: financial data ──────────────────────────────────
        cik = get_cik(symbol)
        if cik is None:
            result["error"] = "CIK not found in SEC EDGAR"
            return result

        facts = fetch_edgar_facts(cik)
        if facts is None:
            result["error"] = f"Could not fetch EDGAR facts for CIK {cik}"
            return result

        time.sleep(0.12)   # EDGAR rate limit: ~10 req/sec max

        oe_ttm        = compute_oe_ttm_edgar(facts)
        oe_series     = compute_oe_annual_series_edgar(facts)
        oe_growth     = compute_oe_growth_10yr(oe_series)
        epv_per_share = compute_epv_edgar(facts, sector, shares)

        def ev_mc(price):
            if not shares: return None, None
            mc = price * shares
            return mc + net_debt, mc

        # ── Current metrics ────────────────────────────────────────────
        ev_c, mc_c = ev_mc(current_price)
        if oe_ttm and mc_c:
            m = metrics_at_price(oe_ttm, ev_c, mc_c, oe_growth, epv_per_share)
            m["price"] = round(current_price, 2)
            result["current"] = m

        # ── Peak since Oct 2022 ────────────────────────────────────────
        ev_p, mc_p = ev_mc(peak_price)
        if oe_ttm and mc_p:
            m = metrics_at_price(oe_ttm, ev_p, mc_p, oe_growth, epv_per_share)
            m["price"] = round(peak_price, 2)
            m["date"]  = str(peak_date.date())
            result["peak_since_oct2022"] = m

        if result["current"] and result["peak_since_oct2022"]:
            result["vs_peak_diff"] = diff_block(result["current"], result["peak_since_oct2022"])

        # ── Bear markets ───────────────────────────────────────────────
        for bear in detect_bear_markets(hist):
            bp = bear["trough_price"]
            ev_b, mc_b = ev_mc(bp)
            if not (oe_ttm and mc_b): continue
            mb = metrics_at_price(oe_ttm, ev_b, mc_b, oe_growth, epv_per_share)
            mb["price"]        = round(bp, 2)
            mb["peak_price"]   = round(bear["peak_price"], 2)
            mb["peak_date"]    = str(bear["peak_date"].date()) if hasattr(bear["peak_date"], "date") else str(bear["peak_date"])
            mb["trough_date"]  = str(bear["trough_date"].date()) if hasattr(bear["trough_date"], "date") else str(bear["trough_date"])
            mb["drawdown_pct"] = round(bear["drawdown_pct"], 2)
            ev_bp, mc_bp = ev_mc(bear["peak_price"])
            mp = metrics_at_price(oe_ttm, ev_bp, mc_bp, oe_growth, epv_per_share)
            mb["vs_bear_peak_diff"] = diff_block(mb, mp)
            result["bear_markets"].append(mb)

    except Exception as e:
        result["error"] = str(e)
    return result


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_full_calculation(tickers=None, progress_callback=None):
    load_live_cik_map()   # try live map as supplement to hardcoded
    if tickers is None:
        tickers = TICKERS
    results = {}
    for i, sym in enumerate(tickers):
        results[sym] = fetch_ticker_data(sym)
        if progress_callback:
            progress_callback(sym, i + 1, len(tickers))
    return results


def save_results(results, path="oe_data.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} tickers -> {path}")


def load_results(path="oe_data.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


if __name__ == "__main__":
    print("Testing SEC EDGAR connectivity...")
    try:
        test = requests.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
                            headers=EDGAR_HEADERS, timeout=60)
        print(f"  AAPL facts: HTTP {test.status_code} ({len(test.content)} bytes)")
    except Exception as e:
        print(f"  Connectivity test FAILED: {e}")

    print(f"Running OE calculations for {len(TICKERS)} tickers...")
    results = run_full_calculation(
        progress_callback=lambda sym, i, n: print(f"  [{i}/{n}] {sym}")
    )
    save_results(results)
    errors = [s for s, d in results.items() if d.get("error")]
    print(f"\nDone. Errors: {errors if errors else 'none'}")
