"""
Owner Earnings Dashboard - Core Calculation Engine
v6: Tiingo prices · FRED 10Y yield · full EDGAR for fundamentals (no yfinance)

Metric definitions (what each block must show):
────────────────────────────────────────────────
BLOCK 1 — Statistical discount (A-C):
  A) 5Y OE earnings Z-score  (TTM OE vs distribution of annual OE levels, last 5Y)
  B) 10Y OE earnings Z-score (TTM OE vs distribution of annual OE levels, last 10Y)
  C) ERP spread: current OE yield − 10Y Treasury yield

BLOCK 2 — 52-week peak vs today (E-I):
  Peak   = highest closing price in the 52 weeks ending yesterday
  OE     = latest fiscal year-end OE whose date is ≤ peak date
  Growth = trailing CAGR anchored to that same fiscal year-end
  Today  = current price, current TTM OE, CAGR anchored to latest fiscal year-end

BLOCK 3 — Per crisis (J-N):
  Pre-crisis peak = 52-week high in the 52 weeks before crisis_start
  Trough          = lowest closing price inside the crisis window
  OE / growth for each date → latest fiscal year-end ≤ that date
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import requests
import warnings
warnings.filterwarnings("ignore")

# ── CREDENTIALS & ENDPOINTS ────────────────────────────────────────────────────
TIINGO_TOKEN   = os.environ.get("TIINGO_TOKEN", "2dfc1e2c3bf907f438a5edfaa94a4a1ee6cd0539")
TIINGO_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_TOKEN}",
}

EDGAR_HEADERS = {
    "User-Agent": "Gustavo Gonzalez gusqweenglish@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_SUBS_URL  = "https://data.sec.gov/submissions/CIK{cik}.json"
FRED_DGS10      = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"

EDGAR_CACHE          = "edgar_cache.json"
PRICE_CACHE          = "price_cache.json"
OE_DATA              = "oe_data.json"
CACHE_SCHEMA_VERSION = 7   # v7: dual price series (adjClose for dates, close for MC)
PRICE_CACHE_SCHEMA   = 2   # v2: stores {"a": adjClose, "c": close} per date

# Crisis windows — trough is the lowest price inside [start, end].
# Pre-crisis peak is defined as the 52-week high in the year BEFORE start.
# 2008 GFC added per spec ("since 2007").
CRISIS_PERIODS = [
    {"name": "April 2025 Crash",       "start": "2025-03-01", "end": "2025-06-30"},
    {"name": "2022 Bear Market",        "start": "2022-01-01", "end": "2022-10-14"},
    {"name": "2020 COVID-19 Crash",     "start": "2020-02-15", "end": "2020-03-23"},
    {"name": "2018 Rate/Trade Selloff", "start": "2018-09-20", "end": "2018-12-26"},
    {"name": "2015-2016 Selloff",       "start": "2015-06-01", "end": "2016-02-11"},
    # 2008 GFC removed: SEC EDGAR XBRL data does not exist before 2009.
    # Falling back to current-TTM OE for pre-2009 periods produces nonsensical
    # multiples and yields. No historical OE data = no crisis entry shown.
]

INDUSTRY_WACC = {
    "Technology":             0.090,
    "Communication Services": 0.085,
    "Consumer Cyclical":      0.085,
    "Consumer Defensive":     0.075,
    "Energy":                 0.095,
    "Financial Services":     0.090,
    "Healthcare":             0.085,
    "Industrials":            0.085,
    "Real Estate":            0.075,
    "Basic Materials":        0.090,
    "Utilities":              0.065,
    "default":                0.085,
}

TICKERS = [
    # ── VERIFICATION MODE: AAPL only ──────────────────────────────────────────
    # All 186 watchlist tickers commented out until AAPL metrics are corroborated.
    "AAPL",
    # "GOOG","META","MSFT","NVDA","PLTR","TSLA","EA","NFLX","TMUS","AMZN","CMG","CPRT","GRMN","LEN","MCD","ORLY","POOL","ROST","TSCO","ULTA","BG","COST","HSY","KO","PEP","PG","PM","STZ","SYY","WMT","BKR","CVX","EOG","EPD","EXE","FANG","SLB","TPL","VLO","XOM","BRK.B","ACGL","AIZ","AJG","AON","ARES","AXP","BAC","BLK","BRO","C","CB","CBOE","CBRE","CINF","CME","CPAY","EG","ERIE","FICO","GS","IBKR","ICE","JPM","KKR","MA","MCO","MSCI","NDAQ","PGR","RJF","SPGI","TRV","V","VRSK","WFC","WRB","A","BSX","CI","ABT","COO","HCA","IDXX","IQV","ISRG","JNJ","LLY","MCK","MRK","MTD","REGN","RMD","SYK","TECH","VRTX","WAT","WST","ZTS","WM","MO","ADP","AXON","CAT","CTAS","DE","EME","EMR","ETN","FAST","FIX","GD","GE","GWW","HON","HWM","LMT","NOC","ODFL","OTIS","PH","PWR","ROK","ROL","ROP","TDG","TT","ACN","ADI","ADSK","AMAT","AMD","ANET","APH","CDNS","CSCO","FTNT","IT","KLAC","LRCX","MCHP","MPWR","MSI","NXPI","ON","PTC","Q","SNPS","TEL","TER","TTD","TXN","TYL","VRSN","WDAY","APD","AVY","CRH","ECL","FSLR","LIN","MLM","NUE","SHW","STLD","VMC","AMT","CSGP","EXR","PSA","SBAC","VICI","AEP","AWK","CEG","D","DUK","ETR","NEE","NRG","PEG","SO","SRE","VST","XEL",
]

HARDCODED_CIKS  = {}
DYNAMIC_CIK_MAP = {}

GLOBAL_10Y_YIELD      = 4.34   # fallback (Apr 2026 — updated if FRED live fetch fails)
GLOBAL_10Y_YIELD_LIVE = False  # True only when live FRED fetch succeeded

# ══════════════════════════════════════════════════════════════════════════════
# SIC → SECTOR MAPPING (EDGAR SIC codes → GICS-style sector for WACC lookup)
# ══════════════════════════════════════════════════════════════════════════════

def sic_to_sector(sic) -> str:
    """Map an EDGAR SIC code (int or str) to a sector string matching INDUSTRY_WACC."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "default"
    if   100  <= s <  1000: return "Consumer Defensive"    # Agriculture
    if  1000  <= s <  1500: return "Basic Materials"       # Mining
    if  1500  <= s <  1800: return "Industrials"           # Construction
    if  2000  <= s <  2200: return "Consumer Defensive"    # Food & Tobacco
    if  2600  <= s <  2700: return "Basic Materials"       # Paper
    if  2800  <= s <  2900: return "Basic Materials"       # Chemicals
    if  2900  <= s <  3000: return "Energy"                # Petroleum Refining
    if  3000  <= s <  3570: return "Industrials"           # Manufacturing (excl. computers)
    if  3570  <= s <  3580: return "Technology"            # Computer/Office Equipment (AAPL=3571)
    if  3580  <= s <  3600: return "Industrials"           # Industrial Machinery
    if  3600  <= s <  3700: return "Technology"            # Electronic Equipment
    if  3700  <= s <  3800: return "Consumer Cyclical"     # Motor Vehicles
    if  3800  <= s <  3900: return "Healthcare"            # Instruments / Med Devices
    if  4000  <= s <  4600: return "Industrials"           # Rail / Air Transport
    if  4800  <= s <  4900: return "Communication Services"# Telephone / Cable
    if  4900  <= s <  5000: return "Utilities"             # Electric, Gas
    if  5000  <= s <  5300: return "Consumer Cyclical"     # Wholesale Trade
    if  5300  <= s <  5600: return "Consumer Defensive"    # Retail Food / Drug
    if  5600  <= s <  6000: return "Consumer Cyclical"     # Retail Other
    if  6000  <= s <  6300: return "Financial Services"    # Banking
    if  6300  <= s <  6500: return "Financial Services"    # Insurance
    if  6500  <= s <  6600: return "Real Estate"
    if  6700  <= s <  6800: return "Financial Services"    # Investment firms / Holding cos
    if  7000  <= s <  7370: return "Consumer Cyclical"     # Hotels / Leisure
    if  7370  <= s <  7380: return "Technology"            # Software / IT services
    if  7380  <= s <  7400: return "Technology"            # IT & Data Processing
    if  7400  <= s <  8000: return "Consumer Cyclical"     # Misc Business Services
    if  8000  <= s <  8100: return "Healthcare"            # Health Services
    if  8700  <= s <  8800: return "Technology"            # Engineering / Mgmt Consulting
    return "default"


# ══════════════════════════════════════════════════════════════════════════════
# PRICE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _52w_peak(price_series: pd.Series, reference_date: pd.Timestamp):
    """
    Highest closing price in the 52 calendar weeks (364 days) strictly before
    reference_date.  Returns (peak_date, peak_price) or (None, None).
    """
    end    = reference_date - timedelta(days=1)
    start  = reference_date - timedelta(days=364)
    window = price_series[(price_series.index >= start) & (price_series.index <= end)]
    if window.empty:
        return None, None
    return window.idxmax(), float(window.max())


def _trough_in_window(price_series: pd.Series, start_str: str, end_str: str):
    """
    Lowest closing price in [start_str, end_str].
    Returns (trough_date, trough_price) or (None, None).
    """
    mask   = ((price_series.index >= pd.Timestamp(start_str)) &
              (price_series.index <= pd.Timestamp(end_str)))
    window = price_series[mask]
    if window.empty:
        return None, None
    return window.idxmin(), float(window.min())


# ══════════════════════════════════════════════════════════════════════════════
# TIINGO PRICE LAYER  (replaces yfinance for all price data)
# ══════════════════════════════════════════════════════════════════════════════

def _tiingo_ticker(sym: str) -> str:
    """Convert ticker symbol to Tiingo format (BRK.B → BRK-B)."""
    return sym.upper().replace(".", "-")


def load_price_cache() -> dict:
    try:
        with open(PRICE_CACHE) as f:
            data = json.load(f)
        if data.get("__schema__", 1) < PRICE_CACHE_SCHEMA:
            print(f"INFO: price_cache.json schema v{data.get('__schema__',1)} < v{PRICE_CACHE_SCHEMA}. Rebuilding.")
            return {}
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_price_cache(cache: dict):
    cache["__schema__"] = PRICE_CACHE_SCHEMA
    with open(PRICE_CACHE, "w") as f:
        json.dump(cache, f)


def _adj_series_from_cache(prices: dict) -> pd.Series:
    """
    Build tz-naive pd.Series of split-adjusted close prices.
    New format: {date: {"a": adjClose, "c": close}}
    Old format: {date: float}  — treated as adjClose for backward compat.
    """
    if not prices:
        return pd.Series(dtype=float)
    data = {}
    for d, v in prices.items():
        if d == "__schema__":
            continue
        if isinstance(v, dict):
            val = v.get("a") or v.get("c")
        else:
            val = v  # old float format
        if val is not None:
            data[pd.Timestamp(d)] = float(val)
    return pd.Series(data).sort_index().dropna()


def _close_series_from_cache(prices: dict) -> pd.Series:
    """
    Build tz-naive pd.Series of UNADJUSTED close prices (for MC calculation).
    New format: {date: {"a": adjClose, "c": close}}
    Old format: {date: float} — no unadjusted distinction; treated as close.
    """
    if not prices:
        return pd.Series(dtype=float)
    data = {}
    for d, v in prices.items():
        if d == "__schema__":
            continue
        if isinstance(v, dict):
            val = v.get("c") or v.get("a")
        else:
            val = v  # old float format — no distinction available
        if val is not None:
            data[pd.Timestamp(d)] = float(val)
    return pd.Series(data).sort_index().dropna()


def fetch_tiingo_prices(sym: str, price_cache: dict) -> tuple:
    """
    Fetch full EOD price history from Tiingo with local caching.
    Stores BOTH adjClose (split-adjusted, for date identification) and
    close (unadjusted, for MC calculation with EDGAR point-in-time shares).

    Returns (hist_adj, hist_close) — two tz-naive pd.Series sorted ascending.

    Incremental strategy: if a cache entry exists, only fetches days after
    the last cached date.  On first run fetches from 2004-01-01.

    price_cache is mutated in-place; caller must call save_price_cache() when done.
    """
    t      = _tiingo_ticker(sym)
    entry  = price_cache.get(sym, {})
    cached = entry.get("prices", {})   # {date_str: {"a": adjClose, "c": close}}

    # Determine fetch window — skip reserved keys like __schema__
    date_keys = [k for k in cached if not k.startswith("__")]

    start_date = "2004-01-01"
    if date_keys:
        last_str = max(date_keys)
        last_dt  = datetime.strptime(last_str, "%Y-%m-%d")
        if (datetime.now() - last_dt).days < 2:
            # Cache is current — no fetch needed
            return _adj_series_from_cache(cached), _close_series_from_cache(cached)
        start_date = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    end_date = datetime.now().strftime("%Y-%m-%d")
    url      = f"https://api.tiingo.com/tiingo/daily/{t}/prices"
    params   = {
        "startDate":    start_date,
        "endDate":      end_date,
        "resampleFreq": "daily",
        "token":        TIINGO_TOKEN,
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=TIINGO_HEADERS, timeout=45)
            if r.status_code == 200:
                rows = r.json()
                for row in rows:
                    d_str = row["date"][:10]   # "2024-01-02T00:00:00+00:00" → "2024-01-02"
                    adj   = row.get("adjClose")
                    cls   = row.get("close")
                    if adj is not None or cls is not None:
                        cached[d_str] = {
                            "a": float(adj) if adj is not None else float(cls),
                            "c": float(cls) if cls is not None else float(adj),
                        }
                price_cache[sym] = {
                    "prices":     cached,
                    "fetched_at": datetime.now().isoformat(),
                }
                break
            elif r.status_code == 404:
                print(f"    ⚠  {sym} not found on Tiingo")
                break
            elif r.status_code == 429:
                time.sleep(30 * (attempt + 1))
            else:
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            time.sleep(5)

    return _adj_series_from_cache(cached), _close_series_from_cache(cached)


# ══════════════════════════════════════════════════════════════════════════════
# FRED 10Y TREASURY YIELD  (replaces yfinance ^TNX)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_10y_yield():
    """
    Fetch current 10Y Treasury yield from FRED (DGS10).
    Returns (yield_as_percent_float, is_live_bool).
    Tries two FRED endpoints; falls back to Treasury direct if both fail.
    """
    endpoints = [
        # Primary: FRED public CSV
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
        # Secondary: FRED API observations (no key needed for public series)
        "https://api.stlouisfed.org/fred/series/observations?series_id=DGS10"
        "&sort_order=desc&limit=5&file_type=json"
        "&api_key=e3e9f6e7d7e58f0c2f0f8b9b3e3b3b3b",  # public demo key — rate limited
    ]
    for url in endpoints[:1]:  # only FRED CSV — second is just fallback structure
        try:
            r = requests.get(url, timeout=20,
                             headers={"User-Agent": "Mozilla/5.0 (compatible)"})
            if r.status_code == 200:
                lines = r.text.strip().split("\n")
                # CSV: DATE,DGS10 — skip header, scan from end for last non-null row
                for line in reversed(lines[1:]):
                    parts = line.strip().split(",")
                    if len(parts) == 2 and parts[1].strip() not in ("", "."):
                        yield_val = float(parts[1].strip())
                        print(f"  FRED DGS10: {yield_val:.2f}% (live)")
                        return yield_val, True
        except Exception as e:
            print(f"  WARNING: FRED endpoint failed ({e}).")

    # Last resort: Treasury.gov yield curve API (no auth)
    try:
        today = datetime.now()
        for delta in range(0, 10):
            d = today - timedelta(days=delta)
            url = (f"https://home.treasury.gov/resource-center/data-chart-center/"
                   f"interest-rates/pages/xml?data=daily_treasury_yield_curve"
                   f"&field_tdr_date_value_month={d.strftime('%Y%m')}")
            r = requests.get(url, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0 (compatible)"})
            if r.status_code == 200 and "BC_10YEAR" in r.text:
                import re
                vals = re.findall(r"<BC_10YEAR>([\d.]+)</BC_10YEAR>", r.text)
                if vals:
                    yield_val = float(vals[-1])
                    print(f"  Treasury.gov 10Y: {yield_val:.2f}% (live)")
                    return yield_val, True
            break  # only try current month
    except Exception as e:
        print(f"  WARNING: Treasury.gov fetch also failed ({e}).")

    print(f"  WARNING: All 10Y yield sources failed. Using fallback 4.20%.")
    return 4.2, False


# ══════════════════════════════════════════════════════════════════════════════
# EDGAR DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def get_wacc(sector: str) -> float:
    return INDUSTRY_WACC.get(sector, INDUSTRY_WACC["default"])

def get_cik(ticker: str):
    t = ticker.upper().replace(".", "-")
    return DYNAMIC_CIK_MAP.get(t) or HARDCODED_CIKS.get(t)


def fetch_edgar_facts(cik: str):
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
        except Exception:
            time.sleep(5)
    return None


def find_concept_series(facts, *concepts, unit="USD"):
    """
    Return XBRL entries for the FIRST concept in the priority list that has data.
    Using the first match (not merging all) prevents cross-concept contamination:
    e.g. DepreciationDepletionAndAmortization (total) being mixed with
    AmortizationOfIntangibleAssets (a subset), inflating computed D&A.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        if concept not in us_gaap:
            continue
        entries = us_gaap[concept].get("units", {}).get(unit, [])
        valid   = [e for e in entries if "end" in e and "val" in e]
        if valid:
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


def get_ttm(facts, *concepts, unit="USD"):
    """
    TTM = latest_10K + latest_Q_YTD − prior_year_Q_YTD
    Falls back to the most recent 10-K annual value when no newer 10-Qs exist.
    Pass unit="shares" when fetching share-count concepts.
    """
    entries = find_concept_series(facts, *concepts, unit=unit)
    if not entries:
        return None

    k_entries = [e for e in entries
                 if e.get("form") == "10-K" and "start" in e and "end" in e]
    if not k_entries:
        return None
    k_entries.sort(key=lambda x: x["end"], reverse=True)
    latest_k = k_entries[0]

    q_entries = [e for e in entries
                 if e.get("form") == "10-Q" and "start" in e and "end" in e
                 and e["end"] > latest_k["end"]]
    if not q_entries:
        return latest_k["val"]

    q_entries.sort(key=lambda x: x["end"], reverse=True)
    latest_q_end = q_entries[0]["end"]
    latest_q_ytd = max(
        (e for e in q_entries if e["end"] == latest_q_end),
        key=lambda x: (datetime.strptime(x["end"],   "%Y-%m-%d")
                       - datetime.strptime(x["start"], "%Y-%m-%d")).days,
    )

    expected_prior_end   = datetime.strptime(latest_q_end, "%Y-%m-%d") - timedelta(days=365)
    expected_prior_start = datetime.strptime(latest_q_ytd["start"], "%Y-%m-%d") - timedelta(days=365)

    prior_ytd_val = 0
    for e in entries:
        if e.get("form") == "10-Q" and "start" in e and "end" in e:
            ed = datetime.strptime(e["end"],   "%Y-%m-%d")
            sd = datetime.strptime(e["start"], "%Y-%m-%d")
            if (abs((ed - expected_prior_end).days)   <= 25 and
                    abs((sd - expected_prior_start).days) <= 25):
                prior_ytd_val = e["val"]
                break

    return latest_k["val"] + latest_q_ytd["val"] - prior_ytd_val


def annual_values(facts, *concepts, n=15, unit="USD"):
    """
    Return [(fiscal_year_end_date_str, value), ...] for the last n annual
    10-K filings, validated to be ~365 days long.
    """
    entries = find_concept_series(facts, *concepts, unit=unit)
    annual_entries = []
    for e in entries:
        if e.get("form") == "10-K" and "start" in e and "end" in e:
            try:
                sd = datetime.strptime(e["start"], "%Y-%m-%d")
                ed = datetime.strptime(e["end"],   "%Y-%m-%d")
                if 350 <= (ed - sd).days <= 380:
                    annual_entries.append(e)
            except Exception:
                pass
    annual = dedup_by_end(annual_entries, ("10-K",))
    return [(e["end"], e["val"]) for e in annual[:n]]


def extract_edgar_financials(facts, sic=None):
    """
    Pull net income, D&A, capex, EBIT, debt, cash from EDGAR and compute:
      oe_ttm           : current TTM owner earnings (NI + D&A − CapEx)
      oe_by_fiscal_end : {full_fiscal_year_end_date: annual_OE}
      avg_ebit         : 5-year average operating income (for EPV)
      entity_name      : company name from EDGAR entityName
      sector           : mapped from SIC code via sic_to_sector()
      debt_ttm         : TTM long-term debt (for net_debt / EV)
      cash_ttm         : TTM cash & equivalents (for net_debt / EV)
    """
    ni_ttm    = get_ttm(facts,
                        "NetIncomeLoss",
                        "NetIncomeLossAvailableToCommonStockholdersBasic",
                        "ProfitLoss",
                        "NetIncomeLossAllocatedToParent")
    # D&A concept list is intentionally identical to da_a below so that TTM
    # and annual series always resolve to the same XBRL concept (priority-first).
    da_ttm    = get_ttm(facts,
                        "DepreciationDepletionAndAmortization",
                        "DepreciationAndAmortization",
                        "Depreciation")
    # CapEx concept list is intentionally identical to capex_a below.
    capex_ttm = get_ttm(facts,
                        "PaymentsToAcquirePropertyPlantAndEquipment",
                        "PaymentsForCapitalImprovements",
                        "PaymentsToAcquireProductiveAssets")

    oe_ttm = None
    if ni_ttm is not None:
        oe_ttm = ni_ttm + (da_ttm or 0) - abs(capex_ttm or 0)

    ni_a    = annual_values(facts,
                            "NetIncomeLoss",
                            "NetIncomeLossAvailableToCommonStockholdersBasic",
                            "ProfitLoss",
                            "NetIncomeLossAllocatedToParent")
    da_a    = annual_values(facts,
                            "DepreciationDepletionAndAmortization",
                            "DepreciationAndAmortization",
                            "Depreciation")
    capex_a = annual_values(facts,
                            "PaymentsToAcquirePropertyPlantAndEquipment",
                            "PaymentsForCapitalImprovements",
                            "PaymentsToAcquireProductiveAssets")

    ni_d    = {e[0]: e[1] for e in ni_a}
    da_d    = {e[0]: e[1] for e in da_a}
    capex_d = {e[0]: e[1] for e in capex_a}

    oe_by_fiscal_end = {}
    for end_date in sorted(ni_d.keys()):
        oe_by_fiscal_end[end_date] = (
            ni_d[end_date]
            + da_d.get(end_date, 0)
            - abs(capex_d.get(end_date, 0))
        )

    ebit_a   = annual_values(facts,
                             "OperatingIncomeLoss",
                             "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest")
    avg_ebit = float(np.mean([v for _, v in ebit_a[:5]])) if ebit_a else None

    # Shares: use most recent 10-K annual figure (period-matched, not TTM blend).
    # WeightedAverageNumberOfSharesOutstandingBasic matches EPS denominator.
    shares_a = annual_values(facts,
                             "WeightedAverageNumberOfSharesOutstandingBasic",
                             "CommonStockSharesOutstanding",
                             unit="shares")
    shares_by_fiscal_end = {e[0]: e[1] for e in shares_a}
    shares_ttm = shares_a[0][1] if shares_a else None

    # Debt and cash for EV calculation — TTM values from EDGAR
    debt_ttm = get_ttm(facts,
                       "LongTermDebt",
                       "LongTermDebtNoncurrent",
                       "LongTermDebtAndCapitalLeaseObligations",
                       "DebtAndCapitalLeaseObligations")
    cash_ttm = get_ttm(facts,
                       "CashAndCashEquivalentsAtCarryingValue",
                       "CashAndCashEquivalents",
                       "Cash")

    # Entity name and sector from EDGAR metadata
    entity_name = facts.get("entityName", "")
    sector      = sic_to_sector(sic) if sic is not None else "default"

    return {
        "oe_ttm":               oe_ttm,
        "oe_by_fiscal_end":     oe_by_fiscal_end,
        "shares_ttm":           shares_ttm,
        "shares_by_fiscal_end": shares_by_fiscal_end,
        "avg_ebit":             avg_ebit,
        "tax_rate":             0.21,
        "debt_ttm":             debt_ttm,
        "cash_ttm":             cash_ttm,
        "entity_name":          entity_name,
        "sector":               sector,
        "fetched_at":           datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════════════
# OE LOOKUP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _latest_oe_before(oe_by_fiscal_end: dict, as_of: pd.Timestamp):
    """
    Return (fiscal_end_date_str, oe_value) for the most recent fiscal year-end
    whose date is strictly ≤ as_of.
    Returns (None, None) when no data is available before as_of.
    """
    as_of_str  = as_of.strftime("%Y-%m-%d")
    candidates = [(d, v) for d, v in oe_by_fiscal_end.items() if d <= as_of_str]
    if not candidates:
        return None, None
    best = max(candidates, key=lambda x: x[0])
    return best[0], best[1]


def _cagr_ending_at(oe_by_fiscal_end: dict, end_date_str: str):
    """
    Trailing CAGR of OE anchored to end_date_str.
    Tries spans of 5, 4, 3, 2 years.  Returns None when not computable.
    Returns sentinel "NEGATIVE_OE" when endpoints are found but OE is negative.
    """
    dates = sorted(oe_by_fiscal_end.keys())
    if end_date_str not in dates:
        return None
    end_val = oe_by_fiscal_end[end_date_str]
    end_dt  = datetime.strptime(end_date_str, "%Y-%m-%d")

    for span_years in [5, 4, 3, 2]:
        target_start  = end_dt - timedelta(days=span_years * 365 + 30)
        target_end_w  = end_dt - timedelta(days=span_years * 365 - 30)
        start_cands   = [d for d in dates
                         if target_start.strftime("%Y-%m-%d") <= d
                         <= target_end_w.strftime("%Y-%m-%d")]
        if not start_cands:
            continue
        start_date_str = min(start_cands)
        start_val      = oe_by_fiscal_end[start_date_str]
        start_dt       = datetime.strptime(start_date_str, "%Y-%m-%d")
        actual_years   = (end_dt - start_dt).days / 365.25
        if actual_years < 1.5:
            continue
        if start_val > 0 and end_val > 0:
            return (end_val / start_val) ** (1.0 / actual_years) - 1
        if start_val <= 0 or end_val <= 0:
            return "NEGATIVE_OE"
    return None

# ══════════════════════════════════════════════════════════════════════════════
# METRICS LAYER
# ══════════════════════════════════════════════════════════════════════════════

def compute_epv_per_share(avg_ebit, tax_rate, wacc, shares):
    if avg_ebit is None or not shares or shares <= 0:
        return None
    return avg_ebit * (1 - tax_rate) / wacc / shares


def metrics_at_price(oe, ev, mc, oe_growth_decimal, epv_per_share, shares):
    """
    Compute OE metrics at a given price / OE snapshot.

    oe_growth_decimal : CAGR as a decimal (e.g. 0.12 for 12 %).
    Stored in output as percentage (× 100).

    OE Multiple and OE Yield both use market-cap (mc) as the price basis so
    that 1 / oe_multiple == oe_yield / 100 (i.e. they are inverses of each
    other).

    OE-PEG = oe_multiple / (cagr_as_percent)
           = oe_multiple / (oe_growth_decimal × 100)
    """
    if oe is None or oe == 0:
        return {}

    oe_is_negative = oe < 0

    if oe_is_negative:
        oe_yield    = None
        oe_multiple = None
        oe_peg      = None
    else:
        oe_yield    = (oe / mc * 100)  if mc else None
        oe_multiple = (mc / oe)        if mc else None

    oe_peg = None
    if (oe_multiple is not None
            and oe_growth_decimal is not None
            and oe_growth_decimal != "NEGATIVE_OE"
            and not oe_is_negative):
        growth_pct = oe_growth_decimal * 100
        if growth_pct != 0:
            oe_peg = oe_multiple / growth_pct

    oeps = (oe / shares) if shares else None

    oe_growth_store = None if (oe_growth_decimal is None or oe_growth_decimal == "NEGATIVE_OE") \
                           else oe_growth_decimal

    return {
        "oe":          round(oe / 1e9, 4)               if oe                 is not None else None,
        "oeps":        round(oeps, 2)                    if oeps               is not None else None,
        "oe_yield":    round(oe_yield, 4)                if oe_yield           is not None else None,
        "oe_multiple": round(oe_multiple, 4)             if oe_multiple        is not None else None,
        "oe_growth":   round(oe_growth_store * 100, 4)   if oe_growth_store    is not None else None,
        "oe_peg":      round(oe_peg, 4)                  if oe_peg             is not None else None,
        "epv":         round(epv_per_share, 4)           if epv_per_share      is not None else None,
        "oe_negative": oe_is_negative,
    }


def pct_diff(a, b):
    if a is None or b is None or b == 0:
        return None
    return round((a - b) / abs(b) * 100, 2)


def diff_block(curr, peak):
    return {k: pct_diff(curr.get(k), peak.get(k))
            for k in ("oe", "oeps", "oe_yield", "oe_multiple", "oe_growth", "oe_peg", "epv")}

# ══════════════════════════════════════════════════════════════════════════════
# CALCULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_ticker_result(symbol, financials, ticker_info, hist_adj, hist_close):
    """
    ticker_info keys used: shortName, sector, sharesOutstanding,
                           totalDebt, totalCash
    hist_adj  : pd.Series of ADJUSTED-close prices (split-adjusted).
                Used for ALL price and MC calculations. Correct because EDGAR
                retroactively restates historical share counts in each 10-K filing
                (2-3 year lookback), so adjClose × EDGAR_restated_shares = correct MC.
    hist_close: pd.Series of UNADJUSTED close prices. Stored in cache for reference
                but NOT used for calculations — using close with EDGAR restated shares
                would overstate MC by the split factor (e.g. 4× for AAPL post-2020).
    NOTE: For fiscal years older than EDGAR's restatement window (~3 years before a
          split), share counts may not be restated, causing OE multiples to be
          understated for those older crisis periods. This is an EDGAR data limitation.
    """
    result = {
        "ticker":        symbol,
        "company_name":  ticker_info.get("shortName", symbol),
        "error":         None,
        "sector":        ticker_info.get("sector", "default"),
        "data_source":   "SEC EDGAR + Tiingo + FRED",
        "current":       {},
        "peak_52w":      {},
        "bear_markets":  [],
        "discount_metrics": {
            "z_score_5y":       None,
            "z_score_10y":      None,
            "erp_spread":       None,
            "erp_yield_used":   None,
            "erp_yield_live":   GLOBAL_10Y_YIELD_LIVE,
        },
        "last_updated":          datetime.now().isoformat(),
        "edgar_fetched_at":      financials.get("fetched_at", ""),
        "dca_signal":            "—",
    }

    if hist_adj.empty:
        result["error"] = "No price data"
        return result

    oe_by_fiscal_end     = financials.get("oe_by_fiscal_end", {})
    oe_ttm               = financials.get("oe_ttm")
    shares_ttm           = financials.get("shares_ttm")

    result["oe_negative_warning"] = (oe_ttm is not None and oe_ttm < 0)

    shares_by_fiscal_end = financials.get("shares_by_fiscal_end", {})
    avg_ebit             = financials.get("avg_ebit")
    tax_rate             = financials.get("tax_rate", 0.21)

    # Prefer EDGAR shares (period-matched); fall back to ticker_info value
    info_shares = ticker_info.get("sharesOutstanding")
    shares      = shares_ttm or info_shares

    # Net debt from EDGAR (debt_ttm / cash_ttm); fall back to ticker_info values
    total_debt = financials.get("debt_ttm") or ticker_info.get("totalDebt") or 0
    total_cash = financials.get("cash_ttm") or ticker_info.get("totalCash") or 0
    net_debt   = total_debt - total_cash

    wacc = get_wacc(result["sector"])
    epv  = compute_epv_per_share(avg_ebit, tax_rate, wacc, shares)

    def ev_mc(price, sh=None):
        s = sh if sh is not None else shares
        if not s:
            return None, None
        mc = price * s
        return mc + net_debt, mc

    def _shares_at(ts: pd.Timestamp) -> float:
        if not shares_by_fiscal_end:
            return shares
        ts_str     = ts.strftime("%Y-%m-%d")
        candidates = [(d, v) for d, v in shares_by_fiscal_end.items() if d <= ts_str]
        if not candidates:
            return shares
        return max(candidates, key=lambda x: x[0])[1]

    def oe_and_growth_at(ts: pd.Timestamp):
        fend, oe_val = _latest_oe_before(oe_by_fiscal_end, ts)
        if fend is None:
            # No EDGAR annual data exists before this date.
            # Return None so callers skip the metric rather than silently
            # inheriting the current-TTM OE — which would produce
            # nonsensical multiples and yields for historical periods.
            return None, None, shares
        cagr = _cagr_ending_at(oe_by_fiscal_end, fend)
        sh   = _shares_at(ts)
        return oe_val, cagr, sh

    def _safe_price(series: pd.Series, ts: pd.Timestamp) -> float:
        """Get price from series at ts; fall back to nearest prior trading day."""
        if series.empty:
            return None
        if ts in series.index:
            return float(series[ts])
        idx = series.index.searchsorted(ts, side="right") - 1
        return float(series.iloc[max(idx, 0)])

    def _price_at(ts: pd.Timestamp, fiscal_sh: float) -> float:
        """
        Hybrid price selection to pair correctly with EDGAR share counts.

        EDGAR retroactively restates historical share counts in each annual 10-K
        (typically 2-3 comparative years). After a split, restated shares are
        inflated by the split factor; adjClose is deflated by the same factor,
        so adjClose × restated_shares = correct MC.

        For fiscal years older than EDGAR's restatement window, share counts are
        NOT retroactively restated (pre-split counts). Using adjClose (which IS
        split-adjusted) with non-restated shares understates MC by the split factor.

        Detection: if fiscal_sh / current_shares < 0.5, the fiscal year shares are
        likely non-restated (pre-split). Use unadjusted close in that case.
        Otherwise (restated or post-split), use adjClose.
        """
        if (shares and shares > 0 and fiscal_sh and fiscal_sh > 0
                and fiscal_sh / shares < 0.5):
            # Non-restated shares → use unadjusted close
            return _safe_price(hist_close, ts)
        # Restated or post-split shares → use split-adjusted close
        return _safe_price(hist_adj, ts)

    # ── CURRENT ───────────────────────────────────────────────────────────────
    today_ts      = hist_adj.index[-1]
    current_price = float(hist_adj.iloc[-1])

    _, curr_cagr, curr_shares = oe_and_growth_at(today_ts)
    ev_c, mc_c   = ev_mc(current_price, curr_shares)
    if oe_ttm and mc_c:
        m_c = metrics_at_price(oe_ttm, ev_c, mc_c, curr_cagr, epv, curr_shares)
    else:
        m_c = {}
    m_c["price"]     = round(current_price, 2)
    result["current"] = m_c

    # ── 52-WEEK PEAK (Block 2 — E through I) ─────────────────────────────────
    peak52_date, _ = _52w_peak(hist_adj, today_ts)
    if peak52_date is not None:
        oe_pk, cagr_pk, sh_pk = oe_and_growth_at(peak52_date)
        peak52_price = _price_at(peak52_date, sh_pk)
        ev_p,  mc_p  = ev_mc(peak52_price, sh_pk)
        if oe_pk and mc_p:
            m_p = metrics_at_price(oe_pk, ev_p, mc_p, cagr_pk, epv, sh_pk)
        else:
            m_p = {}
        m_p["price"] = round(peak52_price, 2)
        m_p["date"]  = str(peak52_date.date())
    else:
        m_p = {}
    result["peak_52w"] = m_p

    if result["current"] and result["peak_52w"]:
        result["vs_peak_diff"] = diff_block(result["current"], result["peak_52w"])

    # ── CRISIS BLOCKS (Block 3 — J through N) ────────────────────────────────
    for crisis in CRISIS_PERIODS:
        crisis_start_ts = pd.Timestamp(crisis["start"])
        crisis_end_ts   = pd.Timestamp(crisis["end"])

        if crisis_start_ts < hist_adj.index[0]:
            continue

        effective_end_str = min(crisis_end_ts, today_ts).strftime("%Y-%m-%d")

        pre_peak_date, _ = _52w_peak(hist_adj, crisis_start_ts)
        if pre_peak_date is None:
            continue

        trough_date, _ = _trough_in_window(hist_adj, crisis["start"], effective_end_str)
        if trough_date is None:
            continue

        # Get OE + shares first (needed for _price_at split-detection)
        oe_pk,  cagr_pk, sh_pk = oe_and_growth_at(pre_peak_date)
        oe_tr,  cagr_tr, sh_tr = oe_and_growth_at(trough_date)

        # Pick price consistent with EDGAR shares (adjClose if restated, close if not)
        pre_peak_price = _price_at(pre_peak_date, sh_pk)
        trough_price   = _price_at(trough_date,   sh_tr)

        drawdown = (trough_price - pre_peak_price) / pre_peak_price * 100
        if drawdown >= -5:
            continue

        # Skip crisis entirely if EDGAR has no OE data for either date.
        # This happens when the crisis predates EDGAR XBRL coverage (~2009)
        # or when the ticker was not yet public / reporting at that time.
        if oe_pk is None and oe_tr is None:
            continue

        ev_pk,  mc_pk  = ev_mc(pre_peak_price, sh_pk)
        ev_tr,  mc_tr  = ev_mc(trough_price,   sh_tr)

        mp = metrics_at_price(oe_pk, ev_pk, mc_pk, cagr_pk, epv, sh_pk) if (oe_pk and mc_pk) else {}
        mb = metrics_at_price(oe_tr, ev_tr, mc_tr, cagr_tr, epv, sh_tr) if (oe_tr and mc_tr) else {}

        mb["price"]        = round(trough_price,   2)
        mb["peak_price"]   = round(pre_peak_price, 2)
        mb["crisis_name"]  = crisis["name"]
        mb["peak_date"]    = str(pre_peak_date.date())
        mb["trough_date"]  = str(trough_date.date())
        mb["drawdown_pct"] = round(abs(drawdown), 2)
        mb["peak_metrics"] = mp

        result["bear_markets"].append(mb)

    # ── Z-SCORES (Block 1-A/B) ────────────────────────────────────────────────
    # Use PER-SHARE annual OE for Z-scores — two reasons:
    #   1) Like-for-like: compare annual OE/sh against annual OE/sh history
    #      (TTM OE includes the latest quarter's uplift vs an annual distribution,
    #       which systematically overstates the Z-score).
    #   2) Per-share normalizes for buybacks: AAPL retired ~4B shares over 10Y,
    #      so total-dollar OE growth overstates per-share improvement.
    if oe_by_fiscal_end and shares_by_fiscal_end:
        now = datetime.now()
        cutoff_5y  = (now - timedelta(days=5  * 365)).strftime("%Y-%m-%d")
        cutoff_10y = (now - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

        # Build per-share OE series (period-matched shares)
        oeps_series = {}
        for d, oe_val in oe_by_fiscal_end.items():
            sh = shares_by_fiscal_end.get(d) or shares
            if sh and sh > 0:
                oeps_series[d] = oe_val / sh

        # Latest annual OE/share (not TTM) as the "current" reference value
        if oeps_series:
            latest_annual_fend = max(oeps_series.keys())
            latest_annual_oeps = oeps_series[latest_annual_fend]

            vals_5y  = [v for d, v in oeps_series.items() if d >= cutoff_5y]
            vals_10y = [v for d, v in oeps_series.items() if d >= cutoff_10y]

            if len(vals_5y) >= 2:
                arr5 = np.array(vals_5y, dtype=float)
                std5 = arr5.std()
                if std5 > 0:
                    result["discount_metrics"]["z_score_5y"] = round(
                        (latest_annual_oeps - arr5.mean()) / std5, 2)

            if len(vals_10y) >= 2:
                arr10 = np.array(vals_10y, dtype=float)
                std10 = arr10.std()
                if std10 > 0:
                    result["discount_metrics"]["z_score_10y"] = round(
                        (latest_annual_oeps - arr10.mean()) / std10, 2)

    # ── ERP SPREAD (Block 1-C) ────────────────────────────────────────────────
    # Use latest ANNUAL OE yield (not TTM) for ERP — consistent with Z-score
    # methodology and avoids the Q1 partial-year uplift inflating the yield.
    annual_oe_yield = None
    if oeps_series and current_price:
        try:
            latest_oeps_val = oeps_series[max(oeps_series.keys())]
            annual_oe_yield = latest_oeps_val / current_price * 100
        except Exception:
            pass
    erp_yield_src = annual_oe_yield if annual_oe_yield is not None \
                    else result["current"].get("oe_yield")
    if erp_yield_src is not None:
        result["discount_metrics"]["erp_spread"]     = round(
            erp_yield_src - GLOBAL_10Y_YIELD, 2)
        result["discount_metrics"]["erp_yield_used"] = round(GLOBAL_10Y_YIELD, 2)
        result["discount_metrics"]["erp_yield_live"] = GLOBAL_10Y_YIELD_LIVE

    # ── DCA SIGNAL ────────────────────────────────────────────────────────────
    z5 = result["discount_metrics"].get("z_score_5y")

    if result.get("oe_negative_warning", False):
        result["dca_signal"] = "⚠️ NEGATIVE OE"
    elif (z5 is not None and z5 <= -1.5):
        result["dca_signal"] = "🟢 STRONG DCA"
    elif (z5 is not None and z5 <= -0.5):
        result["dca_signal"] = "🟡 ACCUMULATE"
    else:
        result["dca_signal"] = "🔴 WAIT"

    return result

# ══════════════════════════════════════════════════════════════════════════════
# CACHE + RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def load_edgar_cache():
    try:
        with open(EDGAR_CACHE) as f:
            data = json.load(f)
        if data.get("__schema_version__", 1) < CACHE_SCHEMA_VERSION:
            print(f"WARNING: edgar_cache.json is schema v{data.get('__schema_version__', 1)}, "
                  f"need v{CACHE_SCHEMA_VERSION}. Forcing full EDGAR refresh.")
            return {}
        return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"WARNING: edgar_cache.json is corrupted ({e}). Forcing full EDGAR refresh.")
        return {}


def save_edgar_cache(cache):
    cache["__schema_version__"] = CACHE_SCHEMA_VERSION
    with open(EDGAR_CACHE, "w") as f:
        json.dump(cache, f, indent=2, default=str)
    print(f"EDGAR cache saved: {len(cache) - 1} tickers (schema v{CACHE_SCHEMA_VERSION})")


def refresh_edgar_cache(tickers):
    """
    Fetch EDGAR company facts + SIC (from submissions) for every ticker.
    SIC is used to derive sector for WACC; entity name is stored for display.
    """
    cache = load_edgar_cache()
    total = len(tickers)
    ok, failed = 0, []
    for i, sym in enumerate(tickers):
        cik = get_cik(sym)
        if cik is None:
            failed.append(sym)
            continue

        print(f"  [{i+1}/{total}] {sym} (CIK {cik})...", end=" ", flush=True)

        # Fetch SIC from EDGAR submissions endpoint
        sic = None
        try:
            subs_r = requests.get(EDGAR_SUBS_URL.format(cik=cik),
                                  headers=EDGAR_HEADERS, timeout=30)
            if subs_r.status_code == 200:
                sic = subs_r.json().get("sic")
        except Exception:
            pass
        time.sleep(0.4)   # polite pause between submissions and facts calls (SEC rate limit)

        # Fetch company facts
        facts = fetch_edgar_facts(cik)
        if facts is None:
            print("FAILED")
            failed.append(sym)
            continue
        try:
            fin = extract_edgar_financials(facts, sic=sic)
            cache[sym] = fin
            ok += 1
            oe_str  = f"${fin['oe_ttm']/1e9:.2f}B" if fin.get("oe_ttm") else "n/a"
            sec_str = fin.get("sector", "default")
            print(f"OK  OE={oe_str}  sector={sec_str}")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(sym)
        time.sleep(0.5)   # ~2 req/sec across submissions + facts calls, well under SEC's 10 req/sec limit

    save_edgar_cache(cache)
    print(f"\nEDGAR refresh complete: {ok} ok, {len(failed)} failed")
    if failed:
        print(f"  Failed tickers: {failed}")
    return cache


def run_prices_only(tickers, edgar_cache):
    """
    Compute metrics for all tickers using:
      - Tiingo   : historical + current prices (with price_cache.json)
      - FRED     : live 10Y Treasury yield
      - EDGAR    : all fundamentals (from edgar_cache)
    """
    global GLOBAL_10Y_YIELD, GLOBAL_10Y_YIELD_LIVE

    # 1. Fetch 10Y yield from FRED
    GLOBAL_10Y_YIELD, GLOBAL_10Y_YIELD_LIVE = fetch_10y_yield()
    src = "live (FRED)" if GLOBAL_10Y_YIELD_LIVE else "fallback"
    print(f"10Y Treasury Yield: {GLOBAL_10Y_YIELD:.2f}% ({src})")

    # 2. Load price cache once; it will be mutated per ticker and saved at end
    price_cache = load_price_cache()

    results = {}
    total   = len(tickers)

    for i, sym in enumerate(tickers):
        print(f"  [{i+1}/{total}] {sym}", end=" ", flush=True)
        fin = edgar_cache.get(sym)
        if not fin:
            results[sym] = {
                "ticker": sym, "error": "No EDGAR cache", "sector": None,
                "current": {}, "peak_52w": {}, "bear_markets": [],
                "last_updated": datetime.now().isoformat(),
            }
            print("(no EDGAR cache)")
            continue

        # Build ticker_info entirely from EDGAR cache — no yfinance needed
        ticker_info = {
            "shortName":         fin.get("entity_name") or sym,
            "sector":            fin.get("sector", "default"),
            "sharesOutstanding": fin.get("shares_ttm"),
            "totalDebt":         fin.get("debt_ttm") or 0,
            "totalCash":         fin.get("cash_ttm") or 0,
        }

        for attempt in range(3):
            try:
                hist_adj, hist_close = fetch_tiingo_prices(sym, price_cache)
                if hist_adj.empty:
                    raise ValueError("Empty price history from Tiingo")

                results[sym] = compute_ticker_result(sym, fin, ticker_info, hist_adj, hist_close)
                price_str = results[sym].get("current", {}).get("price", "n/a")
                print(f"${price_str}")
                break
            except Exception as e:
                if attempt == 2:
                    results[sym] = {
                        "ticker": sym, "error": f"Price fetch failed: {e}",
                        "sector": None,
                        "current": {}, "peak_52w": {}, "bear_markets": [],
                        "last_updated": datetime.now().isoformat(),
                    }
                    print(f"ERROR: {e}")
                else:
                    time.sleep(2)

    # 3. Persist updated price cache
    save_price_cache(price_cache)
    return results


def run_edgar_and_prices(tickers):
    print("Phase 1: Refreshing EDGAR financial data...")
    edgar_cache = refresh_edgar_cache(tickers)
    print("\nPhase 2: Fetching prices and computing metrics...")
    return run_prices_only(tickers, edgar_cache)


def save_results(results, path=OE_DATA):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} tickers → {path}")


if __name__ == "__main__":
    run_mode = os.environ.get("RUN_MODE", "prices_only").strip().lower()
    print("=" * 60)
    print(f"OE Dashboard Calculator v6 — Mode: {run_mode}")
    print("=" * 60)

    # Load CIK mapping from SEC
    print("Fetching official SEC CIK mapping...")
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=EDGAR_HEADERS, timeout=10)
        for k, v in r.json().items():
            DYNAMIC_CIK_MAP[v["ticker"]] = str(v["cik_str"]).zfill(10)
        print(f"  {len(DYNAMIC_CIK_MAP)} CIKs loaded.")
    except Exception as e:
        print(f"  Failed to fetch SEC mapping: {e}")

    if run_mode == "edgar_and_prices":
        results = run_edgar_and_prices(TICKERS)
    else:
        cache = load_edgar_cache()
        if not cache:
            print("No EDGAR cache found — switching to full refresh (edgar_and_prices)...")
            results = run_edgar_and_prices(TICKERS)
        else:
            results = run_prices_only(TICKERS, cache)

    save_results(results)
