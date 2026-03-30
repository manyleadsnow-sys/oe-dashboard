"""
Owner Earnings Dashboard - Core Calculation Engine

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
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import requests
import warnings
warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
EDGAR_HEADERS = {
    "User-Agent": "Gustavo Gonzalez gusqweenglish@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}
EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_CACHE          = "edgar_cache.json"
OE_DATA              = "oe_data.json"
CACHE_SCHEMA_VERSION = 5   # v5: capex_ttm concept list unified with capex_a

# Crisis windows — trough is the lowest price inside [start, end].
# Pre-crisis peak is defined as the 52-week high in the year BEFORE start.
# 2008 GFC added per spec ("since 2007").
CRISIS_PERIODS = [
    {"name": "April 2025 Crash",       "start": "2025-03-01", "end": "2025-06-30"},
    {"name": "2022 Bear Market",        "start": "2022-01-01", "end": "2022-10-14"},
    {"name": "2020 COVID-19 Crash",     "start": "2020-02-15", "end": "2020-03-23"},
    {"name": "2018 Rate/Trade Selloff", "start": "2018-09-20", "end": "2018-12-26"},
    {"name": "2015-2016 Selloff",       "start": "2015-06-01", "end": "2016-02-11"},
    {"name": "2008 GFC",                "start": "2007-10-01", "end": "2009-03-09"},
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
    "AAPL","GOOG","LLY","TSLA","AMZN",
]

HARDCODED_CIKS  = {}
DYNAMIC_CIK_MAP = {}

GLOBAL_10Y_YIELD      = 4.2    # fallback
GLOBAL_10Y_YIELD_LIVE = False  # True only when live fetch succeeded

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


def extract_edgar_financials(facts):
    """
    Pull net income, D&A, capex, and EBIT from EDGAR and compute:
      oe_ttm           : current TTM owner earnings (NI + D&A − CapEx)
      oe_by_fiscal_end : {full_fiscal_year_end_date: annual_OE}
                         Keyed by full date string (e.g. "2023-09-30"), NOT
                         just year — enables precise "latest OE before date" lookups.
      avg_ebit         : 5-year average operating income (for EPV)
    """
    ni_ttm    = get_ttm(facts,
                        "NetIncomeLoss",
                        "NetIncomeLossAvailableToCommonStockholdersBasic",
                        "ProfitLoss",
                        "NetIncomeLossAllocatedToParent")
    # D&A concept list is intentionally identical to da_a below so that TTM
    # and annual series always resolve to the same XBRL concept (priority-first).
    # DepreciationAmortizationAndAccretionNet and AmortizationOfIntangibleAssets
    # are excluded: the former can include non-D&A accretion items; the latter
    # is a subset of total D&A and would undercount if used as the sole figure.
    da_ttm    = get_ttm(facts,
                        "DepreciationDepletionAndAmortization",
                        "DepreciationAndAmortization",
                        "Depreciation")
    # CapEx concept list is intentionally identical to capex_a below so that TTM
    # and annual series always resolve to the same XBRL concept (priority-first).
    # PaymentsToAcquireBusinessesAndPropertyPlantAndEquipment is excluded because
    # it bundles acquisition payments with CapEx, overstating maintenance/growth capex.
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

    # Full date-string keys (e.g. "2023-09-30")
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

    # Shares from EDGAR: use the most recent 10-K annual figure, NOT a TTM
    # blend. Shares don't accumulate like income — adding Q YTD deltas produces
    # a nonsensical interpolated count. The latest fiscal year-end 10-K value
    # is the correct period-matched denominator for OE per share.
    # WeightedAverageNumberOfSharesOutstandingBasic is listed first because it
    # matches the denominator used in EPS computation, making it consistent with
    # net income as the OE numerator. CommonStockSharesOutstanding is a point-
    # in-time count at fiscal year-end and can diverge from weighted average
    # for companies with active buyback programmes like Apple.
    shares_a = annual_values(facts,
                             "WeightedAverageNumberOfSharesOutstandingBasic",
                             "CommonStockSharesOutstanding",
                             unit="shares")
    shares_by_fiscal_end = {e[0]: e[1] for e in shares_a}
    # shares_ttm = latest annual 10-K share count (first entry after dedup+sort)
    shares_ttm = shares_a[0][1] if shares_a else None

    return {
        "oe_ttm":              oe_ttm,
        "oe_by_fiscal_end":    oe_by_fiscal_end,
        "shares_ttm":          shares_ttm,
        "shares_by_fiscal_end": shares_by_fiscal_end,
        "avg_ebit":            avg_ebit,
        "tax_rate":            0.21,
        "fetched_at":          datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════════════
# OE LOOKUP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _latest_oe_before(oe_by_fiscal_end: dict, as_of: pd.Timestamp):
    """
    Return (fiscal_end_date_str, oe_value) for the most recent fiscal year-end
    whose date is strictly ≤ as_of.
    This guarantees that OE at the "peak" date and OE at the "trough" date
    will differ whenever those dates straddle a fiscal year-end boundary,
    and will be the same only when both fall within the same fiscal year
    (which is correct — no new annual data was published in between).
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
    The span is measured between actual fiscal year-end dates in the dict,
    so it tolerates companies with non-December fiscal year-ends.
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
    other).  Using EV for the multiple but mc for the yield broke that identity
    and produced inconsistent signals for net-cash companies like Apple.

    OE-PEG = oe_multiple / (cagr_as_percent)
           = oe_multiple / (oe_growth_decimal × 100)
    """
    if not oe or oe == 0:
        return {}

    oe_yield    = (oe / mc * 100)  if mc else None
    oe_multiple = (mc / oe)        if mc else None

    oe_peg = None
    if oe_multiple is not None and oe_growth_decimal:
        growth_pct = oe_growth_decimal * 100     # e.g. 12.0
        if growth_pct != 0:
            oe_peg = oe_multiple / growth_pct

    oeps = (oe / shares) if shares else None

    return {
        "oe":          round(oe / 1e9, 4)               if oe                 is not None else None,
        "oeps":        round(oeps, 2)                    if oeps               is not None else None,
        "oe_yield":    round(oe_yield, 4)                if oe_yield           is not None else None,
        "oe_multiple": round(oe_multiple, 4)             if oe_multiple        is not None else None,
        "oe_growth":   round(oe_growth_decimal * 100, 4) if oe_growth_decimal  is not None else None,
        "oe_peg":      round(oe_peg, 4)                  if oe_peg             is not None else None,
        "epv":         round(epv_per_share, 4)           if epv_per_share      is not None else None,
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

def compute_ticker_result(symbol, financials, yf_info, hist):
    result = {
        "ticker":        symbol,
        "company_name":  yf_info.get("shortName", symbol),
        "error":         None,
        "sector":        yf_info.get("sector", "default"),
        "data_source":   "SEC EDGAR + Yahoo Finance",
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

    if hist.empty:
        result["error"] = "No price data"
        return result

    oe_by_fiscal_end     = financials.get("oe_by_fiscal_end", {})
    oe_ttm               = financials.get("oe_ttm")
    shares_ttm           = financials.get("shares_ttm")
    shares_by_fiscal_end = financials.get("shares_by_fiscal_end", {})
    avg_ebit             = financials.get("avg_ebit")
    tax_rate             = financials.get("tax_rate", 0.21)

    # Prefer EDGAR shares (period-matched); fall back to yfinance live float.
    # yfinance fast_info/sharesOutstanding reflects the current market float and
    # drifts from any specific fiscal period for active buyback programmes.
    yf_shares = yf_info.get("sharesOutstanding") or yf_info.get("impliedSharesOutstanding")
    shares    = shares_ttm or yf_shares
    net_debt = (yf_info.get("totalDebt") or 0) - (yf_info.get("totalCash") or 0)
    wacc     = get_wacc(result["sector"])
    epv      = compute_epv_per_share(avg_ebit, tax_rate, wacc, shares)

    def ev_mc(price, sh=None):
        """Return (enterprise_value, market_cap) at the given price and share count.
        Uses period-matched sh when supplied; falls back to the TTM/yf share count."""
        s = sh if sh is not None else shares
        if not s:
            return None, None
        mc = price * s
        return mc + net_debt, mc

    def _shares_at(ts: pd.Timestamp) -> float:
        """
        Most recent EDGAR fiscal-year-end share count whose date is ≤ ts.
        Falls back to shares (TTM/yf) when EDGAR has no share history.
        """
        if not shares_by_fiscal_end:
            return shares
        ts_str     = ts.strftime("%Y-%m-%d")
        candidates = [(d, v) for d, v in shares_by_fiscal_end.items() if d <= ts_str]
        if not candidates:
            return shares
        return max(candidates, key=lambda x: x[0])[1]

    def oe_and_growth_at(ts: pd.Timestamp):
        """OE value + CAGR + period-matched shares at date ts."""
        fend, oe_val = _latest_oe_before(oe_by_fiscal_end, ts)
        if fend is None:
            return oe_ttm, None, shares
        cagr    = _cagr_ending_at(oe_by_fiscal_end, fend)
        sh      = _shares_at(ts)
        return oe_val, cagr, sh

    # ── CURRENT ───────────────────────────────────────────────────────────────
    current_price = float(hist.iloc[-1])
    today_ts      = hist.index[-1]

    _, curr_cagr, curr_shares = oe_and_growth_at(today_ts)
    ev_c, mc_c   = ev_mc(current_price, curr_shares)
    if oe_ttm and mc_c:
        m_c = metrics_at_price(oe_ttm, ev_c, mc_c, curr_cagr, epv, curr_shares)
    else:
        m_c = {}
    m_c["price"]     = round(current_price, 2)
    result["current"] = m_c

    # ── 52-WEEK PEAK (Block 2 — E through I) ─────────────────────────────────
    # Peak = highest close in the 52 weeks before today
    peak52_date, peak52_price = _52w_peak(hist, today_ts)
    if peak52_date is not None:
        oe_pk, cagr_pk, sh_pk = oe_and_growth_at(peak52_date)
        ev_p,  mc_p    = ev_mc(peak52_price, sh_pk)
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

        # Skip crises before the available price history
        if crisis_start_ts < hist.index[0]:
            continue

        # Clip ongoing crises to today
        effective_end_str = min(crisis_end_ts, today_ts).strftime("%Y-%m-%d")

        # Pre-crisis peak = 52-week high in the year before crisis start
        pre_peak_date, pre_peak_price = _52w_peak(hist, crisis_start_ts)
        if pre_peak_date is None:
            continue

        # Trough = lowest price inside the crisis window
        trough_date, trough_price = _trough_in_window(
            hist, crisis["start"], effective_end_str)
        if trough_date is None:
            continue

        drawdown = (trough_price - pre_peak_price) / pre_peak_price * 100
        if drawdown >= -5:      # ignore trivial moves
            continue

        # OE and CAGR anchored to each date independently
        oe_pk,  cagr_pk, sh_pk = oe_and_growth_at(pre_peak_date)
        oe_tr,  cagr_tr, sh_tr = oe_and_growth_at(trough_date)

        ev_pk,  mc_pk  = ev_mc(pre_peak_price, sh_pk)
        ev_tr,  mc_tr  = ev_mc(trough_price,   sh_tr)

        mp = metrics_at_price(oe_pk, ev_pk, mc_pk, cagr_pk, epv, sh_pk) if (oe_pk and mc_pk) else {}
        mb = metrics_at_price(oe_tr, ev_tr, mc_tr, cagr_tr, epv, sh_tr) if (oe_tr and mc_tr) else {}

        mb["price"]        = round(trough_price,   2)
        mb["peak_price"]   = round(pre_peak_price, 2)
        mb["crisis_name"]  = crisis["name"]
        mb["peak_date"]    = str(pre_peak_date.date())
        mb["trough_date"]  = str(trough_date.date())
        mb["drawdown_pct"] = round(abs(drawdown), 2)  # FIX: Wrapped in abs()
        mb["peak_metrics"] = mp

        result["bear_markets"].append(mb)

    # ── Z-SCORES (Block 1-A/B) ────────────────────────────────────────────────
    # Z-score the current TTM OE dollar level against the full distribution of
    # annual OE figures from oe_by_fiscal_end. All fiscal year-ends within the
    # lookback window are included — the TTM is computed from quarterly filings
    # and is NOT identical to any single annual value, so no self-comparison issue.
    if oe_by_fiscal_end and oe_ttm is not None:
        now = datetime.now()
        cutoff_5y  = (now - timedelta(days=5  * 365)).strftime("%Y-%m-%d")
        cutoff_10y = (now - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

        vals_5y  = [v for d, v in oe_by_fiscal_end.items() if d >= cutoff_5y]
        vals_10y = [v for d, v in oe_by_fiscal_end.items() if d >= cutoff_10y]

        if len(vals_5y) >= 2:
            arr5 = np.array(vals_5y, dtype=float)
            std5 = arr5.std()
            if std5 > 0:
                result["discount_metrics"]["z_score_5y"] = round(
                    (oe_ttm - arr5.mean()) / std5, 2)

        if len(vals_10y) >= 2:
            arr10 = np.array(vals_10y, dtype=float)
            std10 = arr10.std()
            if std10 > 0:
                result["discount_metrics"]["z_score_10y"] = round(
                    (oe_ttm - arr10.mean()) / std10, 2)

    # ── ERP SPREAD (Block 1-C) ────────────────────────────────────────────────
    if result["current"].get("oe_yield") is not None:
        result["discount_metrics"]["erp_spread"]     = round(
            result["current"]["oe_yield"] - GLOBAL_10Y_YIELD, 2)
        result["discount_metrics"]["erp_yield_used"] = round(GLOBAL_10Y_YIELD, 2)
        result["discount_metrics"]["erp_yield_live"] = GLOBAL_10Y_YIELD_LIVE

    # ── DCA SIGNAL ────────────────────────────────────────────────────────────
    z5   = result["discount_metrics"].get("z_score_5y")
    if (z5 is not None and z5 <= -1.5):
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
        # Reject caches built by an older version of extract_edgar_financials.
        # A v1/v2 cache may lack shares_ttm / shares_by_fiscal_end, or may have
        # been built with the old merge-all D&A concept fetch.
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
            fin = extract_edgar_financials(facts)
            cache[sym] = fin
            ok += 1
            oe_str = f"${fin['oe_ttm']/1e9:.2f}B" if fin.get("oe_ttm") else "n/a"
            print(f"OK (OE TTM: {oe_str})")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(sym)
        time.sleep(0.15)
    save_edgar_cache(cache)
    print(f"\nEDGAR refresh complete: {ok} ok, {len(failed)} failed")
    return cache


def run_prices_only(tickers, edgar_cache):
    global GLOBAL_10Y_YIELD, GLOBAL_10Y_YIELD_LIVE
    try:
        tnx     = yf.Ticker("^TNX")
        fetched = tnx.history(period="5d")["Close"].dropna()
        if fetched.empty:
            raise ValueError("Empty TNX series")
        # ^TNX is quoted in percent already (e.g. 44.4 = 4.44%) — no division needed.
        GLOBAL_10Y_YIELD      = float(fetched.iloc[-1])
        GLOBAL_10Y_YIELD_LIVE = True
        print(f"10Y Treasury Yield: {GLOBAL_10Y_YIELD:.2f}% (live)")
    except Exception as e:
        GLOBAL_10Y_YIELD_LIVE = False
        print(f"WARNING: TNX fetch failed ({e}). Fallback: {GLOBAL_10Y_YIELD:.2f}%")

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
            print("(no cache)")
            continue

        for attempt in range(3):
            try:
                time.sleep(1 + attempt)  # Slightly longer backoff
                t = yf.Ticker(sym)
                
                # 1. Guarantee share count using fast_info (bypasses standard .info rate limits)
                raw_info = t.info or {}
                shares = t.fast_info.get("shares") or raw_info.get("sharesOutstanding")
                
                # 2. FIX: Pull Enterprise Value Balance Sheet numbers safely
                bs = t.balance_sheet
                total_debt = 0.0
                total_cash = 0.0
                if bs is not None and not bs.empty:
                    if "Total Debt" in bs.index:
                        total_debt = float(bs.loc["Total Debt"].iloc[0]) if pd.notna(bs.loc["Total Debt"].iloc[0]) else 0.0
                    
                    if "Cash And Cash Equivalents" in bs.index:
                        total_cash = float(bs.loc["Cash And Cash Equivalents"].iloc[0]) if pd.notna(bs.loc["Cash And Cash Equivalents"].iloc[0]) else 0.0
                    elif "Total Cash" in bs.index:
                        total_cash = float(bs.loc["Total Cash"].iloc[0]) if pd.notna(bs.loc["Total Cash"].iloc[0]) else 0.0

                info = {
                    "shortName": raw_info.get("shortName", sym),
                    "sector": raw_info.get("sector", "default"),
                    "sharesOutstanding": shares,
                    "totalDebt": total_debt,
                    "totalCash": total_cash
                }

                # 3. Fetch history
                hist = t.history(period="max", interval="1d")["Close"].dropna()
                if hist.empty:
                    raise ValueError("Empty price history")
                
                # 4. Safely strip timezones to prevent Pandas TypeError crashes
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_convert(None)
                
                results[sym] = compute_ticker_result(sym, fin, info, hist)
                print(f"${results[sym].get('current', {}).get('price', 'n/a')}")
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
    print(f"OE Dashboard Calculator — Mode: {run_mode}")
    print("=" * 60)

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
            print("No cache found — switching to full EDGAR refresh...")
            results = run_edgar_and_prices(TICKERS)
        else:
            results = run_prices_only(TICKERS, cache)

    save_results(results)