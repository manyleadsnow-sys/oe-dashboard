"""
Owner Earnings Dashboard - Core Calculation Engine
Metrics: OE, OE Yield, OE Multiple (EV/OE), OE Growth Rate (10yr CAGR),
         OE-PEG Ratio, EPV (industry-adjusted WACC)
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, date
import json
import warnings
warnings.filterwarnings("ignore")

PEAK_START_DATE = "2022-10-12"

TICKERS = [
    # AI & Technology
    "AAPL","GOOG","META","MSFT","NVDA","PLTR","TSLA",
    # Communication Services
    "EA","NFLX","TMUS",
    # Consumer Discretionary
    "AMZN","CMG","CPRT","GRMN","LEN","MCD","ORLY","POOL","ROST","TSCO","ULTA",
    # Consumer Staples
    "BG","COST","HSY","KO","PEP","PG","PM","STZ","SYY","WMT",
    # Energy
    "BKR","CVX","EOG","EPD","EXE","FANG","SLB","TPL","VLO","XOM",
    # Financials
    "BRK-B","ACGL","AIZ","AJG","AON","ARES","AXP","BAC","BLK","BRO",
    "C","CB","CBOE","CBRE","CINF","CME","CPAY","EG","ERIE","FICO",
    "GS","IBKR","ICE","JPM","KKR","MA","MCO","MSCI","NDAQ","PGR",
    "RJF","SPGI","TRV","V","VRSK","WFC","WRB",
    # Health Care
    "A","BSX","CI","ABT","COO","HCA","IDXX","IQV","ISRG","JNJ",
    "LLY","MCK","MRK","MTD","REGN","RMD","SYK","TECH","VRTX","WAT","WST","ZTS",
    # Industrials
    "WM","MO","ADP","AXON","CAT","CTAS","DE","EME","EMR","ETN",
    "FAST","FIX","GD","GE","GWW","HON","HWM","LMT","NOC","ODFL",
    "OTIS","PH","PWR","ROK","ROL","ROP","TDG","TT",
    # Information Technology
    "ACN","ADI","ADSK","AMAT","AMD","ANET","APH","CDNS","CSCO","FTNT",
    "IT","KLAC","LRCX","MCHP","MPWR","MSI","NXPI","ON","PTC","Q",
    "SNPS","TEL","TER","TTD","TXN","TYL","VRSN","WDAY",
    # Materials
    "APD","AVY","CRH","ECL","FSLR","LIN","MLM","NUE","SHW","STLD","VMC",
    # Real Estate
    "AMT","CSGP","EXR","PSA","SBAC","VICI",
    # Utilities
    "AEP","AWK","CEG","D","DUK","ETR","NEE","NRG","PEG","SO","SRE","VST","XEL"
]

# Industry WACC mapping (approximate industry-adjusted rates)
INDUSTRY_WACC = {
    "Technology":           0.090,
    "Communication Services":0.085,
    "Consumer Cyclical":    0.085,
    "Consumer Defensive":   0.075,
    "Energy":               0.095,
    "Financial Services":   0.090,
    "Healthcare":           0.085,
    "Industrials":          0.085,
    "Real Estate":          0.075,
    "Basic Materials":      0.090,
    "Utilities":            0.065,
    "default":              0.085,
}

def get_wacc(sector: str) -> float:
    return INDUSTRY_WACC.get(sector, INDUSTRY_WACC["default"])


def safe_get(d: dict, *keys, default=None):
    for k in keys:
        if d and k in d:
            return d[k]
    return default


def compute_owner_earnings(cf: dict, bs_curr: dict, bs_prev: dict) -> float | None:
    """
    OE = Net Income + D&A - CapEx ± Working Capital changes
    WC change = (Current Assets - Cash) - (Current Liabilities - Short-term Debt)  [curr] minus [prev]
    """
    net_income = safe_get(cf, "Net Income", "NetIncome")
    da = safe_get(cf, "Depreciation & Amortization", "Reconciled Depreciation",
                  "DepreciationAndAmortization", "Depreciation")
    capex = safe_get(cf, "Capital Expenditure", "CapitalExpenditures")

    if net_income is None or da is None or capex is None:
        return None

    # CapEx is usually negative in yfinance cash flow
    capex_val = abs(capex)

    # Working capital change
    def wc(bs):
        if bs is None:
            return None
        ca = safe_get(bs, "Current Assets", "TotalCurrentAssets")
        cash = safe_get(bs, "Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents")
        cl = safe_get(bs, "Current Liabilities", "TotalCurrentLiabilities")
        std = safe_get(bs, "Short Long Term Debt", "CurrentDebt", "ShortTermDebt") or 0
        if ca is None or cl is None:
            return None
        return (ca - (cash or 0)) - (cl - std)

    wc_curr = wc(bs_curr)
    wc_prev = wc(bs_prev)

    if wc_curr is not None and wc_prev is not None:
        delta_wc = wc_curr - wc_prev
    else:
        delta_wc = 0

    return net_income + da - capex_val - delta_wc


def compute_epv(ticker_obj, sector: str, shares: float) -> float | None:
    """
    EPV = Adjusted EBIT × (1 - tax_rate) / WACC
    """
    try:
        inc = ticker_obj.financials
        if inc is None or inc.empty:
            return None
        row_ebit = None
        for label in ["EBIT", "Ebit", "Operating Income"]:
            if label in inc.index:
                row_ebit = inc.loc[label]
                break
        if row_ebit is None:
            return None

        # Average EBIT over available years (normalise)
        ebit_vals = row_ebit.dropna().values
        if len(ebit_vals) == 0:
            return None
        avg_ebit = np.mean(ebit_vals)

        # Tax rate
        tax_row = None
        for label in ["Tax Rate For Calcs", "Effective Tax Rate"]:
            if label in inc.index:
                tax_row = inc.loc[label]
                break
        if tax_row is not None:
            tax_rate = tax_row.dropna().mean()
            if np.isnan(tax_rate) or tax_rate <= 0 or tax_rate >= 1:
                tax_rate = 0.21
        else:
            tax_rate = 0.21

        wacc = get_wacc(sector)
        epv_total = avg_ebit * (1 - tax_rate) / wacc
        if shares and shares > 0:
            return epv_total / shares
        return None
    except Exception:
        return None


def get_historical_oe_series(ticker_obj) -> pd.Series:
    """
    Returns annual OE values indexed by year for up to 10 years.
    """
    try:
        cf_annual = ticker_obj.cashflow        # columns = dates
        bs_annual = ticker_obj.balance_sheet
        if cf_annual is None or cf_annual.empty:
            return pd.Series(dtype=float)

        results = {}
        cols = cf_annual.columns.tolist()
        bs_cols = bs_annual.columns.tolist() if (bs_annual is not None and not bs_annual.empty) else []

        for i, col in enumerate(cols):
            cf_dict = cf_annual[col].to_dict()
            bs_curr = bs_annual[bs_cols[i]].to_dict() if i < len(bs_cols) else None
            bs_prev = bs_annual[bs_cols[i+1]].to_dict() if (i+1) < len(bs_cols) else None
            oe = compute_owner_earnings(cf_dict, bs_curr, bs_prev)
            if oe is not None:
                yr = pd.to_datetime(col).year
                results[yr] = oe

        return pd.Series(results).sort_index()
    except Exception:
        return pd.Series(dtype=float)


def compute_oe_growth_10yr(oe_series: pd.Series) -> float | None:
    """10-year CAGR of Owner Earnings"""
    if len(oe_series) < 2:
        return None
    years = oe_series.index.tolist()
    oldest = oe_series.iloc[0]
    newest = oe_series.iloc[-1]
    n = years[-1] - years[0]
    if n <= 0 or oldest <= 0 or newest <= 0:
        return None
    return (newest / oldest) ** (1 / n) - 1


def detect_bear_markets(price_series: pd.Series, threshold=0.20, top_n=10):
    """
    Auto-detect bear market troughs: drawdowns > threshold from a rolling peak.
    Returns list of dicts: {peak_date, trough_date, peak_price, trough_price, drawdown_pct}
    """
    if price_series.empty:
        return []

    rolling_max = price_series.expanding().max()
    drawdown = (price_series - rolling_max) / rolling_max

    bears = []
    in_bear = False
    peak_date = None
    peak_price = None
    trough_date = None
    trough_price = None

    for dt, dd in drawdown.items():
        price = price_series[dt]
        roll_max = rolling_max[dt]

        if not in_bear and dd <= -threshold:
            in_bear = True
            # find the actual peak (last time price == rolling max before this)
            peak_idx = (price_series[:dt] == roll_max)
            peak_date = peak_idx[peak_idx].index[-1] if peak_idx.any() else dt
            peak_price = roll_max
            trough_date = dt
            trough_price = price

        elif in_bear:
            if price < trough_price:
                trough_date = dt
                trough_price = price
            elif dd > -threshold / 2:  # recovery
                bears.append({
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "peak_price": float(peak_price),
                    "trough_price": float(trough_price),
                    "drawdown_pct": float((trough_price - peak_price) / peak_price * 100)
                })
                in_bear = False

    if in_bear and trough_date is not None:
        bears.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "peak_price": float(peak_price),
            "trough_price": float(trough_price),
            "drawdown_pct": float((trough_price - peak_price) / peak_price * 100)
        })

    # Sort by drawdown depth, return top N
    bears.sort(key=lambda x: x["drawdown_pct"])
    return bears[:top_n]


def pct_diff(a, b):
    """Percentage difference from b to a: (a - b) / |b| * 100"""
    if a is None or b is None or b == 0:
        return None
    return round((a - b) / abs(b) * 100, 2)


def compute_metrics_at_price(oe_ttm, ev_at_price, market_cap_at_price, shares,
                              oe_growth, epv_per_share, price):
    """
    Given OE (TTM) and price-adjusted EV/market cap, compute all 6 metrics.
    OE itself doesn't change with price — it's fundamental.
    OE yield, multiple, PEG, EPV margin of safety do.
    """
    oe_yield = (oe_ttm / market_cap_at_price * 100) if (market_cap_at_price and market_cap_at_price != 0) else None
    oe_multiple = (ev_at_price / oe_ttm) if (oe_ttm and oe_ttm != 0 and ev_at_price) else None
    oe_peg = (oe_multiple / (oe_growth * 100)) if (oe_multiple and oe_growth and oe_growth != 0) else None
    epv = epv_per_share  # EPV per share is fundamental, compare to price
    return {
        "oe": round(oe_ttm / 1e9, 4) if oe_ttm else None,           # in $B
        "oe_yield": round(oe_yield, 4) if oe_yield else None,        # %
        "oe_multiple": round(oe_multiple, 4) if oe_multiple else None,
        "oe_growth": round(oe_growth * 100, 4) if oe_growth else None, # %
        "oe_peg": round(oe_peg, 4) if oe_peg else None,
        "epv": round(epv, 4) if epv else None,                       # per share $
    }


def fetch_ticker_data(symbol: str) -> dict:
    """
    Main function: fetch all data and compute all metrics for one ticker.
    Returns structured dict ready for JSON serialisation.
    """
    result = {
        "ticker": symbol,
        "error": None,
        "sector": None,
        "current": {},
        "peak_since_oct2022": {},
        "bear_markets": [],
        "last_updated": datetime.now().isoformat(),
    }

    try:
        t = yf.Ticker(symbol)
        info = t.info or {}

        sector = info.get("sector", "default")
        result["sector"] = sector
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        net_debt = (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)

        # ── Price history ───────────────────────────────────────────────
        hist_full = t.history(period="max", interval="1d")["Close"].dropna()
        # Strip timezone from index to avoid comparison errors
        hist_full.index = hist_full.index.tz_localize(None) if hist_full.index.tz is None else hist_full.index.tz_convert(None)
        hist_post_oct22 = hist_full[hist_full.index >= pd.Timestamp(PEAK_START_DATE)]

        current_price = float(hist_full.iloc[-1]) if not hist_full.empty else None
        if current_price is None:
            result["error"] = "No price data"
            return result

        # Peak since Oct 12 2022
        if hist_post_oct22.empty:
            result["error"] = "No price data since Oct 2022"
            return result
        peak_price = float(hist_post_oct22.max())
        peak_date  = hist_post_oct22.idxmax()

        # ── Financials ──────────────────────────────────────────────────
        cf_q  = t.quarterly_cashflow
        bs_q  = t.quarterly_balance_sheet

        # TTM cash flow (sum last 4 quarters)
        def ttm_sum(df, *labels):
            for lbl in labels:
                if df is not None and not df.empty and lbl in df.index:
                    vals = df.loc[lbl].dropna().iloc[:4]
                    if len(vals) >= 2:
                        return vals.sum()
            return None

        net_income = ttm_sum(cf_q, "Net Income", "NetIncome")
        da         = ttm_sum(cf_q, "Depreciation & Amortization",
                             "Reconciled Depreciation", "DepreciationAndAmortization")
        capex      = ttm_sum(cf_q, "Capital Expenditure", "CapitalExpenditures")

        bs_curr_dict = bs_q.iloc[:, 0].to_dict() if (bs_q is not None and not bs_q.empty) else None
        bs_prev_dict = bs_q.iloc[:, 1].to_dict() if (bs_q is not None and bs_q.shape[1] > 1) else None

        oe_cf = {"Net Income": net_income, "Depreciation & Amortization": da,
                 "Capital Expenditure": capex}
        oe_ttm = compute_owner_earnings(oe_cf, bs_curr_dict, bs_prev_dict)

        # Historical annual OE for growth rate
        oe_series = get_historical_oe_series(t)
        oe_growth = compute_oe_growth_10yr(oe_series)

        # EPV per share
        epv_per_share = compute_epv(t, sector, shares)

        # ── Helper: EV and Market Cap at any price ──────────────────────
        def ev_at(price_val):
            if shares is None:
                return None, None
            mc = price_val * shares
            ev = mc + net_debt
            return ev, mc

        # ── Current metrics ─────────────────────────────────────────────
        ev_curr, mc_curr = ev_at(current_price)
        if oe_ttm and mc_curr:
            result["current"] = compute_metrics_at_price(
                oe_ttm, ev_curr, mc_curr, shares, oe_growth, epv_per_share, current_price)
            result["current"]["price"] = round(current_price, 2)

        # ── Peak metrics ────────────────────────────────────────────────
        ev_peak, mc_peak = ev_at(peak_price)
        if oe_ttm and mc_peak:
            result["peak_since_oct2022"] = compute_metrics_at_price(
                oe_ttm, ev_peak, mc_peak, shares, oe_growth, epv_per_share, peak_price)
            result["peak_since_oct2022"]["price"] = round(peak_price, 2)
            result["peak_since_oct2022"]["date"]  = str(peak_date.date())

        # ── Diffs: current vs peak ───────────────────────────────────────
        if result["current"] and result["peak_since_oct2022"]:
            curr = result["current"]
            peak = result["peak_since_oct2022"]
            result["vs_peak_diff"] = {
                "oe":         pct_diff(curr.get("oe"),         peak.get("oe")),
                "oe_yield":   pct_diff(curr.get("oe_yield"),   peak.get("oe_yield")),
                "oe_multiple":pct_diff(curr.get("oe_multiple"),peak.get("oe_multiple")),
                "oe_growth":  pct_diff(curr.get("oe_growth"),  peak.get("oe_growth")),
                "oe_peg":     pct_diff(curr.get("oe_peg"),     peak.get("oe_peg")),
                "epv":        pct_diff(curr.get("epv"),        peak.get("epv")),
            }

        # ── Bear markets ────────────────────────────────────────────────
        bears = detect_bear_markets(hist_full)
        for bear in bears:
            bprice = bear["trough_price"]
            ev_b, mc_b = ev_at(bprice)
            if oe_ttm and mc_b:
                metrics_b = compute_metrics_at_price(
                    oe_ttm, ev_b, mc_b, shares, oe_growth, epv_per_share, bprice)
                metrics_b["price"]         = round(bprice, 2)
                metrics_b["peak_price"]    = round(bear["peak_price"], 2)
                metrics_b["peak_date"]     = str(bear["peak_date"].date()) if hasattr(bear["peak_date"], "date") else str(bear["peak_date"])
                metrics_b["trough_date"]   = str(bear["trough_date"].date()) if hasattr(bear["trough_date"], "date") else str(bear["trough_date"])
                metrics_b["drawdown_pct"]  = round(bear["drawdown_pct"], 2)
                # diff vs that bear's peak
                ev_bp, mc_bp = ev_at(bear["peak_price"])
                metrics_peak = compute_metrics_at_price(
                    oe_ttm, ev_bp, mc_bp, shares, oe_growth, epv_per_share, bear["peak_price"])
                metrics_b["vs_bear_peak_diff"] = {
                    "oe":         pct_diff(metrics_b.get("oe"),         metrics_peak.get("oe")),
                    "oe_yield":   pct_diff(metrics_b.get("oe_yield"),   metrics_peak.get("oe_yield")),
                    "oe_multiple":pct_diff(metrics_b.get("oe_multiple"),metrics_peak.get("oe_multiple")),
                    "oe_growth":  pct_diff(metrics_b.get("oe_growth"),  metrics_peak.get("oe_growth")),
                    "oe_peg":     pct_diff(metrics_b.get("oe_peg"),     metrics_peak.get("oe_peg")),
                    "epv":        pct_diff(metrics_b.get("epv"),        metrics_peak.get("epv")),
                }
                result["bear_markets"].append(metrics_b)

    except Exception as e:
        result["error"] = str(e)

    return result


def run_full_calculation(tickers=None, progress_callback=None) -> dict:
    """
    Run calculations for all tickers. Returns dict keyed by ticker.
    progress_callback(ticker, i, total) called after each ticker.
    """
    if tickers is None:
        tickers = TICKERS

    results = {}
    total = len(tickers)
    for i, sym in enumerate(tickers):
        results[sym] = fetch_ticker_data(sym)
        if progress_callback:
            progress_callback(sym, i + 1, total)

    return results


def save_results(results: dict, path="oe_data.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} tickers → {path}")


def load_results(path="oe_data.json") -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


if __name__ == "__main__":
    print(f"Running OE calculations for {len(TICKERS)} tickers...")
    results = run_full_calculation(
        progress_callback=lambda sym, i, n: print(f"  [{i}/{n}] {sym}")
    )
    save_results(results)
    errors = [s for s, d in results.items() if d.get("error")]
    print(f"\nDone. Errors on: {errors if errors else 'none'}")
