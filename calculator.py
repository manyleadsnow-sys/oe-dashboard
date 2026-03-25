"""
Owner Earnings Dashboard - Core Calculation Engine
Metrics: OE, OE Yield, OE Multiple (EV/OE), OE Growth Rate (10yr CAGR),
         OE-PEG Ratio, EPV (industry-adjusted WACC)
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

PEAK_START_DATE = "2022-10-12"

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


def find_row(df, *candidates):
    if df is None or df.empty:
        return None
    index_str   = [str(i) for i in df.index]
    index_lower = [s.lower() for s in index_str]
    for c in candidates:
        c_lower = c.lower()
        if c in df.index:
            return df.loc[c]
        for i, lbl in enumerate(index_lower):
            if lbl == c_lower:
                return df.iloc[i]
        for i, lbl in enumerate(index_lower):
            if c_lower in lbl:
                return df.iloc[i]
    return None


def row_ttm(df, *candidates):
    row = find_row(df, *candidates)
    if row is None:
        return None
    vals = pd.to_numeric(row, errors='coerce').dropna()
    if len(vals) == 0:
        return None
    return float(vals.iloc[:4].sum())


def compute_oe_ttm(cf_q, bs_q):
    net_income = row_ttm(cf_q,
        "Net Income", "NetIncome",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operations",
        "Net Income Including Noncontrolling Interests")

    da = row_ttm(cf_q,
        "Depreciation & Amortization",
        "Depreciation And Amortization",
        "Reconciled Depreciation",
        "DepreciationAndAmortization",
        "Depreciation Amortization Depletion",
        "Depreciation And Amortization In Income Statement",
        "Depreciation")

    capex = row_ttm(cf_q,
        "Capital Expenditure",
        "Capital Expenditures",
        "CapitalExpenditures",
        "Purchase Of Property Plant And Equipment",
        "Capital Expenditures Reported",
        "Purchases Of Property And Equipment")

    if net_income is None or da is None or capex is None:
        return None

    capex_val = abs(capex)

    def get_wc(col_idx):
        if bs_q is None or bs_q.empty or col_idx >= bs_q.shape[1]:
            return None
        ca_row  = find_row(bs_q, "Current Assets", "Total Current Assets", "TotalCurrentAssets")
        cl_row  = find_row(bs_q, "Current Liabilities", "Total Current Liabilities", "TotalCurrentLiabilities")
        csh_row = find_row(bs_q, "Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents",
                           "Cash Cash Equivalents And Short Term Investments",
                           "Cash And Short Term Investments")
        std_row = find_row(bs_q, "Current Debt", "Short Term Debt", "CurrentDebt",
                           "Current Portion Of Long Term Debt", "Short Long Term Debt")
        if ca_row is None or cl_row is None:
            return None
        def v(row):
            if row is None: return 0.0
            try:
                val = pd.to_numeric(row, errors='coerce').iloc[col_idx]
                return float(val) if not pd.isna(val) else 0.0
            except Exception:
                return 0.0
        return (v(ca_row) - v(csh_row)) - (v(cl_row) - v(std_row))

    wc_curr  = get_wc(0)
    wc_prev  = get_wc(4)
    delta_wc = (wc_curr - wc_prev) if (wc_curr is not None and wc_prev is not None) else 0.0

    return net_income + da - capex_val - delta_wc


def compute_oe_annual_series(t):
    try:
        cf_a = t.cashflow
        if cf_a is None or cf_a.empty:
            return pd.Series(dtype=float)
        results = {}
        for i in range(cf_a.shape[1]):
            cf_col = cf_a.iloc[:, i:i+1]
            ni  = row_ttm(cf_col, "Net Income", "NetIncome",
                          "Net Income Common Stockholders",
                          "Net Income From Continuing Operations")
            da  = row_ttm(cf_col, "Depreciation & Amortization",
                          "Depreciation And Amortization",
                          "Reconciled Depreciation", "DepreciationAndAmortization",
                          "Depreciation Amortization Depletion", "Depreciation")
            cap = row_ttm(cf_col, "Capital Expenditure", "Capital Expenditures",
                          "CapitalExpenditures",
                          "Purchase Of Property Plant And Equipment",
                          "Purchases Of Property And Equipment")
            if ni is None or da is None or cap is None:
                continue
            try:
                yr = pd.to_datetime(cf_a.columns[i]).year
                results[yr] = ni + da - abs(cap)
            except Exception:
                pass
        return pd.Series(results).sort_index()
    except Exception:
        return pd.Series(dtype=float)


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


def compute_epv(t, sector, shares):
    try:
        inc = t.financials
        if inc is None or inc.empty:
            return None
        ebit_row = find_row(inc, "EBIT", "Ebit", "Operating Income",
                            "Operating Income Loss", "Operating Profit")
        if ebit_row is None:
            return None
        ebit_vals = pd.to_numeric(ebit_row, errors='coerce').dropna().values
        if len(ebit_vals) == 0:
            return None
        avg_ebit = float(np.mean(ebit_vals))

        tax_row = find_row(inc, "Tax Rate For Calcs", "Effective Tax Rate",
                           "Tax Provision", "Income Tax Expense")
        tax_rate = 0.21
        if tax_row is not None:
            tax_vals = pd.to_numeric(tax_row, errors='coerce').dropna()
            if tax_vals.abs().mean() > 1:
                pretax_row = find_row(inc, "Pretax Income", "Income Before Tax",
                                      "Earnings Before Income Taxes")
                if pretax_row is not None:
                    pretax_vals = pd.to_numeric(pretax_row, errors='coerce').dropna()
                    valid = pretax_vals[pretax_vals.abs() > 0]
                    if len(valid) > 0:
                        rates = tax_vals.reindex(valid.index) / valid
                        r = float(rates.dropna().mean())
                        if 0 < r < 0.6:
                            tax_rate = r
            else:
                r = float(tax_vals.mean())
                if 0 < r < 0.6:
                    tax_rate = r

        wacc = get_wacc(sector)
        epv_total = avg_ebit * (1 - tax_rate) / wacc
        if shares and shares > 0:
            return epv_total / shares
        return None
    except Exception:
        return None


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


def pct_diff(a, b):
    if a is None or b is None or b == 0:
        return None
    return round((a - b) / abs(b) * 100, 2)


def metrics_at_price(oe_ttm, ev, mc, oe_growth, epv_per_share):
    oe_yield    = (oe_ttm / mc * 100)             if (mc and mc != 0 and oe_ttm) else None
    oe_multiple = (ev / oe_ttm)                   if (oe_ttm and oe_ttm != 0 and ev) else None
    oe_peg      = (oe_multiple / (oe_growth * 100)) if (oe_multiple and oe_growth and oe_growth != 0) else None
    return {
        "oe":          round(oe_ttm / 1e9, 4)      if oe_ttm          is not None else None,
        "oe_yield":    round(oe_yield, 4)           if oe_yield        is not None else None,
        "oe_multiple": round(oe_multiple, 4)        if oe_multiple     is not None else None,
        "oe_growth":   round(oe_growth * 100, 4)   if oe_growth       is not None else None,
        "oe_peg":      round(oe_peg, 4)             if oe_peg          is not None else None,
        "epv":         round(epv_per_share, 4)      if epv_per_share   is not None else None,
    }


def diff_block(curr, peak):
    return {k: pct_diff(curr.get(k), peak.get(k))
            for k in ("oe","oe_yield","oe_multiple","oe_growth","oe_peg","epv")}


def fetch_ticker_data(symbol):
    result = {"ticker": symbol, "error": None, "sector": None,
              "current": {}, "peak_since_oct2022": {},
              "bear_markets": [], "last_updated": datetime.now().isoformat()}
    try:
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

        cf_q = t.quarterly_cashflow
        bs_q = t.quarterly_balance_sheet

        oe_ttm        = compute_oe_ttm(cf_q, bs_q)
        oe_series     = compute_oe_annual_series(t)
        oe_growth     = compute_oe_growth_10yr(oe_series)
        epv_per_share = compute_epv(t, sector, shares)

        def ev_mc(price):
            if not shares: return None, None
            mc = price * shares
            return mc + net_debt, mc

        ev_c, mc_c = ev_mc(current_price)
        if oe_ttm and mc_c:
            m = metrics_at_price(oe_ttm, ev_c, mc_c, oe_growth, epv_per_share)
            m["price"] = round(current_price, 2)
            result["current"] = m

        ev_p, mc_p = ev_mc(peak_price)
        if oe_ttm and mc_p:
            m = metrics_at_price(oe_ttm, ev_p, mc_p, oe_growth, epv_per_share)
            m["price"] = round(peak_price, 2)
            m["date"]  = str(peak_date.date())
            result["peak_since_oct2022"] = m

        if result["current"] and result["peak_since_oct2022"]:
            result["vs_peak_diff"] = diff_block(result["current"], result["peak_since_oct2022"])

        for bear in detect_bear_markets(hist):
            bp = bear["trough_price"]
            ev_b, mc_b = ev_mc(bp)
            if not (oe_ttm and mc_b): continue
            mb = metrics_at_price(oe_ttm, ev_b, mc_b, oe_growth, epv_per_share)
            mb["price"]       = round(bp, 2)
            mb["peak_price"]  = round(bear["peak_price"], 2)
            mb["peak_date"]   = str(bear["peak_date"].date()) if hasattr(bear["peak_date"], "date") else str(bear["peak_date"])
            mb["trough_date"] = str(bear["trough_date"].date()) if hasattr(bear["trough_date"], "date") else str(bear["trough_date"])
            mb["drawdown_pct"]= round(bear["drawdown_pct"], 2)
            ev_bp, mc_bp = ev_mc(bear["peak_price"])
            mp = metrics_at_price(oe_ttm, ev_bp, mc_bp, oe_growth, epv_per_share)
            mb["vs_bear_peak_diff"] = diff_block(mb, mp)
            result["bear_markets"].append(mb)

    except Exception as e:
        result["error"] = str(e)
    return result


def run_full_calculation(tickers=None, progress_callback=None):
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
    print(f"Running OE calculations for {len(TICKERS)} tickers...")
    results = run_full_calculation(
        progress_callback=lambda sym, i, n: print(f"  [{i}/{n}] {sym}")
    )
    save_results(results)
    errors = [s for s, d in results.items() if d.get("error")]
    print(f"\nDone. Errors: {errors if errors else 'none'}")
