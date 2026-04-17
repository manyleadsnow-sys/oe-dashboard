"""
Standalone metric checker — AAPL and A (Agilent)
Sources: SEC EDGAR (financials) + Tiingo (prices)
Run:  python check_metrics.py
"""
import requests, json, numpy as np, time
from datetime import datetime, timedelta

TIINGO_TOKEN   = "2dfc1e2c3bf907f438a5edfaa94a4a1ee6cd0539"
TIINGO_HEADERS = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_TOKEN}"}
EDGAR_HEADERS  = {"User-Agent": "Gustavo Gonzalez gusqweenglish@gmail.com", "Accept-Encoding": "gzip, deflate"}

CRISES = [
    {"name": "April 2025 Liberation Day", "start": "2025-03-01", "end": "2025-06-30"},
    {"name": "2022 Bear Market",           "start": "2022-01-01", "end": "2022-10-14"},
    {"name": "2020 COVID Crash",           "start": "2020-02-15", "end": "2020-03-23"},
    {"name": "2015-2016 Selloff",          "start": "2015-06-01", "end": "2016-02-11"},
    {"name": "2008 GFC",                   "start": "2007-10-01", "end": "2009-03-09"},
]

# ── EDGAR ──────────────────────────────────────────────────────────────────────

def find_concept(facts, *concepts, unit="USD"):
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for c in concepts:
        if c not in us_gaap:
            continue
        entries = us_gaap[c].get("units", {}).get(unit, [])
        valid = [e for e in entries if "end" in e and "val" in e]
        if valid:
            valid.sort(key=lambda x: (x["end"], x.get("filed", "")), reverse=True)
            return valid
    return []

def dedup(entries, forms=("10-Q","10-K")):
    filtered = [e for e in entries if e.get("form","") in forms]
    seen = {}
    for e in filtered:
        k = e["end"]
        if k not in seen or e.get("filed","") > seen[k].get("filed",""):
            seen[k] = e
    return sorted(seen.values(), key=lambda x: x["end"], reverse=True)

def get_ttm(facts, *concepts, unit="USD"):
    entries = find_concept(facts, *concepts, unit=unit)
    if not entries:
        return None
    k_ent = sorted([e for e in entries if e.get("form")=="10-K" and "start" in e], key=lambda x: x["end"], reverse=True)
    if not k_ent:
        return None
    lk = k_ent[0]
    q_ent = [e for e in entries if e.get("form")=="10-Q" and "start" in e and e["end"] > lk["end"]]
    if not q_ent:
        return lk["val"]
    q_ent.sort(key=lambda x: x["end"], reverse=True)
    lq_end = q_ent[0]["end"]
    lq = max((e for e in q_ent if e["end"]==lq_end),
             key=lambda x: (datetime.strptime(x["end"],"%Y-%m-%d")-datetime.strptime(x["start"],"%Y-%m-%d")).days)
    pe = datetime.strptime(lq_end,"%Y-%m-%d") - timedelta(days=365)
    ps = datetime.strptime(lq["start"],"%Y-%m-%d") - timedelta(days=365)
    prior = 0
    for e in entries:
        if e.get("form")=="10-Q" and "start" in e:
            ed = datetime.strptime(e["end"],"%Y-%m-%d")
            sd = datetime.strptime(e["start"],"%Y-%m-%d")
            if abs((ed-pe).days)<=25 and abs((sd-ps).days)<=25:
                prior = e["val"]; break
    return lk["val"] + lq["val"] - prior

def annual_vals(facts, *concepts, n=15, unit="USD"):
    entries = find_concept(facts, *concepts, unit=unit)
    rows = []
    for e in entries:
        if e.get("form")=="10-K" and "start" in e:
            try:
                sd = datetime.strptime(e["start"],"%Y-%m-%d")
                ed = datetime.strptime(e["end"],"%Y-%m-%d")
                if 350 <= (ed-sd).days <= 380:
                    rows.append(e)
            except:
                pass
    rows = dedup(rows, ("10-K",))
    return [(e["end"], e["val"]) for e in rows[:n]]

def fetch_edgar(sym):
    # CIK lookup
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=EDGAR_HEADERS, timeout=15)
    cik = None
    for k, v in r.json().items():
        if v["ticker"].upper() == sym.upper():
            cik = str(v["cik_str"]).zfill(10)
            break
    print(f"  CIK: {cik}")
    time.sleep(0.4)
    r2 = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=EDGAR_HEADERS, timeout=90)
    r2.raise_for_status()
    facts = r2.json()

    ni_ttm    = get_ttm(facts, "NetIncomeLoss","NetIncomeLossAvailableToCommonStockholdersBasic","ProfitLoss")
    da_ttm    = get_ttm(facts, "DepreciationDepletionAndAmortization","DepreciationAndAmortization","Depreciation")
    cx_ttm    = get_ttm(facts, "PaymentsToAcquirePropertyPlantAndEquipment","PaymentsForCapitalImprovements","PaymentsToAcquireProductiveAssets")
    oe_ttm    = (ni_ttm or 0) + (da_ttm or 0) - abs(cx_ttm or 0) if ni_ttm is not None else None

    ni_a  = annual_vals(facts, "NetIncomeLoss","NetIncomeLossAvailableToCommonStockholdersBasic","ProfitLoss")
    da_a  = annual_vals(facts, "DepreciationDepletionAndAmortization","DepreciationAndAmortization","Depreciation")
    cx_a  = annual_vals(facts, "PaymentsToAcquirePropertyPlantAndEquipment","PaymentsForCapitalImprovements","PaymentsToAcquireProductiveAssets")
    ni_d  = {e[0]:e[1] for e in ni_a}
    da_d  = {e[0]:e[1] for e in da_a}
    cx_d  = {e[0]:e[1] for e in cx_a}
    oe_a  = {d: ni_d[d] + da_d.get(d,0) - abs(cx_d.get(d,0)) for d in sorted(ni_d)}

    sh_a  = annual_vals(facts, "WeightedAverageNumberOfSharesOutstandingBasic","CommonStockSharesOutstanding", unit="shares")
    sh_d  = {e[0]:e[1] for e in sh_a}
    sh_ttm = sh_a[0][1] if sh_a else None

    return {"entity": facts.get("entityName",""), "oe_ttm": oe_ttm, "oe_a": oe_a, "sh_ttm": sh_ttm, "sh_d": sh_d}

# ── Tiingo ─────────────────────────────────────────────────────────────────────

def fetch_prices(sym):
    t = sym.upper().replace(".","-")
    url = f"https://api.tiingo.com/tiingo/daily/{t}/prices"
    params = {"startDate":"2004-01-01","endDate":datetime.now().strftime("%Y-%m-%d"),"resampleFreq":"daily","token":TIINGO_TOKEN}
    r = requests.get(url, params=params, headers=TIINGO_HEADERS, timeout=60)
    r.raise_for_status()
    prices = {}
    for row in r.json():
        d = row["date"][:10]
        v = row.get("adjClose") or row.get("close")
        if v: prices[d] = float(v)
    return prices

# ── Computation helpers ────────────────────────────────────────────────────────

def oe_cagr(oe_a, end_str):
    if end_str not in oe_a:
        return None
    ev = oe_a[end_str]
    ed = datetime.strptime(end_str,"%Y-%m-%d")
    dates = sorted(oe_a.keys())
    for span in [5,4,3,2]:
        ts = ed - timedelta(days=span*365+30)
        te = ed - timedelta(days=span*365-30)
        cands = [d for d in dates if ts.strftime("%Y-%m-%d") <= d <= te.strftime("%Y-%m-%d")]
        if not cands: continue
        sd = min(cands); sv = oe_a[sd]
        yrs = (ed - datetime.strptime(sd,"%Y-%m-%d")).days / 365.25
        if yrs < 1.5: continue
        if sv > 0 and ev > 0:
            return (ev/sv)**(1/yrs) - 1
        return None
    return None

def latest_before(d_dict, ref_str, fallback=None):
    cands = [(d,v) for d,v in d_dict.items() if d <= ref_str]
    if not cands: return None, fallback
    best = max(cands, key=lambda x: x[0])
    return best[0], best[1]

def peak_before(prices, ref_str):
    ref = datetime.strptime(ref_str,"%Y-%m-%d")
    s = (ref-timedelta(days=364)).strftime("%Y-%m-%d")
    e = (ref-timedelta(days=1)).strftime("%Y-%m-%d")
    w = {d:v for d,v in prices.items() if s<=d<=e}
    if not w: return None, None
    pk = max(w, key=w.get)
    return pk, w[pk]

def trough_in(prices, s, e):
    w = {d:v for d,v in prices.items() if s<=d<=e}
    if not w: return None, None
    tr = min(w, key=w.get)
    return tr, w[tr]

def metrics(oe, shares, price, oe_a, fye):
    if not all([oe, shares, price]):
        return {}
    mc   = price * shares
    oeps = oe / shares
    mult = mc/oe if oe > 0 else None
    yld  = oe/mc*100 if oe > 0 else None
    gr   = oe_cagr(oe_a, fye) if fye else None
    peg  = mult/(gr*100) if (mult and gr and gr>0) else None
    return {
        "Price":       f"${price:.2f}",
        "OE ($B)":     f"{oe/1e9:.3f}",
        "OE/share":    f"{oeps:.2f}",
        "OE multiple": f"{mult:.2f}x" if mult else "n/a",
        "OE yield":    f"{yld:.3f}%" if yld else "n/a",
        "OE CAGR":     f"{gr*100:.2f}%" if gr else "n/a",
        "OE-PEG":      f"{peg:.3f}" if peg else "n/a",
    }

def z_scores(oe_ttm, oe_a, today):
    now = datetime.strptime(today,"%Y-%m-%d")
    c5  = (now-timedelta(days=5*365)).strftime("%Y-%m-%d")
    c10 = (now-timedelta(days=10*365)).strftime("%Y-%m-%d")
    def z(vals):
        a = np.array(vals,float); s = a.std()
        return round((oe_ttm-a.mean())/s,3) if s>0 else None
    v5  = [v for d,v in oe_a.items() if d>=c5]
    v10 = [v for d,v in oe_a.items() if d>=c10]
    return (z(v5) if len(v5)>=2 else None), (z(v10) if len(v10)>=2 else None)

def print_metrics(label, m):
    print(f"  {label}")
    for k,v in m.items():
        print(f"    {k:<14}: {v}")

# ── Main ───────────────────────────────────────────────────────────────────────

for SYM in ["AAPL", "A"]:
    print(f"\n{'='*65}")
    print(f"  {SYM}")
    print(f"{'='*65}")

    fin    = fetch_edgar(SYM)
    time.sleep(0.5)
    prices = fetch_prices(SYM)

    today  = max(prices.keys())
    print(f"  Entity : {fin['entity']}")
    print(f"  OE TTM : ${fin['oe_ttm']/1e9:.3f}B")
    print(f"  Shares : {fin['sh_ttm']:,.0f}")
    print(f"  Price  : ${prices[today]:.2f}  ({today}, Tiingo)")

    z5, z10 = z_scores(fin["oe_ttm"], fin["oe_a"], today)
    print(f"\n  5Y OE Z-score  : {z5}")
    print(f"  10Y OE Z-score : {z10}")

    # Current
    fye, _ = latest_before(fin["oe_a"], today)
    sh_c   = latest_before(fin["sh_d"], today, fin["sh_ttm"])[1]
    print("\n── CURRENT ──────────────────────────────────────────────────")
    print_metrics("Today", metrics(fin["oe_ttm"], fin["sh_ttm"], prices[today], fin["oe_a"], fye))

    # 52w peak
    pk_d, pk_p = peak_before(prices, today)
    if pk_d:
        fye_pk, oe_pk = latest_before(fin["oe_a"], pk_d)
        sh_pk = latest_before(fin["sh_d"], pk_d, fin["sh_ttm"])[1]
        print(f"\n── 52-WEEK PEAK  ({pk_d}) ──────────────────────────────────")
        print_metrics(f"Peak ${pk_p:.2f}", metrics(oe_pk, sh_pk, pk_p, fin["oe_a"], fye_pk))

    # Crises
    first = min(prices.keys())
    for c in CRISES:
        cs = c["start"]; ce = min(c["end"], today)
        print(f"\n── {c['name'].upper()} ──────────────────────────────────────")
        if cs < first:
            print("  No price history available")
            continue
        pre_d, pre_p = peak_before(prices, cs)
        tr_d,  tr_p  = trough_in(prices, cs, ce)
        if not pre_d or not tr_d:
            print("  Insufficient data"); continue
        dd = (tr_p - pre_p)/pre_p*100
        print(f"  Pre-crisis peak : {pre_d}  ${pre_p:.2f}")
        print(f"  Trough          : {tr_d}  ${tr_p:.2f}  ({dd:.1f}% drawdown)")

        fye_p, oe_p = latest_before(fin["oe_a"], pre_d)
        sh_p = latest_before(fin["sh_d"], pre_d, fin["sh_ttm"])[1]
        print_metrics("Pre-crisis peak metrics", metrics(oe_p, sh_p, pre_p, fin["oe_a"], fye_p))

        fye_t, oe_t = latest_before(fin["oe_a"], tr_d)
        sh_t = latest_before(fin["sh_d"], tr_d, fin["sh_ttm"])[1]
        print_metrics("Trough metrics         ", metrics(oe_t, sh_t, tr_p, fin["oe_a"], fye_t))

print("\nDone.")
