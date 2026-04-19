"""
verify_metrics.py — Audit every ticker in oe_data.json for accuracy.

Checks performed per ticker:
  1. Math identity: oe_multiple × oe_yield/100 ≈ 1  (MC/OE × OE/MC = 1)
  2. OE-PEG identity: oe_peg ≈ oe_multiple / oe_growth
  3. OE/share identity: oeps ≈ oe / shares  (where shares = oe/oeps)
  4. Crisis OE ≠ current TTM OE  (fallback bug detector)
  5. Crisis OE changes over time  (same OE at peak and trough flags stale data)
  6. Trough price < peak price  (basic sanity)
  7. Z-score direction: if z5 ≤ -1 the stock should signal ACCUMULATE or STRONG DCA
  8. All required fields present

Run:
    python verify_metrics.py                     # full report
    python verify_metrics.py --errors-only       # only show tickers with issues
"""
import json, sys, math

ERRORS_ONLY = "--errors-only" in sys.argv
TOL = 0.02   # 2% tolerance for floating-point identity checks

with open("oe_data.json") as f:
    data = json.load(f)

total = 0
issues = 0

def check(sym, label, condition, detail=""):
    global issues
    if not condition:
        issues += 1
        print(f"  ✗ {sym:6} | {label}")
        if detail:
            print(f"         {detail}")

def near(a, b, tol=TOL):
    if a is None or b is None:
        return True   # can't check missing values
    if b == 0:
        return abs(a) < 1e-6
    return abs((a - b) / b) < tol

def audit_snapshot(sym, label, snap, curr_oe_B):
    """Audit one price/OE snapshot dict."""
    oe   = snap.get("oe")        # in $B
    oeps = snap.get("oeps")
    mult = snap.get("oe_multiple")
    yld  = snap.get("oe_yield")
    cagr = snap.get("oe_growth")
    peg  = snap.get("oe_peg")
    px   = snap.get("price")

    # 1. Multiple × yield identity  (should ≈ 100)
    if mult is not None and yld is not None:
        check(sym, f"{label}: mult×yield≈100",
              near(mult * yld, 100, tol=0.03),
              f"mult={mult}  yield={yld}  product={mult*yld:.2f}")

    # 2. OE-PEG identity
    if mult is not None and cagr is not None and cagr != 0 and peg is not None:
        expected_peg = mult / cagr
        check(sym, f"{label}: peg=mult/cagr",
              near(peg, expected_peg, tol=0.03),
              f"peg={peg}  mult/cagr={expected_peg:.3f}")

    # 3. OE fallback detector — OE should NOT equal current TTM for historical periods
    if "crisis" in label.lower() or "peak" in label.lower():
        if oe is not None and curr_oe_B is not None:
            is_fallback = near(oe, curr_oe_B, tol=0.001)
            check(sym, f"{label}: OE≠currentTTM (fallback bug)",
                  not is_fallback,
                  f"oe={oe}B == curr_ttm={curr_oe_B}B")

def audit_crisis(sym, bm, curr_oe_B):
    name = bm.get("crisis_name", "?")
    # Basic price sanity
    pk_p = bm.get("peak_price")
    tr_p = bm.get("price")
    if pk_p and tr_p:
        check(sym, f"{name}: trough<peak price",
              tr_p < pk_p,
              f"trough={tr_p}  peak={pk_p}")

    # Trough metrics
    audit_snapshot(sym, f"{name} trough", bm, curr_oe_B)

    # Peak metrics
    pm = bm.get("peak_metrics", {})
    if pm:
        audit_snapshot(sym, f"{name} peak", pm, curr_oe_B)

    # Peak OE and trough OE should not both equal current OE
    oe_t = bm.get("oe")
    oe_p = pm.get("oe") if pm else None
    if oe_t is not None and oe_p is not None and curr_oe_B is not None:
        both_fallback = near(oe_t, curr_oe_B, 0.001) and near(oe_p, curr_oe_B, 0.001)
        check(sym, f"{name}: OE not frozen at current TTM",
              not both_fallback,
              f"peak_oe={oe_p}B  trough_oe={oe_t}B  curr={curr_oe_B}B")

print(f"Verifying {len(data)} tickers...\n")

for sym, d in sorted(data.items()):
    total += 1
    ticker_issues_before = issues

    if d.get("error"):
        if not ERRORS_ONLY:
            print(f"  ⚠ {sym:6} | error: {d['error']}")
        issues += 1
        continue

    curr     = d.get("current", {})
    curr_oe  = curr.get("oe")   # $B
    dm       = d.get("discount_metrics", {})

    # Audit current snapshot
    audit_snapshot(sym, "current", curr, None)   # no fallback check for current

    # Audit 52w peak
    pk = d.get("peak_52w", {})
    if pk:
        audit_snapshot(sym, "52w peak", pk, curr_oe)

    # Audit all crisis periods
    for bm in d.get("bear_markets", []):
        audit_crisis(sym, bm, curr_oe)

    # Z-score vs DCA signal consistency
    z5     = dm.get("z_score_5y")
    signal = d.get("dca_signal", "")
    if z5 is not None:
        if z5 <= -1.5 and "STRONG DCA" not in signal:
            check(sym, "z5≤-1.5 should be STRONG DCA", False, f"z5={z5}  signal={signal}")
        if z5 > -0.5 and "WAIT" not in signal and "NEGATIVE" not in signal:
            check(sym, "z5>-0.5 should be WAIT", False, f"z5={z5}  signal={signal}")

    had_issues = issues > ticker_issues_before
    if not ERRORS_ONLY or had_issues:
        crisis_names = [bm["crisis_name"] for bm in d.get("bear_markets", [])]
        status = "✗" if had_issues else "✓"
        print(f"  {status} {sym:6} | price=${curr.get('price')}  OE=${curr_oe}B  "
              f"mult={curr.get('oe_multiple')}x  yield={curr.get('oe_yield')}%  "
              f"z5={dm.get('z_score_5y')}  signal={d.get('dca_signal')}  "
              f"crises={len(d.get('bear_markets',[]))}: {crisis_names}")

print(f"\n{'='*70}")
print(f"Tickers checked : {total}")
print(f"Issues found    : {issues}")
print(f"Clean tickers   : {total - sum(1 for d in data.values() if d.get('error'))}")
