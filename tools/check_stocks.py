#!/usr/bin/env python
"""Check data/stocks: coverage, gaps, adjusted-vs-raw sanity, spot checks (2330, 2317, a capital reduction).
Usage: python check_stocks.py [--codes 2330,2317]   (no network)"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/workspace/fubon-threads/data/stocks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="2330,2317,2412,2327")
    a = ap.parse_args()
    st = json.loads((OUT / "_state.json").read_text()) if (OUT / "_state.json").exists() else {}
    print("state:", st)
    for m in ("twse", "tpex"):
        fs = sorted((OUT / "cache" / m).glob("*.json.gz"))
        print(f"cache {m}: {len(fs)} weekday files, {fs[0].name[:8] if fs else '-'} .. {fs[-1].name[:8] if fs else '-'}")
    if not (OUT / "universe.csv").exists():
        print("universe.csv not built yet (daily download still running?)"); return
    u = pd.read_csv(OUT / "universe.csv", dtype={"code": str}, parse_dates=["first_date", "last_date"])
    print("\nuniverse:", len(u), u.groupby(["market", "status"]).size().to_dict())
    print("delisted with official delist date:", int((u.status.eq("delisted") & ~u.delist_src.fillna("").str.startswith("inferred")).sum()),
          " inferred:", int(u.delist_src.fillna("").str.startswith("inferred").sum()))
    print("date range:", u.first_date.min().date(), "..", u.last_date.max().date())
    ca = pd.read_parquet(OUT / "corporate_actions.parquet")
    print("\ncorporate actions by type/market:\n", ca.groupby(["type", "market"]).size().to_string())
    print("by source:\n", ca.groupby("source").size().to_string())
    ap_ = ca[ca.applied]
    print(f"\nevents applied {len(ap_)} of {len(ca)} (rest: ex-date outside a stock's trading history); "
          f"official prev_close vs data prev_close mismatch >1%: {int(ap_.mismatch.sum())}")
    print("  largest mismatches:\n", ap_.assign(d=(ap_.prev_close_data / ap_.prev_close - 1)).loc[lambda x: x.mismatch]
          .sort_values("d", key=abs, ascending=False).head(6)[["code", "ex_date", "type", "prev_close", "prev_close_data", "adj_factor"]].to_string(index=False))
    # market calendar gaps
    cal = {}
    for m in ("twse", "tpex"):
        cal[m] = sorted(p.name[:8] for p in (OUT / "cache" / m).glob("*.json.gz") if p.stat().st_size > 30)
    only = set(cal["twse"]) ^ set(cal["tpex"])
    print("trading days twse/tpex:", len(cal["twse"]), len(cal["tpex"]), " days in only one market:", sorted(only)[:10])
    # sanity: adjusted daily returns vs price limits
    bad, big, n = [], 0, 0
    for p in sorted((OUT / "adj").glob("*.parquet")):
        d = pd.read_parquet(p)
        r = d.ret.dropna()
        lim = np.where(r.index < "2015-06-01", 0.07, 0.10) + 0.012
        x = r[(r.abs() > lim)]
        x = x.iloc[1:] if len(x) and x.index[0] == r.index[0] else x
        n += len(r); big += len(x)
        if len(x):
            bad.append((p.stem, len(x), x.abs().max(), x.abs().idxmax().date()))
        if (d.factor <= 0).any() or d.factor.isna().any():
            print("BAD factor", p.stem)
    print(f"\nadj returns beyond price limit: {big} of {n} ({big / max(n, 1):.4%}); stocks affected {len(bad)}")
    for b in sorted(bad, key=lambda z: -z[2])[:12]:
        print("   ", b)
    for c in a.codes.split(","):
        if not (OUT / "raw" / f"{c}.parquet").exists():
            print(c, "missing"); continue
        raw, adj = pd.read_parquet(OUT / "raw" / f"{c}.parquet"), pd.read_parquet(OUT / "adj" / f"{c}.parquet")
        e = ap_[ap_.code == c].rename(columns={"ex_date": "date", "type": "action", "adj_factor": "factor", "prev_close": "prev_close_official"})
        yrs = (raw.index[-1] - raw.index[0]).days / 365.25
        pr = (raw.close.iloc[-1] / raw.close.iloc[0]) ** (1 / yrs) - 1
        tr = (adj.close.iloc[-1] / adj.close.iloc[0]) ** (1 / yrs) - 1
        print(f"\n{c} {raw.name.iloc[-1]} {raw.index[0].date()}..{raw.index[-1].date()} rows {len(raw)}  events {len(e)} "
              f"price CAGR {pr:.2%}  total-return CAGR {tr:.2%}  factor at start {adj.factor.iloc[0]:.4f}")
        print(e.tail(4)[["date", "action", "factor", "prev_close_data", "prev_close_official"]].to_string(index=False))


if __name__ == "__main__":
    main()
