"""73a 由 TAIFEX RPT 逐筆檔建 TXO 日盤 1 分鐘 ATM straddle 序列（Reddit #11 用）。輸出在 repo 外（授權）。

每個交易日：最近到期序列（當日到期則用下一檔），日盤 08:45–13:45，每個 (strike,cp) 每分鐘最後成交價並在日內前向填補；
straddle(K) = C+P（兩邊都已有成交才算）；每分鐘取 TXF 開盤附近 ±10 檔內 straddle 最小者（同 Reddit 作法）。
每 2 個交易日取 1 日（降低運算）；輸出 data/txo_1m_straddle/YYYY-MM-DD.parquet：minute, K, straddle, F(min K proxy)。
"""
import sys, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.opt_convert import index_sources, read_rpt, parse
from tools.optdaily import expiries

ZIP = "/workspace/fubon-threads/data/options_daily/zip"; OUT = Path("/workspace/fubon-threads/data/txo_1m_straddle"); OUT.mkdir(exist_ok=True)


def one(args):
    d, refs = args
    o = OUT / f"{d}.parquet"
    if o.exists(): return d, "skip"
    try:
        b, _, _ = read_rpt(refs); df, *_ = parse(b, d)
        if df is None: return d, "none"
        df = df[~df.night]
        E = expiries(); t = pd.Timestamp(d); e = E[(E.first_date <= t) & (E.expiry_date > t)].sort_values("expiry_date")
        ex = e.expiry.iloc[0]
        x = df[df.expiry.str.strip() == ex]
        x = x[(x.sec >= 8 * 3600 + 45 * 60) & (x.sec < 13 * 3600 + 45 * 60)]
        x["m"] = ((x.sec - (8 * 3600 + 45 * 60)) // 60).astype(int)
        last = x.groupby(["m", "strike", "cp"]).price.last().unstack(["strike", "cp"]).reindex(range(300)).ffill()
        ks = sorted(set(k for k, c in last.columns))
        S = pd.DataFrame({k: last[(k, "C")] + last[(k, "P")] for k in ks if (k, "C") in last and (k, "P") in last})
        if S.empty: return d, "empty"
        k0 = S.iloc[:30].min().idxmin() if S.iloc[:30].notna().any().any() else S.min().idxmin()
        j = ks.index(k0); S = S[[k for k in ks[max(0, j - 10): j + 11] if k in S]]
        res = pd.DataFrame({"minute": range(300), "K": S.idxmin(axis=1).values, "straddle": S.min(axis=1).values, "expiry": ex})
        res.to_parquet(o); return d, "ok"
    except Exception as e:
        return d, f"err {e}"


if __name__ == "__main__":
    src = index_sources(ZIP); days = sorted(d for d in src if d >= "2017-01-01")[::2]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for i, (d, s) in enumerate(ex.map(one, [(d, src[d]) for d in days], chunksize=4)):
            if i % 50 == 0 or s.startswith("err"): print(i, d, s, flush=True)
    print("done")
