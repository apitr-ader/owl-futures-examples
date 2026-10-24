"""29 手續費敏感度：每口單邊手續費 50 → 25 元（其餘不變；滑價 1 / 0.5 檔），不修改 21-28 的預設。

重評：27 pick（vol-scaled 複合動能）、28 s126 best、26 lb20 每到期、26 DM lb252 每季到期、25 DM re-pick（月底/季框架 NEW/NEW05 成本）。
手續費佔總成本比例：同設定以手續費 0 重跑，share = (cost − cost_comm0) / cost。
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load(name, fn):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


m28 = load("mv28", "28_mom_over_vol.py")
m25 = load("cs25", "25_cost_sensitivity.py")
m26, D = m28.m26, m28.D
OUT = Path("results")
DATA, TXF = "data/stock_futures", "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet"

EXP = [("27 pick (vol-scaled composite)", "CMP", dict(skip=0, thr="rf", vs=1, cad=1, buf=0, mkt=1, vt=0.15, scheme="EW", lb=126)),
       ("28 s126 best", "MOV", dict(sc="s126", thr="rf", cad=1, buf=0, mkt=1, vt=0.15, scheme="EW", lb=126, win=252)),
       ("26 lb20 every expiry", "DMS", dict(lb=20, skip=0, cad=1, buf=1, mkt=1, vt=0.15, scheme="EW")),
       ("26 DM lb252 quarterly", "DM", dict(lb=252, skip=21, cad=3, buf=0, mkt=0, vt=0.15, scheme="EW"))]
P25 = dict(lb=252, skip=0, freq=3, buf=1, mkt=0, vt=0.15, scheme="EW", tf=1.0)


def main():
    m28.o22.prepare(DATA, TXF)
    D["txv"] = D["txr"].values; D["txr_s"] = D["txr"]; D["exp"] = m26.expiry_calendar()
    tI = (1 + D["txr"]).cumprod()
    D["txf_cmp"] = np.mean([(tI / tI.shift(L) - 1).values for L in m28.m27.LEGS], 0)
    D["txf_252"] = D["txf_mom"].copy()
    m25.o22.prepare(DATA, TXF)
    D25 = m25.D; D25["txv"] = D25["txr"].values; D25["txr_s"] = D25["txr"]
    for k in ("W", "M", "Q"):
        m25.m24.schedule(k)
    rows = []
    for comm in (50.0, 25.0, 0.0):
        for slip in (1.0, 0.5):
            m28.dm.COMMISSION = comm
            for name, fam, P in EXP:
                m28.market(fam)
                r, _ = m26.evaluate(fam, P, slip, 0.0, "x")
                rows += [dict(strategy=name, comm=comm, slip=slip, **{k: x[k] for k in ("period", "CAGR", "sharpe", "mdd", "cost_pct_yr", "turnover_yr")}) for x in r]
            m25.COMM0 = comm
            r, _ = m25.evaluate("DM", P25, "NEW" if slip == 1 else "NEW05", 0.0, None, "x")
            rows += [dict(strategy="25 DM re-pick (month-end, spread rolls)", comm=comm, slip=slip, **{k: x[k] for k in ("period", "CAGR", "sharpe", "mdd", "cost_pct_yr", "turnover_yr")}) for x in r]
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "comm_detail.csv", index=False)
    p = t.pivot_table(index=["strategy", "slip"], columns=["period", "comm"], values=["CAGR", "sharpe", "cost_pct_yr"])
    s = pd.DataFrame({
        "OOS CAGR 50": p[("CAGR", "OOS", 50.0)], "OOS CAGR 25": p[("CAGR", "OOS", 25.0)],
        "OOS SR 50": p[("sharpe", "OOS", 50.0)], "OOS SR 25": p[("sharpe", "OOS", 25.0)],
        "exSR 50": p[("sharpe", "OOS_ex2526", 50.0)], "exSR 25": p[("sharpe", "OOS_ex2526", 25.0)],
        "cost%/yr 50": p[("cost_pct_yr", "OOS", 50.0)], "cost%/yr 25": p[("cost_pct_yr", "OOS", 25.0)],
        "comm share 50": 1 - p[("cost_pct_yr", "OOS", 0.0)] / p[("cost_pct_yr", "OOS", 50.0)],
        "comm share 25": 1 - p[("cost_pct_yr", "OOS", 0.0)] / p[("cost_pct_yr", "OOS", 25.0)],
        "IS SR 50": p[("sharpe", "IS", 50.0)], "IS SR 25": p[("sharpe", "IS", 25.0)]})
    s.to_csv(OUT / "comm_summary.csv")
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
