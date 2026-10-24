"""41 股票期貨日K 由期交所逐筆資料重建（tools/sfdaily.py → data/stock_futures_daily），重跑 32 的 '27 pick'。

資料：每個商品代碼（含已下市）日盤、排除價差、量 = (B+S)/2；近月連續 = 尚未到期的最近月契約
（到期日取 TX 各月最後交易日，即第 3 個週三含假日順延；隔日換月），收盤 = 最後成交價（RPT 無結算價）；
加法換月調整同 data/stock_futures/daily。標的對應：富邦 products.csv（現存）＋ 已下市代碼以價格比對
（2008-2022 個股收盤，對數價差中位數 < 1%、>= 10 個重疊日、次佳 3 倍以上）。
可選池（同 31/32）：有成交、>= 22 日歷史、排除 ETF 與調整型（'1'/'2' 結尾）；10 檔；每到期換倉；vt 15%；EW。
大盤濾網：(a) 原 TXF 21/63/126 平均報酬 > 0；(b) TXF 收盤 > 250 日均線。
樣本：IS 2010(起資料可用的第一個到期日)-2019 與 2014-2019；OOS 2020-01 ~ 2026-08-18（新資料終點，舊資料同截）。
成本：手續費 50/邊、稅 2e-5、滑價 1 tick（另 0.5）、不計換月成本（同 32）。
對照：39 TXF SMA250（H1 L0）與 TXF B&H。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location("nu32", Path(__file__).with_name("32_four_strategies_new_universe.py"))
m32 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m32)
_s = importlib.util.spec_from_file_location("ma39", Path(__file__).with_name("39_txf_ma_scaling.py"))
m39 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m39)
m31, m24, o22, dm, D = m32.m31, m32.m24, m32.o22, m32.dm, m32.D
OUT = Path("results")
TXF = "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet"
DATA = {"old (Fubon)": "data/stock_futures", "new (TAIFEX ticks)": "data/stock_futures_daily"}
IS1, EX1, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-08-19")
STARTS = {"IS2010": pd.Timestamp("2010-01-01"), "IS2014": pd.Timestamp("2014-01-01")}
P0 = m32.STRATS["27 pick"]
FILTERS = {"cmp (old)": "cmp", "TXF>MA250": "ma250"}


def periods(start):
    return (("IS", start, IS1), ("OOS", IS1, END), ("OOS_ex2526", IS1, EX1))


def setup(data, start):
    _prep = o22.prepare
    o22.prepare = lambda d, t: _prep(data, TXF)
    m31.IS0, m31.END = start, END
    try:
        m32.setup()
    finally:
        o22.prepare = _prep
    D["X"] = [x for x in D["X"] if x >= 130]          # 126 日回看完整（避免負索引）
    tI = pd.Series((1 + D["txr"]).cumprod().values)
    D["sig"]["ma250"] = np.where(tI.rolling(250).mean().notna(), np.sign(tI - tI.rolling(250).mean()), np.nan)
    D["und"] = np.array(D["M"]["A"].columns)          # trace 用代碼（標的去重已在 setup 完成）


def main():
    pd.set_option("display.width", 250)
    new_syms = set(pd.read_csv("data/stock_futures_daily/products.csv").query("underlying_type=='S'").symbol)
    old_syms = set(pd.read_csv("data/stock_futures/products.csv").symbol)
    added = sorted(s for s in new_syms - old_syms if s[-1] not in "12")
    rows, curves, sel = [], {}, []
    for dname, path in DATA.items():
        for sname, start in STARTS.items():
            setup(path, start)
            nuniv = np.mean([len(D["univ"][x]) for x in D["X"] if D["days"][x] >= start])
            for fname, f in FILTERS.items():
                P = {**P0, "msig": f}
                for slip in (1.0, 0.5):
                    m32.COST["slip"] = slip
                    o = m32.simulate(P, start, END, trace=True)
                    key = f"{dname} | {fname} | {sname}"
                    if slip == 1.0:
                        curves[key] = o["r"]
                        rec = pd.DataFrame(o["recs"])
                        if len(rec):
                            rec["data"], rec["filter"], rec["start"] = dname, fname, sname
                            sel.append(rec)
                    for per, lo, hi in periods(start):
                        s = m24.stats(m24.win(o["r"], lo, hi))
                        rows.append(dict(data=dname, filter=fname, start=sname, slip=slip, period=per, CAGR=s["CAGR"],
                                         sharpe=s["sharpe"], MDD=s["mdd"], calmar=s["CAGR"] / abs(s["mdd"]) if s["mdd"] else np.nan,
                                         turnover_yr=o["turn"] / o["yrs"], cost_pct_yr=o["cost"] / o["yrs"] * 100,
                                         gross=o["gross"], avg_universe=nuniv))
                m32.COST["slip"] = 1.0
    # 對照：TXF B&H（同一 TXF 檔）與 39 SMA250 H1 L0
    X = m39.m37.load(); X = X[X.index >= pd.Timestamp("2010-01-01")]
    refs = {"TXF B&H": m39.sim(X, np.ones(len(X)))["r"], "39 TXF SMA250 H1 L0": m39.run(X, m39.state(X, 250), 1, 0)["r"]}
    for k, r in refs.items():
        for sname, start in STARTS.items():
            curves[f"{k} | {sname}"] = m24.win(r, start, END)
            for per, lo, hi in periods(start):
                s = m24.stats(m24.win(r, lo, hi))
                rows.append(dict(data=k, filter="-", start=sname, slip=1.0, period=per, CAGR=s["CAGR"], sharpe=s["sharpe"],
                                 MDD=s["mdd"], calmar=s["CAGR"] / abs(s["mdd"]) if s["mdd"] else np.nan))
    T = pd.DataFrame(rows); T.to_csv(OUT / "rb_summary.csv", index=False)
    t = T[T.slip == 1.0]
    print(t.pivot_table(index=["start", "data", "filter"], columns="period", values=["CAGR", "sharpe", "MDD"], sort=False).round(3).to_string())
    print("\nslip 0.5 OOS:\n", T[(T.slip == 0.5) & (T.period == "OOS")][["data", "filter", "start", "CAGR", "sharpe", "MDD"]].round(3).to_string(index=False))
    print("\nOOS cost/turnover/universe:\n", T[(T.slip == 1.0) & (T.period == "OOS") & T["filter"].ne("-")][["data", "filter", "start", "turnover_yr", "cost_pct_yr", "gross", "avg_universe"]].round(2).to_string(index=False))
    # 存活者偏差：新增（已下市/不在現存清單）代碼被選中的比例
    S = pd.concat(sel)
    S["added"] = S.und.isin(added)
    g = S[S.data.str.startswith("new")].groupby(["filter", "start"]).agg(slots=("added", "size"), added_slots=("added", "sum"),
                                                                       added_names=("und", lambda u: u[u.isin(added)].nunique()))
    g["added_share"] = g.added_slots / g.slots
    print(f"\nadded symbols (stock, not in Fubon products.csv): {len(added)}")
    print(g.to_string())
    top = S[S.data.str.startswith("new") & S.added].groupby("und").size().sort_values(ascending=False).head(15)
    print("most-selected added symbols:", top.to_dict())
    g.to_csv(OUT / "rb_added_selection.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for k, r in curves.items():
        if "IS2010" not in k:
            continue
        e = (1 + r).cumprod()
        st = dict(c="k", lw=2) if "B&H" in k else dict(c="gray", lw=1.5) if "SMA250" in k else \
            dict(lw=1.1, ls="-" if k.startswith("new") else "--")
        lab = k.replace(" | IS2010", "")
        axs[0].plot(e.index, e, label=lab, **st); axs[1].plot(e.index, e / e.cummax() - 1, **st)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=8); axs[0].set_title("27 pick, old (Fubon, survivors) vs rebuilt (TAIFEX ticks, incl. delisted), 1 tick, from 2010")
    axs[1].set_ylabel("drawdown")
    for a in axs:
        a.axvline(IS1, c="gray", lw=0.8); a.axvline(EX1, c="gray", lw=0.6, ls=":")
    plt.tight_layout(); plt.savefig(OUT / "rb_equity_dd.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
