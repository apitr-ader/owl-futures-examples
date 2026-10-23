"""23 個股期貨 Dual Momentum：短回看（1~4 週 = 5/10/15/20 交易日）

沿用 21/22（不修改）：同資料、可選池規則（>=252 日歷史、60 日量中位數 >= 50 口、排除結尾 1、換月跳空 >15%、
同標的標準優先、排除 ETF）、成本（每口每邊 50 元＋稅 2/100000＋1 跳動點，持有部位換月另扣來回）、
訊號用收盤、次一交易日開盤成交、只用當時已知資料，5 檔、單檔上限 40%。
模擬器用 22 的向量化分數口數版本（與 21 frac 模式一致）。

網格（小）：lb 5/10/15/20 × 換倉 每週/每月 × skip 0/1 日 × 大盤濾網 關/開 × 波動目標 無/15% × 權重 EW/IV/MV(LW,120d)/ERC
          = 256 組；門檻 0。
選參：每個 lb 內，以 IS Sharpe 的 plateau（自己＋只差一個參數的鄰居平均，鄰居含相鄰 lb）挑一組。
DSR 以全部 256 組試驗數計算。
注意：products.csv 只有現存契約 → 倖存者偏差（偏樂觀）。
"""
import argparse
import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("dmo22", Path(__file__).with_name("22_dual_momentum_opt.py"))
o22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(o22)
dm = o22.dm
D = o22.D
IS0, IS1, EX1, END = o22.IS0, o22.IS1, o22.EX1, o22.END

G = dict(lb=[5, 10, 15, 20], reb=["W", "M"], skip=[0, 1], mkt=[0, 1], vt=[0.0, 0.15], scheme=["EW", "IV", "MV", "ERC"])
PERIODS = (("IS", IS0, IS1), ("OOS", IS1, END), ("OOS_ex2526", IS1, EX1))


def schedule(kind, lo, hi):
    days = D["days"]
    s = pd.Series(np.arange(len(days)), days)
    key = [days.isocalendar().year.values, days.isocalendar().week.values] if kind == "W" else [days.year, days.month]
    last = s.groupby(key).last().sort_values().values
    return [int(i) for i in last if lo - pd.Timedelta(days=10) <= days[i] < hi]


def run(P, lo=IS0, hi=END):
    D["mes"] = schedule(P["reb"], lo, hi)
    Q = dict(lb=P["lb"], skip=P["skip"], ntop=5, freq=1, buf=0, minvol=50, minpx=0, thr="0",
             mkt=P["mkt"], vt=P["vt"], scheme=P["scheme"])
    o = o22.run(Q, keep_path=True)
    r = o["r"]
    r = r[(r.index >= lo) & (r.index < hi)]
    return r, o


def stats(r):
    e = (1 + r).cumprod(); yrs = len(r) / 252
    return dict(CAGR=e.iloc[-1] ** (1 / yrs) - 1, vol=r.std() * math.sqrt(252),
                sharpe=r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0.0,
                mdd=(e / e.cummax() - 1).min(), n=len(r), skew=float(r.skew()), kurt=float(r.kurt() + 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stock_futures")
    ap.add_argument("--txf", default="data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    o22.prepare(a.data, a.txf)
    rows, paths = [], {}
    for v in itertools.product(*G.values()):
        P = dict(zip(G, v))
        r, _ = run(P)
        row = dict(P)
        for per, lo, hi in PERIODS:
            row[per + "_sharpe"] = stats(r[(r.index >= lo) & (r.index < hi)])["sharpe"]
        rows.append(row); paths[v] = r
        print(v, round(row["IS_sharpe"], 3), round(row["OOS_sharpe"], 3), flush=True)
    res = pd.DataFrame(rows)
    keys = list(G)
    # plateau
    idx = {k: {x: i for i, x in enumerate(G[k])} for k in keys}
    code = pd.DataFrame({k: res[k].map(idx[k]) for k in keys})
    look = pd.Series(res.IS_sharpe.values, pd.MultiIndex.from_frame(code))
    tot, cnt = res.IS_sharpe.values.copy(), np.ones(len(res))
    for k in ["lb", "reb", "skip", "mkt", "vt"]:
        for d in (-1, 1):
            nb = code.copy(); nb[k] += d
            vv = look.reindex(pd.MultiIndex.from_frame(nb)).values; ok = ~np.isnan(vv)
            tot[ok] += vv[ok]; cnt[ok] += 1
    res["plateau"] = tot / cnt
    res.to_csv(out / "dms_grid.csv", index=False)
    sr0 = dm.expected_max_sr(len(res), float(res.IS_sharpe.var()))
    rho = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]

    tab, chosen = [], {}
    for lb in G["lb"]:
        sub = res[res.lb == lb]
        for sch in ["chosen"] + G["scheme"]:
            s2 = sub if sch == "chosen" else sub[sub.scheme == sch]
            p = s2.sort_values(["plateau", "IS_sharpe"], ascending=False).iloc[0]
            P = {k: p[k] for k in keys}
            if sch == "chosen":
                chosen[lb] = P
            for per, lo, hi in PERIODS:
                r, o = run(P, lo, hi)
                s = stats(r); yrs = len(r) / 252
                tab.append(dict(lb=lb, row=sch, cfg="{} skip{} mkt{} vt{} {}".format(P["reb"], P["skip"], P["mkt"], P["vt"], P["scheme"]),
                                period=per, CAGR=s["CAGR"], sharpe=s["sharpe"], mdd=s["mdd"],
                                turnover_yr=o["turn"] / yrs, cost_pct_yr=o["cost"] / yrs * 100,
                                DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan))
    # 基準
    bm = {"TXF B&H": D["txr"]}
    D["mes"] = schedule("M", IS0, END)
    ew = pd.Series(0.0, D["days"])
    for n, i in enumerate(D["mes"][:-1]):
        U = o22.univ(i, 50, 0); k0, k1 = i + 1, D["mes"][n + 1] + 1
        if len(U) >= dm.MIN_UNIVERSE:
            ew.iloc[k0:k1] = D["ret"][k0:k1, U].mean(1)
    bm["EW universe (gross)"] = ew
    for n, r in bm.items():
        for per, lo, hi in PERIODS:
            s = stats(r[(r.index >= lo) & (r.index < hi)])
            tab.append(dict(lb="bench", row=n, period=per, CAGR=s["CAGR"], sharpe=s["sharpe"], mdd=s["mdd"]))
    tab = pd.DataFrame(tab)
    tab.to_csv(out / "dms_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"試驗數 {len(res)}  SR0 {sr0:.3f}  Spearman(IS,OOS) {rho:.3f}")
    print(res.groupby("lb")[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median().round(3))
    print(res.groupby("reb")[["IS_sharpe", "OOS_sharpe"]].median().round(3))
    print(tab.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 5.5))
    for lb, P in chosen.items():
        r = paths[tuple(P[k] for k in keys)]
        plt.plot(r.index, (1 + r).cumprod(), label="lb{} {} skip{} mkt{} vt{} {}".format(lb, P["reb"], P["skip"], P["mkt"], P["vt"], P["scheme"]))
    for n, r in bm.items():
        r = r[(r.index >= IS0) & (r.index < END)]
        plt.plot(r.index, (1 + r).cumprod(), label=n, lw=1.5, ls="--")
    plt.axvline(IS1, color="k", ls="--", lw=.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title("Short-lookback dual momentum (5 holdings, net) vs benchmarks; IS | OOS")
    plt.tight_layout(); plt.savefig(out / "dms_equity.png", dpi=110); plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axs, ("IS_sharpe", "OOS_sharpe")):
        hm = res.pivot_table(index="lb", columns=["reb", "scheme"], values=col, aggfunc="median")
        im = ax.imshow(hm.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(hm.shape[1]), [f"{a}-{b}" for a, b in hm.columns], rotation=45, fontsize=8)
        ax.set_yticks(range(hm.shape[0]), hm.index)
        for (y, x), v in np.ndenumerate(hm.values):
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"median {col}"); ax.set_ylabel("lookback (days)")
    fig.colorbar(im, ax=axs); fig.savefig(out / "dms_heatmap.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
