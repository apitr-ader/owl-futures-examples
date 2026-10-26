"""52 最佳化 51 的波動率濾網 covered call：IS 2010-19 選參數，檢查 OOS 2020+ 穩定度。

網格：OTM% {0,2,3,5,7,10} × RV 回看 {10,20,60} × 參考窗 {126,252,504} × 分位 {30,50,70} × 方向 {above,below}
  + 每個 OTM% 的「每月都賣」基準。規則/成本/MDD 全部沿用 45/51（收盤訊號、隔日開盤、盤中最差 MDD）。
訊號：E 收盤的 RV_L 與「過去 W 日滾動 RV_L 的 p 分位」比較（只用 ≤E 資料）；暖機不足的月份不賣。
"""
import importlib.util, math
from pathlib import Path
import numpy as np, pandas as pd

def _load(n, f):
    s = importlib.util.spec_from_file_location(n, Path(__file__).with_name(f)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cc = _load("cc45", "45_txf_covered_call.py"); vf = _load("vf51", "51_txf_cc_volfilter.py")
OUT = Path("results")
OTMS = (0, 2, 3, 5, 7, 10); LBS = (10, 20, 60); WINS = (126, 252, 504); PCTS = (30, 50, 70); DIRS = ("above", "below")
IS_, OOS_ = cc.PERIODS[1], cc.PERIODS[2]


def main():
    OUT.mkdir(exist_ok=True)
    F = cc.load_fut(); O = cc.load_opt(); cycles, optdays = cc.build_cycles(F, O)
    full = pd.read_parquet(cc.DATA); full.index = full["date"].dt.tz_localize(None).dt.normalize()
    full = full[full.index <= cc.END]
    cm = full.contractMonth.astype(str); roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    r = np.log(full.close / full.close.shift()); r[roll] = np.log(full.close / full.open)[roll]
    sig = {}
    for L in LBS:
        rv = r.rolling(L).std() * math.sqrt(252)
        for W in WINS:
            for p in PCTS:
                q = rv.rolling(W, min_periods=W).quantile(p / 100)
                sig[(L, W, p)] = {cy["expiry"]: (rv.get(cy["E"], np.nan), q.get(cy["E"], np.nan)) for cy in cycles}
    combos = [dict(otm=o, lb=None, win=None, pct=None, dir="always") for o in OTMS]
    combos += [dict(otm=o, lb=L, win=W, pct=p, dir=d) for o in OTMS for L in LBS for W in WINS for p in PCTS for d in DIRS]
    bh = cc.simulate(F, O, cycles, optdays)
    rows = []; curves = {}
    def met(res, per):
        e, w, Lg, w2 = res; st = cc.stats(e, w, per[1], per[2], w2)
        n = int(((Lg.settle >= per[1]) & (Lg.settle < per[2])).sum()) if len(Lg) else 0
        return dict(CAGR=st["CAGR"], vol=st["vol"], Sharpe=st["Sharpe"], MDD=st["MDD_intraday"], months=n)
    bhm = {k: met(bh, per) for k, per in (("IS", IS_), ("OOS", OOS_), ("FULL", cc.PERIODS[0]))}
    for i, c in enumerate(combos):
        if c["dir"] == "always":
            allow = lambda cy: True
        else:
            s = sig[(c["lb"], c["win"], c["pct"])]; up = c["dir"] == "above"
            def allow(cy, s=s, up=up):
                v, q = s[cy["expiry"]]
                return not np.isnan(q) and ((v > q) if up else (v <= q))
        cc.pick_leg = vf.make_pick(allow)
        res = cc.simulate(F, O, cycles, optdays, ("x", "pct", c["otm"] / 100), 1.0, [])
        row = dict(id=i, **c)
        for k, per in (("IS", IS_), ("OOS", OOS_), ("FULL", cc.PERIODS[0])):
            row.update({f"{k}_{m}": v for m, v in met(res, per).items()})
        rows.append(row); curves[i] = res[0]
    T = pd.DataFrame(rows)
    T["IS_rank_S"] = T.IS_Sharpe.rank(ascending=False, method="min").astype(int)
    T["OOS_rank_S"] = T.OOS_Sharpe.rank(ascending=False, method="min").astype(int)
    T["IS_rank_C"] = T.IS_CAGR.rank(ascending=False, method="min").astype(int)
    T["OOS_rank_C"] = T.OOS_CAGR.rank(ascending=False, method="min").astype(int)
    T.to_csv(OUT / "cc_opt_grid.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    cols = ["otm", "lb", "win", "pct", "dir", "IS_CAGR", "IS_Sharpe", "IS_MDD", "IS_months", "OOS_CAGR", "OOS_Sharpe", "OOS_MDD", "OOS_months", "OOS_rank_S", "OOS_rank_C"]
    def show(d):
        d = d[cols].copy()
        for k in ("IS_CAGR", "IS_MDD", "OOS_CAGR", "OOS_MDD"): d[k] = (d[k] * 100).round(2)
        return d.round(3).to_string(index=False)
    print("N combos", len(T)); print("B&H", {k: {m: round(v, 4) for m, v in d.items()} for k, d in bhm.items()})
    print("\n== IS top10 by Sharpe =="); print(show(T.sort_values("IS_Sharpe", ascending=False).head(10)))
    print("\n== IS top10 by CAGR =="); print(show(T.sort_values("IS_CAGR", ascending=False).head(10)))
    print("\n== always baselines =="); print(show(T[T.dir == "always"]))
    spearmanr = lambda a, b: (a.rank().corr(b.rank()),)
    for m in ("Sharpe", "CAGR"):
        print(f"Spearman IS vs OOS {m}:", round(spearmanr(T[f"IS_{m}"], T[f"OOS_{m}"])[0], 3),
              "| filtered only:", round(spearmanr(T[T.dir != 'always'][f"IS_{m}"], T[T.dir != 'always'][f"OOS_{m}"])[0], 3))
    for k in ("IS", "OOS"):
        print(f"{k} beat B&H: Sharpe {(T[f'{k}_Sharpe'] > bhm[k]['Sharpe']).mean():.3f}  CAGR {(T[f'{k}_CAGR'] > bhm[k]['CAGR']).mean():.3f}")
    print("OOS beat always-same-OTM Sharpe:", round((T[T.dir != "always"].apply(lambda x: x.OOS_Sharpe > T[(T.dir == "always") & (T.otm == x.otm)].OOS_Sharpe.iloc[0], axis=1)).mean(), 3))
    print("\nby win (filtered) mean:"); print(T[T.dir != "always"].groupby("win")[["IS_months", "IS_Sharpe", "OOS_Sharpe"]].mean().round(3))
    print("\nby dir mean:"); print(T.groupby("dir")[["IS_Sharpe", "OOS_Sharpe", "IS_CAGR", "OOS_CAGR"]].mean().round(4))
    print("\nby otm mean:"); print(T.groupby("otm")[["IS_Sharpe", "OOS_Sharpe", "IS_CAGR", "OOS_CAGR"]].mean().round(4))
    chart(T, bhm, curves, bh)


def chart(T, bhm, curves, bh):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f = T[T.dir != "always"]
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for j, d in enumerate(DIRS):
        for i, k in enumerate(("IS", "OOS")):
            a = ax[j, i * 2]; P = f[f.dir == d].pivot_table(index="otm", columns="pct", values=f"{k}_Sharpe", aggfunc="mean")
            im = a.imshow(P.values, cmap="RdYlGn", vmin=f[f"{k}_Sharpe"].min(), vmax=f[f"{k}_Sharpe"].max(), aspect="auto")
            for (y, x), v in np.ndenumerate(P.values): a.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=8)
            a.set_xticks(range(len(P.columns))); a.set_xticklabels(P.columns); a.set_yticks(range(len(P.index))); a.set_yticklabels(P.index)
            a.set_title(f"{k} Sharpe mean, dir={d} (B&H {bhm[k]['Sharpe']:.2f})"); a.set_xlabel("pct"); a.set_ylabel("OTM %")
            a = ax[j, i * 2 + 1]; P = f[f.dir == d].pivot_table(index="lb", columns="win", values=f"{k}_Sharpe", aggfunc="mean")
            a.imshow(P.values, cmap="RdYlGn", aspect="auto")
            for (y, x), v in np.ndenumerate(P.values): a.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=8)
            a.set_xticks(range(len(P.columns))); a.set_xticklabels(P.columns); a.set_yticks(range(len(P.index))); a.set_yticklabels(P.index)
            a.set_title(f"{k} Sharpe mean, dir={d}"); a.set_xlabel("ref window"); a.set_ylabel("RV lookback")
    fig.tight_layout(); fig.savefig(OUT / "cc_opt_heatmap.png", dpi=110)
    fig, a = plt.subplots(figsize=(8, 7))
    for d, col in (("above", "tab:red"), ("below", "tab:blue"), ("always", "black")):
        s = T[T.dir == d]; a.scatter(s.IS_Sharpe, s.OOS_Sharpe, s=14 if d != "always" else 50, c=col, label=d, alpha=.7)
    a.axhline(bhm["OOS"]["Sharpe"], ls="--", c="gray"); a.axvline(bhm["IS"]["Sharpe"], ls="--", c="gray")
    a.set_xlabel("IS Sharpe 2010-19"); a.set_ylabel("OOS Sharpe 2020-26/08"); a.set_title("IS vs OOS Sharpe (dashed = TXF B&H)"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "cc_opt_scatter.png", dpi=110)
    fig, a = plt.subplots(figsize=(13, 6))
    e = bh[0]; a.plot(e.index, e / e.iloc[0], c="k", lw=1.6, label="TXF B&H")
    for lab, k in (("IS-best Sharpe", "IS_Sharpe"), ("IS-best CAGR", "IS_CAGR")):
        r = T.loc[T[k].idxmax()]; e = curves[r.id]
        a.plot(e.index, e / e.iloc[0], lw=1.1, label=f"{lab}: OTM{r.otm}% {r.dir} lb{r.lb} w{r.win} p{r.pct}")
    for o in (5,):
        r = T[(T.dir == "always") & (T.otm == o)].iloc[0]; e = curves[r.id]; a.plot(e.index, e / e.iloc[0], lw=1, ls=":", label=f"always OTM{o}%")
    a.axvline(pd.Timestamp("2020-01-01"), c="gray", ls="--"); a.set_yscale("log"); a.legend(fontsize=8); a.grid(alpha=.3); a.set_title("Equity (log)")
    fig.tight_layout(); fig.savefig(OUT / "cc_opt_equity.png", dpi=110)
    print("charts results/cc_opt_heatmap.png cc_opt_scatter.png cc_opt_equity.png")


if __name__ == "__main__":
    main()
