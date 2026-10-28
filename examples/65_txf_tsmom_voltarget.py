"""65 TXF 時間序列動能 + 波動率目標（research/reddit_quant_ideas.md #13→本檔 F：#12）。日K 2010-。
訊號（第 t 日收盤）：sign(過去 L 日報酬)，L∈{20,60,120,250}；模式 LS（多空）或 LF（多/空手）；另加 EMA 雙均線 (20/100)。
曝險 w = 方向 × min(cap, tv / 已實現波動(EWMA 半衰期 20))，tv∈{無(=1倍), 10%, 15%, 20%}，cap=2。t+1 開盤調整。
報酬（與 37 相同換月處理）：舊部位吃 前收→開（換月日為 0），新部位吃 開→收；成本率/邊 ≈ (0.25 手續費 + 1 點滑價)/價 + 2e-5 稅，換月日持倉雙邊。
連續（分數口）近似，權益 1,000 萬；MDD 用盤中低/高點估。IS 2010-2019，OOS 2020-2026/10。
基準：B&H、B&H+波動目標；隨機：20 日區塊隨機方向 + 同波動目標（300 次）。
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R


def load():
    d = pd.read_parquet(R.DATAD); idx = d.date.dt.tz_localize(None).dt.normalize()
    o, h, l, c = (d[k].values.astype(float) for k in ("open", "high", "low", "close"))
    cm = d.contractMonth.astype(str); roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    pc = np.r_[np.nan, c[:-1]]
    X = pd.DataFrame(dict(o=o, c=c, r=np.where(roll, (c - o) / pc, (c - pc) / pc), gap=np.where(roll, 0, (o - pc) / pc), intr=(c - o) / pc,
                          lo=(l - o) / pc, hi=(h - o) / pc, roll=roll, cr=(1.25 / o + 2e-5)), idx).iloc[1:]
    return X


def sim(X, dirn, tv, cap=2.0):
    vol = np.sqrt((X.r ** 2).ewm(halflife=20, min_periods=20).mean()) * np.sqrt(252)
    lev = pd.Series(1.0, X.index) if tv is None else (tv / vol).clip(upper=cap)
    w = (dirn * lev).fillna(0).shift(1).fillna(0).values; wp = np.r_[0, w[:-1]]
    ret = wp * X.gap.values + w * X.intr.values - np.abs(w - wp) * X.cr.values - np.abs(wp) * X.roll.values * 2 * X.cr.values
    worst = wp * X.gap.values + np.minimum(w * X.lo.values, w * X.hi.values)
    return pd.Series(ret, X.index), pd.Series(np.minimum(worst, 0), X.index), np.abs(w - wp).sum() / (len(w) / 252)


def st(r, wst):
    eq = (1 + r).cumprod(); peak = eq.cummax(); mdd = ((eq.shift(1).fillna(1) * (1 + wst)) / peak.shift(1).fillna(1) - 1).min()
    return dict(Sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, CAGR=eq.iloc[-1] ** (252 / len(r)) - 1, MDD_intra=min(mdd, (eq / peak - 1).min()))


def main():
    X = load(); r = X.r.fillna(0); X = X[X.index >= "2010-01-01"]
    cum = (1 + r).cumprod()
    sigs = {}
    for L in (20, 60, 120, 250):
        s = np.sign(cum / cum.shift(L) - 1); sigs[f"TS{L}_LS"] = s; sigs[f"TS{L}_LF"] = s.clip(lower=0)
    e = (cum.ewm(span=20).mean() > cum.ewm(span=100).mean()).astype(float); sigs["EMA20/100_LS"] = e * 2 - 1; sigs["EMA20/100_LF"] = e
    sigs["BH"] = pd.Series(1.0, X.index)
    ise = X.index < "2020-01-01"; warm = X.index >= "2011-01-01"; rows = []; ser = {}
    for (k, s), tv in itertools.product(sigs.items(), (None, 0.10, 0.15, 0.20)):
        ret, wst, turn = sim(X, s.reindex(X.index), tv); ret, wst = ret[warm], wst[warm]; i = ise[warm]
        a, b = st(ret[i], wst[i]), st(ret[~i], wst[~i]); key = f"{k}|tv{tv}"; ser[key] = ret
        top = ret[~i].nlargest(10).sum() / max(ret[~i].sum(), 1e-12)
        rows.append(dict(key=key, sig=k, tv=tv, IS=a["Sharpe"], OOS=b["Sharpe"], IS_CAGR=a["CAGR"], OOS_CAGR=b["CAGR"], IS_MDD=a["MDD_intra"], OOS_MDD=b["MDD_intra"], turn=turn, OOS_top10=top))
    G = pd.DataFrame(rows); print(G.round(3).to_string())
    g = G[~G.sig.eq("BH")]; sp = g.IS.rank().corr(g.OOS.rank()); print(f"Spearman IS/OOS (non-BH) {sp:.3f}")
    best = g.sort_values("IS").iloc[-1]; print("IS-best", best.key, f"IS {best.IS:.2f} OOS {best.OOS:.2f}")
    bhv = G[(G.sig == "BH") & (G.tv == 0.15)].iloc[0]; print(f"BH tv0.15: IS {bhv.IS:.2f} OOS {bhv.OOS:.2f}")
    rng = np.random.default_rng(11); sims = []
    for _ in range(300):
        nb = len(X) // 20 + 1; s = pd.Series(np.repeat(rng.choice([-1., 1.], nb), 20)[:len(X)], X.index)
        ret, _, _ = sim(X, s, best.tv if best.tv == best.tv else None); ret = ret[warm]; sims.append(ret[~ise[warm]].mean() / ret[~ise[warm]].std() * np.sqrt(252))
    print(f"random block sign OOS Sharpe mean {np.mean(sims):.2f} 5-95% [{np.percentile(sims,5):.2f},{np.percentile(sims,95):.2f}]")
    wf = []; picks = []
    for Y in range(2014, 2027):
        sc = {k: v[(v.index.year >= Y - 3) & (v.index.year < Y)] for k, v in ser.items() if not k.startswith("BH")}
        sc = {k: v.mean() / v.std() for k, v in sc.items()}; b = max(sc, key=sc.get); y = ser[b][ser[b].index.year == Y]; wf.append(y); picks.append((Y, b))
    wf = pd.concat(wf); print("WF picks", picks); oo = wf[wf.index >= "2020-01-01"]
    print(f"WF stitched Sharpe 2014+ {wf.mean()/wf.std()*np.sqrt(252):.3f} | 2020+ {oo.mean()/oo.std()*np.sqrt(252):.3f}")
    G.to_csv(R.OUT / "rq65_grid.csv", index=False)
    plt = R.style(); fig, ax = plt.subplots(figsize=(14, 7))
    for k in (best.key, "TS250_LF|tv0.15", "BH|tv0.15", "BH|tvNone"):
        ax.plot((1 + ser[k]).cumprod(), label=k)
    ax.plot((1 + wf).cumprod() * (1 + ser[best.key][ser[best.key].index < "2014-01-01"]).prod(), label="WF 每年選", ls="--")
    ax.set_yscale("log"); ax.axvline(pd.Timestamp("2020-01-01"), c=R.GRAY, ls="--"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("TXF 時間序列動能 + 波動目標（淨值，log）")
    fig.tight_layout(); fig.savefig(R.OUT / "rq_65_tsmom_voltarget.png", dpi=100); print("chart results/rq_65_tsmom_voltarget.png")


if __name__ == "__main__":
    main()
