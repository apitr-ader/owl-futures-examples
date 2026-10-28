"""64 PnL 元濾網（research/reddit_quant_ideas.md #4）：用策略自身權益曲線決定是否交易。
對象：IB 突破（56 最佳，results/ib_best_trades.csv）、Silver Bullet（55 最佳，results/sb_best_trades.csv）。
以「交易日」為單位：當日是否交易只看之前交易日（< tdate）的影子（未過濾）損益 → 因果。
規則：mom N：過去 N 個交易日影子損益和 > 0 才做；mr N：< 0 才做；
      ma N：影子權益 > 其 N 日均線才做；switch N,M：過去 M 日中 mom-N 與 mr-N 誰的過濾後損益較高就用誰（post 的動能/均值回歸切換）。
IS：IB 2017-06~2021、SB 同；OOS 2022+。隨機基準：同保留比例隨機跳過交易日（500 次）。
注意：兩策略參數本身已在 2017-2021 選出，IS 數字偏樂觀；判斷看 OOS 相對「不過濾」的差。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R


def rules(x):
    out = {"none": np.ones(len(x), bool)}
    cs = np.r_[0, np.cumsum(x)]
    for N in (5, 10, 20, 40):
        past = cs[:-1] - np.r_[np.zeros(N), cs[:-1][:-N]][:len(x)]   # 過去 N 日和（不含當日）
        warm = np.arange(len(x)) >= N
        out[f"mom{N}"] = ~warm | (past > 0); out[f"mr{N}"] = ~warm | (past < 0)
        eq = pd.Series(cs[:-1]); ma = eq.rolling(N).mean().values
        out[f"ma{N}"] = ~warm | (eq.values > ma)
        for M in (50, 100):
            fm, fr = x * out[f"mom{N}"], x * out[f"mr{N}"]
            cm, cr = np.r_[0, np.cumsum(fm)], np.r_[0, np.cumsum(fr)]
            sm = cm[:-1] - np.r_[np.zeros(M), cm[:-1][:-M]][:len(x)]; sr = cr[:-1] - np.r_[np.zeros(M), cr[:-1][:-M]][:len(x)]
            out[f"sw{N}_{M}"] = np.where(sm >= sr, out[f"mom{N}"], out[f"mr{N}"])
    return out


def run(name, path):
    t = pd.read_csv(R.OUT / path, parse_dates=["tdate"])
    d = t.groupby("tdate").net.sum() * R.PV; x = d.values; idx = d.index; ise = idx < "2022-01-01"
    rs = rules(x); rows = []; ser = {}
    for k, m in rs.items():
        y = pd.Series(x * m, idx); ser[k] = y
        rows.append(dict(strategy=name, rule=k, keep=m.mean(), IS=R.sharpe(y[ise]), OOS=R.sharpe(y[~ise]), OOS_net=y[~ise].sum(),
                         OOS_MDD=R.stats(y[~ise])["MDD_close"], top10=y[~ise].nlargest(10).sum() / max(y[~ise].sum(), 1e-9)))
    G = pd.DataFrame(rows); g = G[G.rule != "none"]; sp = g.IS.rank().corr(g.OOS.rank())
    print(f"\n===== {name} (days {len(x)}) Spearman IS/OOS {sp:.3f}"); print(G.round(3).to_string())
    best = g.sort_values("IS").iloc[-1]; rng = np.random.default_rng(5); sims = []
    for _ in range(500):
        m = rng.random(len(x)) < best.keep; sims.append(R.sharpe(pd.Series(x * m, idx)[~ise]))
    print(f"IS-best {best.rule}: IS {best.IS:.2f} OOS {best.OOS:.2f} | random skip OOS mean {np.mean(sims):.2f} 5-95% [{np.percentile(sims,5):.2f},{np.percentile(sims,95):.2f}]")
    wf = []; picks = []
    for Y in range(2020, 2027):
        sc = {k: R.sharpe(v[(v.index.year >= Y - 3) & (v.index.year < Y)]) for k, v in ser.items()}
        b = max(sc, key=sc.get); y = ser[b][ser[b].index.year == Y]; wf.append(y); picks.append((Y, b))
    wf = pd.concat(wf); base = ser["none"][ser["none"].index.year >= 2020]
    print("WF picks", picks, f"stitched Sharpe {R.sharpe(wf):.3f} vs none {R.sharpe(base):.3f}")
    return dict(strategy=name, best=best.rule, IS=best.IS, OOS=best.OOS, none_IS=G.iloc[0].IS, none_OOS=G.iloc[0].OOS, spearman=sp,
                WF=R.sharpe(wf), WF_none=R.sharpe(base), rand_p95=np.percentile(sims, 95)), ser, best.rule


def main():
    S = []; plt = R.style(); fig, ax = plt.subplots(2, 1, figsize=(14, 9))
    for a, (n, p) in zip(ax, (("IB", "ib_best_trades.csv"), ("SB", "sb_best_trades.csv"))):
        s, ser, b = run(n, p); S.append(s)
        for k in ("none", b, "mom20", "mr20", "sw20_100"):
            if k in ser: a.plot(ser[k].cumsum() / 1e4, label=k)
        a.axvline(pd.Timestamp("2022-01-01"), c=R.GRAY, ls="--"); a.set_title(f"{n} 策略 PnL 元濾網（萬元）"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(R.OUT / "rq_64_pnl_metafilter.png", dpi=100)
    S = pd.DataFrame(S); print(S.round(3).to_string()); S.to_csv(R.OUT / "rq64_summary.csv", index=False); print("chart results/rq_64_pnl_metafilter.png")


if __name__ == "__main__":
    main()
