"""63 體制濾網（research/reddit_quant_ideas.md #3、#11/#10）作為風險開關，套用於 TXF 買進持有（日K）與 IB 突破（56 最佳參數交易 results/ib_best_trades.csv）。
指標（第 t 日收盤算，t+1 起生效；IB 夜盤在 t 日 15:00 後開始，同樣只用 t 日收盤資訊）：
  FRO：大型股相關矩陣 21 日 vs 252 日 Frobenius 距離 / N。股票池：每月初取過去 60 日平均成交值前 30 名且有 252 日完整資料者（含下市股，避免倖存者偏差）。
  COMM：共變異數斷裂簡化版：||Σ21Σ252 − Σ252Σ21||_F /(||Σ21||_F||Σ252||_F)（Lie bracket 交換子範數）；AVC：平均相關 21d − 252d。
       BRK = COMM 百分位 > q 且 AVC > 0（相關性上升）。
  GARCH：TXF 日報酬 GARCH(1,1)（常態 MLE，每年用之前全部資料重估，因果），條件波動在過去 252 日百分位，介於 [lo, hi] 才 risk-on。
百分位：指標在過去 504 日的百分位（因果）。關閉條件：FRO 百分位 > q；COMM 同；BRK；GARCH 超出區間。
TXF B&H：倒推調整價，1 口，開→收／收→開分段計，進出場與換月扣成本；IS 2011-2019，OOS 2020+。IB：IS 2017-06~2021，OOS 2022+。
隨機基準：相同關閉比例、以 20 日區塊隨機關閉（200 次）。
"""
import sys, math, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R, _bt_data as B
sys.path.insert(0, "/workspace/fubon-threads/tools"); import stockdata as sd

CACHE = R.OUT / "rq63_indicators.parquet"


def garch_path(r, w, a, b):
    v = np.empty(len(r)); v[0] = r.var()
    for t in range(1, len(r)): v[t] = w + a * r[t - 1] ** 2 + b * v[t - 1]
    return v


def garch_fit(r):
    def nll(p):
        w, a, b = np.exp(p[0]), 1 / (1 + np.exp(-p[1])) * .3, 1 / (1 + np.exp(-p[2]))
        if a + b >= .999: return 1e10
        v = garch_path(r, w, a, b); return 0.5 * np.sum(np.log(v) + r ** 2 / v)
    p = minimize(nll, [math.log(r.var() * .05), 0, 2], method="Nelder-Mead", options=dict(maxiter=2000)).x
    return np.exp(p[0]), 1 / (1 + np.exp(-p[1])) * .3, 1 / (1 + np.exp(-p[2]))


def indicators(D):
    if CACHE.exists(): return pd.read_parquet(CACHE)
    A = sd.panel("close", adjusted=True, start="2008-01-01"); T = sd.panel("turnover", adjusted=False, start="2008-01-01").reindex_like(A)
    Rt = A.pct_change(fill_method=None); t60 = T.rolling(60, min_periods=40).mean(); cnt = Rt.notna().rolling(252).sum()
    days = A.index[A.index >= "2010-01-01"]; out = []; names = None
    for i, t in enumerate(days):
        if i == 0 or t.month != days[i - 1].month or names is None:
            ok = cnt.loc[t] >= 250; names = t60.loc[t][ok].nlargest(30).index
        X = Rt.loc[:t, names].iloc[-252:].fillna(0).values
        if len(X) < 252: continue
        S252 = np.cov(X.T); S21 = np.cov(X[-21:].T)
        d252 = np.sqrt(np.diag(S252)); d21 = np.sqrt(np.diag(S21)) + 1e-12
        C252 = S252 / np.outer(d252, d252); C21 = S21 / np.outer(d21, d21); n = len(names)
        fro = np.linalg.norm(C21 - C252) / n
        comm = np.linalg.norm(S21 @ S252 - S252 @ S21) / (np.linalg.norm(S21) * np.linalg.norm(S252))
        iu = np.triu_indices(n, 1); avc = C21[iu].mean() - C252[iu].mean()
        out.append((t, fro, comm, avc))
    I = pd.DataFrame(out, columns=["date", "fro", "comm", "avc"]).set_index("date"); I.index = pd.to_datetime(I.index)
    I.to_parquet(CACHE); return I


def pct(x, n=504):
    return x.rolling(n, min_periods=252).apply(lambda v: (v[:-1] < v[-1]).mean(), raw=True)


def main():
    D = B.load_ohlcv(R.DATAD, roll_adjust="back"); D.index = D.index.tz_localize(None).normalize()
    cm = D.contractMonth.astype(str); roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    I = indicators(D).reindex(D.index).ffill()
    # GARCH 因果：年 Y 用 < Y 的報酬估參數
    raw = pd.read_parquet(R.DATAD); raw.index = raw.date.dt.tz_localize(None).dt.normalize()
    r = np.log(D.Close).diff().where(~roll, np.log(D.Close / D.Open)).fillna(0) * 100
    gv = pd.Series(np.nan, D.index)
    for Y in range(2012, D.index.year.max() + 1):
        w, a, b = garch_fit(r[r.index.year < Y].values[1:]); v = garch_path(r.values, w, a, b)
        m = D.index.year == Y; gv[m] = np.sqrt(v[m])
    I["garch"] = gv
    P = pd.DataFrame({k: pct(I[k]) for k in ("fro", "comm", "garch")}); P["avc"] = I.avc
    filt = {"none": pd.Series(True, D.index)}
    for q in (0.8, 0.9, 0.95):
        filt[f"FRO>{q}"] = ~(P.fro > q); filt[f"COMM>{q}"] = ~(P.comm > q); filt[f"BRK{q}"] = ~((P.comm > q) & (P.avc > 0))
    for lo, hi in ((0.1, 0.9), (0.0, 0.9), (0.0, 0.8), (0.1, 1.0)): filt[f"GARCH{lo}-{hi}"] = (P.garch >= lo) & (P.garch <= hi)
    filt["GARCH0.1-0.9&FRO>0.9"] = filt["GARCH0.1-0.9"] & filt["FRO>0.9"]
    valid = P.fro.notna() & P.garch.notna(); st = D.index[valid.values][0]; print("indicators valid from", st.date())
    o, h, l, c = (D[k].values for k in ("Open", "High", "Low", "Close"))

    def bh(on):
        pos = on.shift(1).fillna(False).astype(float).values   # t 收盤決定 → t+1 開盤
        pp = np.r_[0, pos[:-1]]; pc = np.r_[c[0], c[:-1]]
        pnl = pp * (o - pc) + pos * (c - o) - np.abs(pos - pp) * (0.25 + 2e-5 * o + 1) - pos * roll * (0.5 + 4e-5 * o + 2)
        mae = np.minimum(0, pp * (o - pc) + pos * (l - o)); return pd.Series(pnl * R.PV, D.index), pd.Series(mae * R.PV, D.index)

    ib = pd.read_csv(R.OUT / "ib_best_trades.csv", parse_dates=["tdate"]); ib["mae"] = -ib.risk
    rows = []; series = {}
    for k, on in filt.items():
        on = on.fillna(True) & True
        dp, mae = bh(on); m = D.index >= st; dp, mae = dp[m], mae[m]
        isd = dp.index < "2020-01-01"; s_is = R.stats(dp[isd], mae[isd]); s_oo = R.stats(dp[~isd], mae[~isd])
        prev = on.shift(1).fillna(True)   # 交易日 tdate 用前一交易日收盤的開關
        keep = prev.reindex(ib.tdate).fillna(True).values.astype(bool); t = ib[keep]
        alld = pd.DatetimeIndex(sorted(ib.tdate.unique())); di, _ = R.daily_from_trades(t, alld); ise = alld < "2022-01-01"
        rows.append(dict(filter=k, off_frac=1 - on[m].mean(), BH_IS=s_is["Sharpe"], BH_OOS=s_oo["Sharpe"], BH_IS_MDD=s_is["MDD_intraday"], BH_OOS_MDD=s_oo["MDD_intraday"],
                         BH_OOS_ann=s_oo["ann_ret"], IB_IS=R.sharpe(di[ise]), IB_OOS=R.sharpe(di[~ise]), IB_kept=keep.mean(),
                         IB_OOS_MDD=R.stats(di[~ise])["MDD_close"]))
        series[k] = (dp, di)
    G = pd.DataFrame(rows); print(G.round(3).to_string())
    for tgt in ("BH", "IB"):
        g = G[G["filter"] != "none"]; print(f"{tgt} Spearman IS/OOS %.3f" % g[f"{tgt}_IS"].rank().corr(g[f"{tgt}_OOS"].rank()))
    # 隨機區塊關閉基準
    rng = np.random.default_rng(3); m = D.index >= st
    for k in ("FRO>0.9", "BRK0.9", "GARCH0.1-0.9", G.sort_values("BH_IS").iloc[-1]["filter"]):
        f = 1 - filt[k][m].mean(); sims = []; simsib = []
        for _ in range(200):
            nb = int(m.sum() / 20) + 1; off = np.repeat(rng.random(nb) < f, 20)[:m.sum()]
            on = pd.Series(True, D.index); on[m] = ~off; dp, mae = bh(on); dp = dp[m]; sims.append(R.sharpe(dp[dp.index >= "2020-01-01"]))
            prev = on.shift(1).fillna(True); keep = prev.reindex(ib.tdate).fillna(True).values.astype(bool)
            alld = pd.DatetimeIndex(sorted(ib.tdate.unique())); di, _ = R.daily_from_trades(ib[keep], alld); simsib.append(R.sharpe(di[alld >= "2022-01-01"]))
        print(f"random block-off {k} (off {f:.2f}): BH OOS Sharpe mean {np.mean(sims):.3f} p95 {np.percentile(sims,95):.3f} | IB OOS mean {np.mean(simsib):.3f} p95 {np.percentile(simsib,95):.3f}")
    # WF：每年以過去 3 年 B&H Sharpe 選濾網
    for tgt, idx in (("BH", 0), ("IB", 1)):
        wf = []; picks = []
        for Y in range(2020 if tgt == "BH" else 2021, 2027):
            sc = {k: R.sharpe(v[idx][(v[idx].index.year >= Y - 3) & (v[idx].index.year < Y)]) for k, v in series.items()}
            b = max(sc, key=lambda k: -9 if sc[k] != sc[k] else sc[k]); y = series[b][idx]; y = y[y.index.year == Y]; wf.append(y); picks.append((Y, b, round(R.sharpe(y), 2)))
        wf = pd.concat(wf); print(tgt, "WF picks", picks, "stitched Sharpe %.3f" % R.sharpe(wf)); G.loc[0, f"{tgt}_WF"] = R.sharpe(wf)
    G.to_csv(R.OUT / "rq63_filters.csv", index=False)
    plt = R.style(); fig, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    for k in ("none", "FRO>0.9", "BRK0.9", "GARCH0.1-0.9"):
        ax[0].plot(series[k][0].cumsum() / 1e4, label=k); ax[1].plot(series[k][1].cumsum() / 1e4, label=k)
    ax[0].set_title("TXF 買進持有 + 體制開關（萬元/1口）"); ax[1].set_title("IB 突破（56 最佳）+ 體制開關"); ax[0].axvline(pd.Timestamp("2020-01-01"), c=R.GRAY, ls="--"); ax[1].axvline(pd.Timestamp("2022-01-01"), c=R.GRAY, ls="--")
    for a in ax[:2]: a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[2]; a.plot(P.fro, label="FRO 百分位", lw=.7); a.plot(P.garch, label="GARCH 百分位", lw=.7); a.plot(P.comm, label="COMM 百分位", lw=.5, alpha=.6); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(R.OUT / "rq_63_regime_filters.png", dpi=100); print("chart results/rq_63_regime_filters.png")
    return filt


if __name__ == "__main__":
    main()
