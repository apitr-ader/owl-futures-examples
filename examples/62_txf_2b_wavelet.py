"""62 Sperandeo 2B 趨勢反轉：滾動視窗 pivot vs 因果小波去噪 pivot（research/reddit_quant_ideas.md #9）。TXF 日 K（2010+）與 60 分 K（2017+）。
價格：換月倒推調整（_bt_data roll_adjust=back）算訊號與損益；持倉跨換月扣一次來回成本。
pivot：
  roll(n)：H[i] 為 [i−n, i+n] 最高 → 於 i+n 確認（因果）；低點對稱。
  wav(L)：每根只用過去 W=128 根收盤做 db4 小波分解、細節係數 soft threshold（universal, MAD），取去噪序列最後一點 s_t；
          s 斜率由正轉負 → 確認 pivot high，價位 = 上一 pivot low 以來最高價（低點對稱）。
2B：最近 3 根最高 > 最近 pivot high 且收盤跌回其下 → 放空（多方對稱）；停損 = 3 根極值 ± 1 點；每個 pivot 只用一次。
  趨勢條件 tf：none / trend（最近 pivot high > 前一 pivot high 才做空頭 2B，即原版「上升趨勢中的假突破」）。
出場：持有 Hd 根（下一根開盤）、停損、或反向訊號反手。
IS：日 K 2010-2019 / 60 分 2017-2021；OOS 之後。grid 小（n 或 L × Hd × tf），隨機方向基準，WF 3 年滾動，top-10 集中度。
"""
import sys, math, itertools
from pathlib import Path
import numpy as np, pandas as pd, pywt
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R, _bt_data as B


def wav_series(c, L, W=128):
    s = np.full(len(c), np.nan)
    for t in range(W, len(c)):
        x = c[t - W + 1:t + 1]; co = pywt.wavedec(x, "db4", level=L, mode="symmetric")
        sig = np.median(np.abs(co[-1])) / 0.6745; th = sig * math.sqrt(2 * math.log(W))
        co = [co[0]] + [pywt.threshold(d, th, "soft") for d in co[1:]]
        s[t] = pywt.waverec(co, "db4", mode="symmetric")[W - 1]
    return s


def pivots_roll(H, L, n):
    N = len(H); ph = np.full(N, np.nan); pl = np.full(N, np.nan); pph = np.full(N, np.nan); ppl = np.full(N, np.nan)
    a, b, pa, pb = np.nan, np.nan, np.nan, np.nan
    for t in range(2 * n, N):
        i = t - n
        if H[i] == H[i - n:t + 1].max(): pa, a = a, H[i]
        if L[i] == L[i - n:t + 1].min(): pb, b = b, L[i]
        ph[t], pl[t], pph[t], ppl[t] = a, b, pa, pb
    return ph, pl, pph, ppl


def pivots_wav(H, L, s):
    N = len(H); ph = np.full(N, np.nan); pl = np.full(N, np.nan); pph = np.full(N, np.nan); ppl = np.full(N, np.nan)
    a, b, pa, pb = np.nan, np.nan, np.nan, np.nan; last = 0
    for t in range(2, N):
        if s[t - 2] == s[t - 2]:
            d1, d0 = s[t - 1] - s[t - 2], s[t] - s[t - 1]
            if d1 > 0 and d0 <= 0: pa, a = a, H[last:t + 1].max(); last = t
            elif d1 < 0 and d0 >= 0: pb, b = b, L[last:t + 1].min(); last = t
        ph[t], pl[t], pph[t], ppl[t] = a, b, pa, pb
    return ph, pl, pph, ppl


def signals(H, L, C, piv, tf):
    ph, pl, pph, ppl = piv; N = len(C); sd = np.zeros(N); ss = np.full(N, np.nan); usedh = usedl = np.nan
    for t in range(3, N):
        hh, ll = H[t - 2:t + 1].max(), L[t - 2:t + 1].min()
        if ph[t] == ph[t] and ph[t] != usedh and hh > ph[t] and C[t] < ph[t] and (tf == "none" or ph[t] > pph[t]):
            sd[t] = -1; ss[t] = hh + 1; usedh = ph[t]
        elif pl[t] == pl[t] and pl[t] != usedl and ll < pl[t] and C[t] > pl[t] and (tf == "none" or pl[t] < ppl[t]):
            sd[t] = 1; ss[t] = ll - 1; usedl = pl[t]
    return sd, ss


def run_tf(label, X, is_end, holds, ann_days):
    o, h, l, c = (X[k].values for k in ("Open", "High", "Low", "Close"))
    cm = X.contractMonth.astype(str); roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    td = X.index.tz_localize(None).normalize() if label == "D" else X["tdate"]
    td = pd.DatetimeIndex(td); alld = pd.DatetimeIndex(sorted(set(td)))
    P = {}
    for n in (3, 5, 10): P[("roll", n)] = pivots_roll(h, l, n)
    for Lv in (2, 3, 4): P[("wav", Lv)] = pivots_wav(h, l, wav_series(c, Lv))
    res = {}; rows = []
    for (pk, Hd, tf) in itertools.product(P, holds, ("none", "trend")):
        sd, ss = signals(h, l, c, P[pk], tf)
        pnl, mae, tr = R.run_sig(o, h, l, c, roll, sd, ss, Hd)
        dp = pd.Series(pnl * R.PV, td).groupby(level=0).sum().reindex(alld, fill_value=0)
        dm = pd.Series(mae * R.PV, td).groupby(level=0).min().reindex(alld, fill_value=0)
        T = pd.DataFrame(tr, columns=["i0", "i1", "dir", "ep", "xp", "gross"]); T["tdate"] = td[T.i1.astype(int)]
        T["cost"] = 0.5 + 2e-5 * (T.ep + T.xp); T["net"] = T.gross - T.cost; T["mae"] = 0.0
        isd = alld < is_end
        key = (pk[0], pk[1], Hd, tf); res[key] = (dp, dm, T)
        rows.append(dict(key=key, method=pk[0], IS_trades=int((T.tdate < is_end).sum()), OOS_trades=int((T.tdate >= is_end).sum()),
                         IS_Sharpe=R.sharpe(dp[isd]), OOS_Sharpe=R.sharpe(dp[~isd]), IS_MDD=R.stats(dp[isd], dm[isd])["MDD_intraday"],
                         OOS_MDD=R.stats(dp[~isd], dm[~isd])["MDD_intraday"]))
    G = pd.DataFrame(rows); DP = np.array([res[k][0].values for k in G.key]); ok = G[G.IS_trades >= 20]
    sp = ok.IS_Sharpe.rank().corr(ok.OOS_Sharpe.rank())
    print(f"\n===== 2B {label} =====\n", G.round(3).to_string())
    print("Spearman %.3f" % sp); print(G.groupby("method")[["IS_Sharpe", "OOS_Sharpe", "IS_MDD", "OOS_MDD"]].mean().round(3).to_string())
    print("roll vs wav paired (same Hd/tf): wav lower OOS MDD share %.2f, wav higher OOS Sharpe share %.2f" % (
        (G[G.method == "wav"].OOS_MDD.values > G[G.method == "roll"].OOS_MDD.values).mean(), (G[G.method == "wav"].OOS_Sharpe.values > G[G.method == "roll"].OOS_Sharpe.values).mean()))
    out = {}
    for m in ("roll", "wav"):
        bk = ok[ok.method == m].sort_values("IS_Sharpe", ascending=False).key.iloc[0]; dp, dm, T = res[bk]
        wfp, wfs = R.walk_forward(G[G.method == m].reset_index(drop=True), DP[(G.method == m).values], alld, is_end.year if label == "60m" else 2013, min_tr=20)
        T2 = T.copy(); rd = R.rand_dir(T2, alld, is_end, 300)
        s = R.report(f"2B_{label}_{m}", G, sp, bk, T2, alld, is_end, wfp, wfs, rd)
        s["OOS_Sharpe"] = R.sharpe(dp[alld >= is_end]); s["IS_Sharpe"] = R.sharpe(dp[alld < is_end]); s["WF_Sharpe"] = R.sharpe(wfs)
        s["OOS_MDD_intra"] = R.stats(dp[alld >= is_end], dm[alld >= is_end])["MDD_intraday"]; s["rand_OOS_p95"] = np.percentile(rd, 95)
        out[m] = (s, dp, wfs)
    return out, G


def main():
    D = B.load_ohlcv(R.DATAD, roll_adjust="back")
    oD, GD = run_tf("D", D, pd.Timestamp("2020-01-01"), (5, 10, 20), 252)
    M = pd.read_parquet("data/TXF_1c_60m_sessions.parquet"); M["date"] = M.date.dt.tz_convert("Asia/Taipei")
    M = M.set_index("date").sort_index(); M = M.rename(columns=str.capitalize).rename(columns={"Contractmonth": "contractMonth"})
    cm = M.contractMonth.astype(str); rl = cm.ne(cm.shift()) & cm.shift().notna(); gap = (M.Open - M.Close.shift()).where(rl, 0.0)
    adj = gap[::-1].cumsum()[::-1].shift(-1, fill_value=0.0)
    for k in ("Open", "High", "Low", "Close"): M[k] = M[k].astype(float) + adj.values
    ts = M.index.tz_localize(None); dd = ts.normalize()
    M["tdate"] = np.where(ts.hour >= 15, dd + pd.Timedelta(days=1), dd)   # 夜盤歸次日（假日誤差不影響 Sharpe 彙總）
    oM, GM = run_tf("60m", M, pd.Timestamp("2022-01-01"), (5, 10, 20), 252)
    S = pd.DataFrame([v[0] for v in list(oD.values()) + list(oM.values())]); S.to_csv(R.OUT / "rq62_summary.csv", index=False); print(S.round(3).to_string())
    pd.concat([GD.assign(tf="D"), GM.assign(tf="60m")]).to_csv(R.OUT / "rq62_grid.csv", index=False)
    plt = R.style(); fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    for a, (lab, o, ie) in zip(ax[:2], (("日K", oD, "2020-01-01"), ("60分K", oM, "2022-01-01"))):
        for m, (s, dp, wfs) in o.items(): a.plot(dp.index, dp.cumsum() / 1e4, label=f"{m} IS最佳"); a.plot(wfs.index, wfs.cumsum() / 1e4, ls=":", label=f"{m} WF")
        a.axvline(pd.Timestamp(ie), c=R.GRAY, ls="--"); a.set_title(f"2B 反轉 TXF {lab}"); a.legend(fontsize=8); a.grid(alpha=.3); a.set_ylabel("萬元/1口")
    for G, mk in ((GD, "o"), (GM, "^")):
        for m in ("roll", "wav"): s = G[G.method == m]; ax[2].scatter(s.IS_Sharpe, s.OOS_Sharpe, marker=mk, label=f"{'日K' if G is GD else '60m'} {m}")
    ax[2].axhline(0, c=R.GRAY); ax[2].axvline(0, c=R.GRAY); ax[2].set_xlabel("IS Sharpe"); ax[2].set_ylabel("OOS Sharpe"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(R.OUT / "rq_62_2b_wavelet.png", dpi=100); print("chart results/rq_62_2b_wavelet.png")


if __name__ == "__main__":
    main()
