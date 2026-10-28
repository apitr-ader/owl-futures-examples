"""66 台股配對交易：Kalman 動態避險比 vs Copula（research/reddit_quant_ideas.md #13），日K。
選對（每年 Y 重選，兩方法用同一批對）：年 Y-1 年底前 60 日平均成交值前 120 名（含之後下市者）、同產業、Y-5~Y-1 有完整資料；
  依形成期 log 價格 OLS 殘差的 AR(1) 半衰期 5~60 日、ADF 型 t 值（殘差差分對落後項）最負排序，取前 20 對。
Kalman：log A = α + β log B，狀態隨機漫步（δ∈{1e-4,1e-5}），z = 新息 / √預測變異；|z|>k 進場（k∈{1.5,2.0}），z 回 0 出場。
Copula：形成期報酬 ECDF 邊際 + 高斯 copula（ρ 由常態分數估），條件機率 h1=P(U1≤u1|U2)、h2；累計錯價指數 M1+=h1-.5、M2+=h2-.5，
  M1>D 且 M2<-D 做空 A 做多 B（反之亦然），兩者回 0 出場，D∈{0.5,1.0,1.5}。
共同：收盤訊號、隔日開盤成交；每腿 100 萬（以進場開盤價計股數，Kalman 依 β 調 B 腿名目）；最長持有 40 日，年底強制平倉。
成本：手續費 0.1425%×5折 每邊、賣出證交稅 0.3%（融券賣出亦課），空方借券費 1.5%/年；未模擬停券（股東會、除權息前）與漲跌停無法成交 → 實際更差。
IS 2015-2019、OOS 2020-2026/10。隨機基準：同對、同持有天數、隨機方向/進場日。
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R
sys.path.insert(0, "/workspace/fubon-threads/tools"); import stockdata as sd

FEE, TAX, BORROW, NOT, NP, HOLD = 0.001425 * 0.5, 0.003, 0.015, 1_000_000, 20, 40


def select(C, T, ind, Y):
    end = C.index[C.index < f"{Y}-01-01"][-1]; frm = C.loc[f"{Y-5}-01-01":end]
    liq = T.loc[:end].iloc[-60:].mean().dropna(); full = frm.notna().mean() > 0.995
    cand = liq[full.reindex(liq.index).fillna(False)].nlargest(120).index
    L = np.log(frm[cand].ffill()); out = []
    for g, names in pd.Series(cand, index=cand).groupby(ind.reindex(cand).values):
        for a, b in itertools.combinations(names.values, 2):
            x, y = L[b].values, L[a].values; X = np.c_[np.ones(len(x)), x]; coef = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ coef
            de, el = np.diff(e), e[:-1]; phi = (el @ de) / (el @ el); res = de - phi * el
            t = phi / np.sqrt(res.var() / (el @ el)); hl = -np.log(2) / np.log(1 + phi) if -1 < phi < 0 else 1e9
            if 5 <= hl <= 60: out.append((a, b, t, hl, coef[1]))
    P = pd.DataFrame(out, columns=["a", "b", "t", "hl", "beta"]).sort_values("t")
    used = set(); pick = []
    for r in P.itertuples():   # 每檔最多出現在 2 對
        if sum(1 for p in pick if r.a in p[:2] or r.b in p[:2]) >= 4: continue
        pick.append((r.a, r.b, r.t, r.hl)); 
        if len(pick) == NP: break
    return pick


def kalman_z(la, lb, delta):
    n = len(la); th = np.zeros(2); Pm = np.eye(2); Vw = delta / (1 - delta) * np.eye(2); Ve = 1e-3; z = np.full(n, np.nan); beta = np.ones(n)
    for t in range(n):
        F = np.array([1.0, lb[t]]); Pm = Pm + Vw; yh = F @ th; Q = F @ Pm @ F + Ve; e = la[t] - yh
        z[t] = e / np.sqrt(Q); K = Pm @ F / Q; th = th + K * e; Pm = Pm - np.outer(K, F) @ Pm; beta[t] = th[1]
        Ve = 0.99 * Ve + 0.01 * e * e
    return z, beta


def copula_sig(ra, rb, nform, D):
    fa, fb = np.sort(ra[1:nform]), np.sort(rb[1:nform]); m = len(fa)
    u1 = np.clip(np.searchsorted(fa, ra) / (m + 1), 1e-3, 1 - 1e-3); u2 = np.clip(np.searchsorted(fb, rb) / (m + 1), 1e-3, 1 - 1e-3)
    x1, x2 = norm.ppf(u1), norm.ppf(u2); rho = np.corrcoef(x1[1:nform], x2[1:nform])[0, 1]; s = np.sqrt(1 - rho ** 2)
    h1 = norm.cdf((x1 - rho * x2) / s) - .5; h2 = norm.cdf((x2 - rho * x1) / s) - .5
    n = len(ra); sig = np.zeros(n); M1 = M2 = 0.0; pos = 0
    for t in range(nform, n):
        M1 += h1[t]; M2 += h2[t]
        if pos == 0:
            if M1 > D and M2 < -D: pos = -1
            elif M1 < -D and M2 > D: pos = 1
        elif (pos == -1 and (M1 <= 0 or M2 >= 0)) or (pos == 1 and (M1 >= 0 or M2 <= 0)):
            pos = 0; M1 = M2 = 0.0
        sig[t] = pos
    return sig


def backtest(sig, beta, Oa, Ob, Ca, Cb, t0):
    """sig[t]：t 收盤目標部位（+1 做多 A 放空 B）；t+1 開盤執行。回傳每日損益、交易列表。"""
    n = len(sig); pnl = np.zeros(n); pos = 0; qa = qb = 0.0; trades = []; ent = 0; acc = 0.0
    for t in range(t0 + 1, n):
        tgt = sig[t - 1] if t < n - 1 else 0
        if pos and (t - ent >= HOLD): tgt = 0 if sig[t - 1] == pos else tgt
        if pos and t - ent >= HOLD and sig[t - 1] == pos: tgt = 0
        if tgt != pos:
            if pos:   # 平倉於開盤
                g = qa * (Oa[t] - Ca[t - 1]) + qb * (Ob[t] - Cb[t - 1]); sells = (qa > 0) * qa * Oa[t] + (qb > 0) * qb * Ob[t]
                c = FEE * (abs(qa) * Oa[t] + abs(qb) * Ob[t]) + TAX * sells; pnl[t] += g - c; acc += g - c; trades.append((ent, t, pos, acc)); qa = qb = 0.0
            pos = tgt
            if pos:
                ent = t; acc = 0.0; qa = pos * NOT / Oa[t]; qb = -pos * NOT * min(max(beta[t - 1], 0.3), 3) / Ob[t]
                sells = (qa < 0) * -qa * Oa[t] + (qb < 0) * -qb * Ob[t]; c = FEE * (abs(qa) * Oa[t] + abs(qb) * Ob[t]) + TAX * sells
                g = qa * (Ca[t] - Oa[t]) + qb * (Cb[t] - Ob[t]); pnl[t] += g - c; acc += g - c; continue
        if pos:
            g = qa * (Ca[t] - Ca[t - 1]) + qb * (Cb[t] - Cb[t - 1]); short = (qa < 0) * -qa * Ca[t] + (qb < 0) * -qb * Cb[t]
            pnl[t] += g - short * BORROW / 252; acc += g - short * BORROW / 252
    return pnl, trades


def main():
    C = sd.panel("close", adjusted=True, start="2009-01-01"); O = sd.panel("open", adjusted=True, start="2009-01-01").reindex_like(C)
    T = sd.panel("turnover", adjusted=False, start="2009-01-01").reindex_like(C)
    ind = pd.read_csv("data/stocks/_industry_map.csv", dtype={"code": str}).set_index("code").ind
    cfgs = [("kal", d, k) for d in (1e-4, 1e-5) for k in (1.5, 2.0)] + [("cop", None, D) for D in (0.5, 1.0, 1.5)]
    DP = {c: [] for c in cfgs}; TR = {c: [] for c in cfgs}; RND = []; rng = np.random.default_rng(9); allpairs = []
    for Y in range(2015, 2027):
        pairs = select(C, T, ind, Y); allpairs += [(Y, a, b, round(t, 2), round(h, 1)) for a, b, t, h in pairs]
        W = C.loc[f"{Y-5}-01-01":f"{Y}-12-31"]; idx = W.index; t0 = int((idx < f"{Y}-01-01").sum()) - 1; yidx = idx[t0 + 1:]
        print(Y, "pairs", len(pairs), "e.g.", pairs[:3], flush=True)
        for a, b, *_ in pairs:
            Ca, Cb = C.loc[idx, a].ffill().values, C.loc[idx, b].ffill().values; Oa = O.loc[idx, a].fillna(C.loc[idx, a]).ffill().values; Ob = O.loc[idx, b].fillna(C.loc[idx, b]).ffill().values
            la, lb = np.log(Ca), np.log(Cb); ra, rb = np.r_[0, np.diff(la)], np.r_[0, np.diff(lb)]
            for c in cfgs:
                if c[0] == "kal":
                    z, beta = kalman_z(la, lb, c[1]); sig = np.zeros(len(z)); p = 0
                    for t in range(t0, len(z)):
                        if p == 0: p = -1 if z[t] > c[2] else (1 if z[t] < -c[2] else 0)
                        elif (p == -1 and z[t] <= 0) or (p == 1 and z[t] >= 0): p = 0
                        sig[t] = p
                else:
                    sig = copula_sig(ra, rb, t0 + 1, c[2]); beta = np.ones(len(sig))
                pnl, tr = backtest(sig, beta, Oa, Ob, Ca, Cb, t0)
                DP[c].append(pd.Series(pnl[t0 + 1:], yidx)); TR[c] += [(Y, a, b, idx[e], x - e, p, v) for e, x, p, v in tr]
            # 隨機基準：隨機方向、隨機進場、持有 10 日，次數同 Kalman(1e-4,2.0)
            ntr = sum(1 for r in TR[cfgs[1]] if r[0] == Y and r[1] == a and r[2] == b); sig = np.zeros(len(Ca))
            for _ in range(ntr):
                s = rng.integers(t0, len(Ca) - 11); sig[s:s + 10] = rng.choice([-1, 1])
            RND.append(pd.Series(backtest(sig, np.ones(len(Ca)), Oa, Ob, Ca, Cb, t0)[0][t0 + 1:], yidx))
    pd.DataFrame(allpairs, columns=["year", "a", "b", "adf_t", "halflife"]).to_csv(R.OUT / "rq66_pairs.csv", index=False)
    rows = []; ser = {}
    for c in cfgs:
        d = pd.concat(DP[c]).groupby(level=0).sum(); ser[c] = d; ise = d.index < "2020-01-01"; t = pd.DataFrame(TR[c], columns=["year", "a", "b", "edate", "days", "dir", "net"])
        oo = t[t.year >= 2020].net; rows.append(dict(cfg=str(c), IS=R.sharpe(d[ise]), OOS=R.sharpe(d[~ise]), IS_net=d[ise].sum(), OOS_net=d[~ise].sum(),
                     OOS_MDD=R.stats(d[~ise])["MDD_close"], trades=len(t), win=(t.net > 0).mean(), avg_days=t.days.mean(),
                     OOS_top10=oo.nlargest(10).sum() / oo.sum() if oo.sum() > 0 else np.nan, short_A_share=(t.dir < 0).mean()))
        yr = d.groupby(d.index.year).sum() / 1e4; print(c, "yearly 萬:", yr.round(0).to_dict())
    G = pd.DataFrame(rows); print(G.round(3).to_string()); G.to_csv(R.OUT / "rq66_summary.csv", index=False)
    for m in ("kal", "cop"):
        g = G[G.cfg.str.contains(m)]; print(m, "Spearman IS/OOS %.3f" % g.IS.rank().corr(g.OOS.rank()))
    rd = pd.concat(RND).groupby(level=0).sum(); print(f"random baseline IS {R.sharpe(rd[rd.index<'2020'])::.2f} OOS {R.sharpe(rd[rd.index>='2020']):.2f}" if False else
          f"random baseline IS {R.sharpe(rd[rd.index < '2020-01-01']):.2f} OOS {R.sharpe(rd[rd.index >= '2020-01-01']):.2f}")
    for m in ("kal", "cop"):
        wf = []
        for Y in range(2018, 2027):
            cs = [c for c in cfgs if c[0] == m]; sc = {c: R.sharpe(ser[c][(ser[c].index.year >= Y - 3) & (ser[c].index.year < Y)]) for c in cs}
            b = max(sc, key=sc.get); wf.append(ser[b][ser[b].index.year == Y])
        wf = pd.concat(wf); print(m, f"WF stitched Sharpe 2018+ {R.sharpe(wf):.3f} 2020+ {R.sharpe(wf[wf.index >= '2020-01-01']):.3f}")
    plt = R.style(); fig, ax = plt.subplots(figsize=(14, 7))
    for c in cfgs: ax.plot(ser[c].cumsum() / 1e4, label=str(c))
    ax.plot(rd.cumsum() / 1e4, label="random", ls="--", c=R.GRAY); ax.axvline(pd.Timestamp("2020-01-01"), c=R.GRAY, ls="--")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("台股配對：Kalman vs Copula（20 對，每腿 100 萬，萬元）"); fig.tight_layout(); fig.savefig(R.OUT / "rq_66_pairs.png", dpi=100)
    print("chart results/rq_66_pairs.png")


if __name__ == "__main__":
    main()
