"""74 月選 OTM put 賣方深度回測（承 70）：履約價規則 × 進場時點 × 尾部避險寬度 × 停損。

時序：訊號日 s 收盤（TXF 收盤 F_s、TXO 收盤算 IV/delta 選履約價）→ s 的下一交易日開盤成交；持有至目標月選結算日 X
  （結算價≈X 日 TXF 收盤）。單一部位，週期不重疊。
履約價：moneyness x∈1..8%（K≈F_s(1-x)）或 |delta|∈{.10,.15,.20,.25,.30}（Black-76, r=0, F=TXF 收盤, T=日曆日/365）。
進場：E+1（前一月選結算翌日，約 20 交易日）或結算前 15/10/5 交易日。
避險：買更價外 put，K_l≈K−w·F_s，w∈{0(裸賣),2%,4%,6%}；保證金近似 = 履約價差×50。
停損（收盤判定、次日開盤平倉）：none｜價值≥2/3/4 倍收取權利金｜TXF 收盤跌破 K｜跌破 (F0+K)/2。
成本：每腿 30 元 + 0.1%×權利金 + 1 tick（進/出場）；結算價內每腿再 30 元 + 0.1%×內含值。
MDD 盤中最差：short 用 day_high、long 用 day_low（無成交以 TXF low/high 內含值估），價差上限=寬度。
IS 2017–2021 選參（另列 2010–16 pre-sample）、OOS 2022+；Spearman、逐年 WF（擴張窗，2010 起）、隨機 config 基準。
輸出：results/dd_put_74_grid.csv、dd_put_74_*.png；交易快取 data/cache/dd_put_trades.pkl（不入 repo）。
"""
import itertools, pickle
import numpy as np, pandas as pd
from scipy.stats import norm
from _rqo import *

END = pd.Timestamp("2026-10-21"); PRE = pd.Timestamp("2010-01-01"); IS0 = pd.Timestamp("2017-01-01")
CACHE = ROOT / "data/cache/dd_put_trades.pkl"
RULES = [("x", v) for v in (.01, .02, .03, .04, .05, .06, .07, .08)] + [("d", v) for v in (.10, .15, .20, .25, .30)]
ENTRIES = ["E+1", 15, 10, 5]
WIDTHS = [0, .02, .04, .06]
STOPS = ["none", "v2", "v3", "v4", "brK", "br50"]
rng = np.random.default_rng(11)


def b76_put(p, F, K, T):
    lo, hi = np.full(len(p), 1e-3), np.full(len(p), 3.0); sq = np.sqrt(T)
    for _ in range(50):
        m = (lo + hi) / 2; d1 = (np.log(F / K) + .5 * m * m * T) / (m * sq); d2 = d1 - m * sq
        v = K * norm.cdf(-d2) - F * norm.cdf(-d1); lo = np.where(v < p, m, lo); hi = np.where(v >= p, m, hi)
    s = (lo + hi) / 2; d1 = (np.log(F / K) + .5 * s * s * T) / (s * sq)
    return s, -norm.cdf(-d1)


def leg_path(pv, K, hold, Fc, Fl, Fh, short):
    """回傳 o,h,l,c（填補後）numpy。"""
    get = lambda f: pv[f][K].reindex(hold).values.astype(float) if K in pv[f].columns else np.full(len(hold), np.nan)
    o, h, l, c = get("day_open"), get("day_high"), get("day_low"), get("day_close")
    ic, il, ih = np.maximum(K - Fc, 0), np.maximum(K - Fl, 0), np.maximum(K - Fh, 0)
    last = np.nan
    for i in range(len(hold)):
        if np.isnan(c[i]): c[i] = max(last, ic[i]) if not np.isnan(last) else ic[i]
        last = c[i]
    c[-1] = ic[-1]
    o = np.where(np.isnan(o), c, o)
    h = np.where(np.isnan(h), np.maximum(c, il), h); h = np.maximum(h, c)
    l = np.where(np.isnan(l), np.minimum(c, ih), l); l = np.minimum(l, c)
    return o, h, l, c


def build():
    F = load_fut("2009-01-01"); days = F.index[F.index <= END]
    E = expiries(); M = E[E.kind == "monthly"].sort_values("expiry_date")
    M = M[(M.expiry_date <= END) & (M.expiry_date >= PRE)]
    allm = E[E.kind == "monthly"].sort_values("expiry_date").reset_index(drop=True)
    O0 = load_opt("2009-10-01", END, cp="P")
    O0 = O0[O0.expiry.isin(M.expiry)]
    T = []
    for X, xd in zip(M.expiry, M.expiry_date):
        if xd not in days: continue
        g = O0[O0.expiry == X]
        pv = {f: g.pivot_table(index="date", columns="strike", values=f) for f in ("day_open", "day_high", "day_low", "day_close")}
        prev = allm[allm.expiry_date < xd].expiry_date
        if prev.empty: continue
        ix = days.searchsorted(xd)
        for en in ENTRIES:
            s = prev.iloc[-1] if en == "E+1" else days[ix - en - 1]
            if s not in days: continue
            si = days.get_loc(s); d1 = days[si + 1]; hold = days[si + 1: ix + 1]
            if s not in pv["day_close"].index or d1 not in pv["day_close"].index: continue
            Fs = F.close[s]; cl = pv["day_close"].loc[s].dropna(); cl = cl[(cl.index < Fs) & (cl > 0.5)]
            if len(cl) < 3: continue
            Tt = max((xd - s).days, 1) / 365
            iv, dl = b76_put(cl.values, np.full(len(cl), Fs), cl.index.values.astype(float), np.full(len(cl), Tt))
            ok = iv < 2.9
            atm = cl.index[np.abs(cl.index - Fs).argmin()]
            Fc, Fl, Fh = F.close[hold].values, F.low[hold].values, F.high[hold].values
            for rule in RULES:
                if rule[0] == "x": K = cl.index[np.abs(cl.index - Fs * (1 - rule[1])).argmin()]
                else:
                    if ok.sum() == 0: continue
                    K = cl.index[ok][np.abs(np.abs(dl[ok]) - rule[1]).argmin()]
                ps = pv["day_open"].at[d1, K] if K in pv["day_open"].columns else np.nan
                if np.isnan(ps): ps = pv["day_close"].at[d1, K] if K in pv["day_close"].columns else np.nan
                if np.isnan(ps) or ps < 1: continue
                so, sh, sl, sc = leg_path(pv, K, hold, Fc, Fl, Fh, True)
                for w in WIDTHS:
                    if w == 0: lo_ = (0, None, np.zeros(len(hold)), np.zeros(len(hold)), np.zeros(len(hold)), np.zeros(len(hold)))
                    else:
                        cand = pv["day_close"].columns[pv["day_close"].columns < K]
                        if len(cand) == 0: continue
                        Kl = cand[np.abs(cand - (K - w * Fs)).argmin()]
                        pl = pv["day_open"].at[d1, Kl] if d1 in pv["day_open"].index else np.nan
                        if np.isnan(pl): pl = pv["day_close"].at[d1, Kl]
                        if np.isnan(pl): continue
                        lo_ = (pl, Kl) + leg_path(pv, Kl, hold, Fc, Fl, Fh, False)
                    T.append(dict(X=X, xd=xd, entry=en, rule=rule, w=w, s=s, d1=d1, hold=hold, Fs=Fs, K=K, Kl=lo_[1], ps=ps, pl=lo_[0],
                                  so=so, sh=sh, sc=sc, lo=lo_[2], ll=lo_[4], lc=lo_[5], Fc=Fc, Fl=Fl, atm_iv=float(iv[cl.index == atm][0]) if (cl.index == atm).any() else np.nan,
                                  iv_k=float(iv[cl.index == K][0]), dte_cal=(xd - s).days))
        print(X, len(T), flush=True) if X.endswith("12") else None
    pickle.dump(T, open(CACHE, "wb")); return T


def sim(t, stop):
    """回傳 (net 點, 日損益點陣列, 盤中最差(相對收盤)陣列, 持有天數, 出場原因)。單位 1 組。"""
    C0 = t["ps"] - (t["pl"] or 0); hedged = t["w"] > 0
    cin = opt_cost_pts(t["ps"]) + (opt_cost_pts(t["pl"]) if hedged else 0)
    vc = t["sc"] - t["lc"]; vw = t["sh"] - t["ll"]
    if hedged: vw = np.minimum(vw, t["K"] - t["Kl"])
    vw = np.maximum(vw, vc); n = len(vc); ex = n - 1; why = "settle"
    if stop != "none":
        if stop[0] == "v": trig = vc >= float(stop[1]) * C0
        elif stop == "brK": trig = t["Fc"] < t["K"]
        else: trig = t["Fc"] < (t["Fs"] + t["K"]) / 2
        trig = trig[:-1]; hit = np.flatnonzero(trig)
        if len(hit): ex = hit[0] + 1; why = "stop"
    pnl = np.empty(ex + 1); prev = C0
    for i in range(ex + 1):
        pnl[i] = prev - vc[i]; prev = vc[i]
    worst = vc[:ex + 1] - vw[:ex + 1]
    pnl[0] -= cin
    if why == "stop":
        vo = t["so"][ex] - t["lo"][ex]; pnl[ex] = vc[ex - 1] - vo - opt_cost_pts(t["so"][ex]) - (opt_cost_pts(max(t["lo"][ex], .1)) if hedged else 0); worst[ex] = 0
    else:
        iS = max(t["K"] - t["Fc"][-1], 0); iL = max(t["Kl"] - t["Fc"][-1], 0) if hedged else 0
        pnl[ex] -= (OCOMM / MULT + OTAX * iS if iS > 0 else 0) + (OCOMM / MULT + OTAX * iL if iL > 0 else 0)
    return pnl.sum(), pnl, worst, ex + 1, why


def key_of(t): return (f"{t['rule'][0]}{t['rule'][1]:.2f}", str(t["entry"]), t["w"])


def daily(TT, stop, days, mask=None):
    pc = pd.Series(0.0, days); wc = pd.Series(0.0, days); rec = []
    for i, t in enumerate(TT):
        if mask is not None and not mask[i]: continue
        net, p, w, L, why = sim(t, stop); idx = t["hold"][:L]
        pc[idx] += p * MULT; wc[idx] = np.minimum(wc[idx].values, w * MULT)
        rec.append(dict(s=t["s"], X=t["X"], net=net * MULT, why=why, C0=t["ps"] - (t["pl"] or 0), K=t["K"], Fs=t["Fs"]))
    return pc, wc, pd.DataFrame(rec)


def pstats(pc, wc, rec, a, b):
    m = (pc.index >= a) & (pc.index < b); p = pc[m]; w = wc[m]
    eq = p.cumsum(); prev = eq.shift(fill_value=0)
    r = rec[(rec.s >= a) & (rec.s < b)] if len(rec) else rec
    return dict(n=len(r), net=p.sum(), Sharpe=sharpe(p), MDDw=float((prev + w - eq.cummax().shift(fill_value=0).clip(lower=0)).min()) if len(p) else 0,
                win=(r.net > 0).mean() if len(r) else np.nan, worst=r.net.min() if len(r) else np.nan, stops=(r.why == "stop").mean() if len(r) else np.nan)


PER = (("PRE", PRE, IS0), ("IS", IS0, IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1)))


def main():
    T = pickle.load(open(CACHE, "rb")) if CACHE.exists() else build()
    F = load_fut("2009-01-01"); days = F.index[(F.index <= END) & (F.index >= PRE)]
    groups = {}
    for t in T: groups.setdefault(key_of(t), []).append(t)
    rows, series = [], {}
    for k, TT in groups.items():
        for st in STOPS:
            pc, wc, rec = daily(TT, st, days); series[k + (st,)] = pc
            r = dict(rule=k[0], entry=k[1], w=k[2], stop=st)
            for nm, a, b in PER:
                for kk, v in pstats(pc, wc, rec, a, b).items(): r[f"{nm}_{kk}"] = v
            rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(RES / "dd_put_74_grid.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    Q = R[R.IS_n >= 30].copy()
    print("configs", len(R), "qualified", len(Q))
    sp = Q.IS_Sharpe.rank().corr(Q.OOS_Sharpe.rank()); print(f"Spearman IS/OOS Sharpe (全 grid) {sp:.3f}")
    cols = ["rule", "entry", "w", "stop", "PRE_Sharpe", "IS_n", "IS_net", "IS_Sharpe", "IS_MDDw", "OOS_n", "OOS_net", "OOS_Sharpe", "OOS_MDDw", "OOS_win", "OOS_worst", "OOS_stops"]
    print("\n== IS Top 15 =="); print(Q.sort_values("IS_Sharpe", ascending=False).head(15)[cols].round(2).to_string(index=False))
    for dim in ("rule", "entry", "w", "stop"):
        print(f"\n== 邊際平均：{dim} =="); print(Q.groupby(dim)[["PRE_Sharpe", "IS_Sharpe", "OOS_Sharpe", "IS_MDDw", "OOS_MDDw", "OOS_net"]].mean().round(2).to_string())
    print("\n== 裸賣 E+1 無停損（基準）各履約價 =="); print(R[(R.entry == "E+1") & (R.w == 0) & (R.stop == "none")][cols].round(2).to_string(index=False))
    best = Q.loc[Q.IS_Sharpe.idxmax()]; bk = (best.rule, best.entry, best.w, best.stop)
    rnd = Q.OOS_Sharpe.values; print(f"\nIS 最佳 {bk} OOS Sharpe {best.OOS_Sharpe:.2f}；隨機 config OOS Sharpe 百分位 {(rnd < best.OOS_Sharpe).mean():.2f} 中位 {np.median(rnd):.2f}")
    # 穩健選擇：IS 前 10% 中 PRE 也 >0 的平均排名
    # WF：每年用 2010..Y-1 擴張窗選最佳（需該窗 ≥30 筆交易 → 用 n 近似）
    wf, picks = [], []
    ok = set(Q.apply(lambda r: (r.rule, r.entry, r.w, r.stop), axis=1))
    for Y in range(2015, 2027):
        a, b = PRE, pd.Timestamp(f"{Y}-01-01"); c = pd.Timestamp(f"{Y+1}-01-01")
        sc = {k: sharpe(s[(s.index >= a) & (s.index < b)]) for k, s in series.items() if k in ok}
        kb = max(sc, key=sc.get); s = series[kb]; wf.append(s[(s.index >= b) & (s.index < c)]); picks.append((Y, kb, round(sc[kb], 2)))
    wf = pd.concat(wf); wo = wf[wf.index >= IS_END]
    print("WF picks:"); [print(" ", p) for p in picks]
    print(f"WF 2022+ Sharpe {sharpe(wo):.2f} net {wo.sum():.0f}")
    pickle.dump(dict(R=R, best=bk, wf=wf, picks=picks, series={k: series[k] for k in [bk, ("x0.03", "E+1", 0, "none")]}), open(ROOT / "data/cache/dd_put_74_summary.pkl", "wb"))
    chart(R, Q, series, bk, wf)


def chart(R, Q, series, bk, wf):
    plt = plt_style()
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    ax = axs[0, 0]; sty(ax)
    for k, c, lb in ((bk, GOLD, f"IS 最佳 {bk}"), (("x0.03", "E+1", 0, "none"), GR, "基準 x=3% 裸賣 E+1 無停損"), (("x0.05", "E+1", 0.04, "none"), "#e07b39", "x=5% + 4% 價差")):
        if k in series: ax.plot(series[k].cumsum() / 1e4, color=c, lw=1.5, label=lb)
    ax.plot(wf.cumsum() / 1e4, color=BLUE, lw=1.2, label="walk-forward")
    for x in (IS0, IS_END): ax.axvline(x, color="white", ls="--", lw=.8)
    ax.set_title("累積淨損益（萬元，1 組）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=8)
    ax = axs[0, 1]; sty(ax)
    ax.scatter(Q.IS_Sharpe, Q.OOS_Sharpe, s=8, color=GOLD, alpha=.5); ax.axhline(0, color=GR); ax.axvline(0, color=GR)
    ax.set_xlabel("IS Sharpe"); ax.set_ylabel("OOS Sharpe"); ax.set_title(f"全 grid IS vs OOS（Spearman {Q.IS_Sharpe.rank().corr(Q.OOS_Sharpe.rank()):.2f}）")
    ax = axs[1, 0]; sty(ax)
    b = R[(R.entry == "E+1") & (R.stop == "none") & R.rule.str.startswith("x")]
    for w, c in zip(WIDTHS, (GOLD, "#e07b39", BLUE, GR)):
        s = b[b.w == w].sort_values("rule"); ax.plot(s.rule.str[1:].astype(float) * 100, s.OOS_Sharpe, "o-", color=c, label=f"OOS 寬度 {w:.0%}" if w else "OOS 裸賣")
        ax.plot(s.rule.str[1:].astype(float) * 100, s.IS_Sharpe, "x--", color=c, alpha=.6)
    ax.set_xlabel("價外 %"); ax.set_ylabel("Sharpe（實線 OOS、虛線 IS）"); ax.set_title("E+1 進場、無停損：價外程度 × 避險寬度"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=8)
    ax = axs[1, 1]; sty(ax)
    g = Q.groupby("stop")[["IS_Sharpe", "OOS_Sharpe"]].mean().reindex(STOPS)
    xx = np.arange(len(g)); ax.bar(xx - .2, g.IS_Sharpe, .4, color=GR, label="IS"); ax.bar(xx + .2, g.OOS_Sharpe, .4, color=GOLD, label="OOS")
    ax.set_xticks(xx); ax.set_xticklabels(["無", "2倍", "3倍", "4倍", "破K", "破半"], color=WH); ax.set_title("停損規則平均 Sharpe"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "dd_put_74_grid.png", dpi=110, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
