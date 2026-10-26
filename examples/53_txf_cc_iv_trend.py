"""53 台指期 covered call：隱含波動率濾網 + 趨勢濾網，walk-forward 驗證。

沿用 45/51：收盤訊號、隔日開盤成交、同成本、盤中最差 MDD（期貨低 + call 高）、1 口 TXF 完全掩護。
IV：每日以近月月選（到期日 > 當日；剩 <7 天則用次月）ATM 履約價（同時有 C/P 收盤、最接近 TXF 收盤）
  的 call、put day_close，用 Black-76（標的=TXF 收盤，r=0）反推 IV 後平均。
濾網（訊號日 E 收盤）：
  IV：none / spread=IV−RV20 > {0,2,4} 波動點 / IV/RV20 > {1.0,1.2,1.4} / IV 在自身過去 252 日分位 > {30,50,70}
  趨勢：none / close>MA20、close>MA60、mom20>0、mom60>0 時 → skip 不賣 或 改賣 10% 價外
  基本 OTM：{5,7,10}%
驗證：(1) 固定 IS 2010-19 → OOS 2020+，Spearman；(2) walk-forward：2015 起每年以前 5 年 Sharpe 選參數、交易下一年、串接。
"""
import importlib.util, math
from pathlib import Path
import numpy as np, pandas as pd

def _load(n, f):
    s = importlib.util.spec_from_file_location(n, Path(__file__).with_name(f)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cc = _load("cc45", "45_txf_covered_call.py"); vf = _load("vf51", "51_txf_cc_volfilter.py")
from tools.optdaily import load_txo_daily, expiries
OUT = Path("results"); RF = cc.RF
IS_, OOS_ = cc.PERIODS[1], cc.PERIODS[2]


def ncdf(x):
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(x / math.sqrt(2)))


def b76(F, K, T, s, cp):
    d1 = (np.log(F / K) + .5 * s * s * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T)
    return np.where(cp == "C", F * ncdf(d1) - K * ncdf(d2), K * ncdf(-d2) - F * ncdf(-d1))


def implied(px, F, K, T, cp):
    lo = np.full(len(px), 1e-3); hi = np.full(len(px), 3.0)
    for _ in range(50):
        mid = (lo + hi) / 2; v = b76(F, K, T, mid, cp)
        hi = np.where(v > px, mid, hi); lo = np.where(v > px, lo, mid)
    iv = (lo + hi) / 2
    intrinsic = np.maximum(np.where(cp == "C", F - K, K - F), 0)
    return np.where((px > intrinsic + 0.05) & (iv < 2.9), iv, np.nan)


def daily_iv(full):
    o = load_txo_daily(cc.START - pd.Timedelta(days=40), cc.END, expiry="monthly",
                       columns=["date", "expiry", "strike", "cp", "day_close"])
    o["date"] = pd.to_datetime(o.date).dt.normalize(); o = o.dropna(subset=["day_close"])
    e = expiries(); e = e[e.kind == "monthly"].sort_values("expiry_date")
    ed = e.expiry_date.values; ex = e.expiry.values
    days = full.index[(full.index >= o.date.min())]
    pick = {}
    for d in days:
        j = np.searchsorted(ed, np.datetime64(d), side="right")
        if j < len(ed) and (ed[j] - np.datetime64(d)) / np.timedelta64(1, "D") < 7: j += 1
        if j < len(ed): pick[d] = (ex[j], pd.Timestamp(ed[j]))
    P = pd.DataFrame([(d, a, b) for d, (a, b) in pick.items()], columns=["date", "expiry", "exp_date"])
    o = o.merge(P, on=["date", "expiry"])
    w = o.pivot_table(index=["date", "expiry", "exp_date", "strike"], columns="cp", values="day_close").dropna().reset_index()
    w["F"] = full.close.reindex(w.date).values
    w["dist"] = (w.strike - w.F).abs()
    a = w.sort_values("dist").groupby("date").head(1).copy()
    a["T"] = ((a.exp_date - a.date).dt.days + 0.5) / 365
    ivc = implied(a.C.values, a.F.values, a.strike.values, a["T"].values, np.array(["C"] * len(a)))
    ivp = implied(a.P.values, a.F.values, a.strike.values, a["T"].values, np.array(["P"] * len(a)))
    a["iv"] = np.nanmean(np.vstack([ivc, ivp]), axis=0)
    return a.set_index("date").iv.sort_index()


def main():
    OUT.mkdir(exist_ok=True)
    F = cc.load_fut(); O = cc.load_opt(); cycles, optdays = cc.build_cycles(F, O)
    full = pd.read_parquet(cc.DATA); full.index = full["date"].dt.tz_localize(None).dt.normalize(); full = full[full.index <= cc.END]
    cm = full.contractMonth.astype(str); roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    r = np.log(full.close / full.close.shift()); r[roll] = np.log(full.close / full.open)[roll]
    rv20 = r.rolling(20).std() * math.sqrt(252); rvmed = rv20.rolling(252, min_periods=252).median()
    iv = daily_iv(full)
    print("IV days", iv.notna().sum(), "of", len(iv), iv.index.min().date(), "median IV", round(iv.median(), 3))
    ivpct = iv.rolling(252, min_periods=200).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    # 趨勢（用 log 報酬累積避免換月跳空）
    lc = r.fillna(0).cumsum()
    ma20 = lc.rolling(20).mean(); ma60 = lc.rolling(60).mean()
    S = pd.DataFrame(dict(rv=rv20, rvmed=rvmed, iv=iv.reindex(full.index), ivpct=ivpct.reindex(full.index),
                          up20=lc > ma20, up60=lc > ma60, m20=lc - lc.shift(20), m60=lc - lc.shift(60)))
    S["spread"] = S.iv - S.rv; S["ratio"] = S.iv / S.rv
    for cy in cycles: cy["s"] = S.loc[cy["E"]]
    sig = pd.DataFrame([dict(expiry=cy["expiry"], E=cy["E"], **cy["s"].to_dict()) for cy in cycles])
    print("signal days with IV:", sig.iv.notna().sum(), "/", len(sig), "| ivpct:", sig.ivpct.notna().sum())

    IVF = [("none", None)] + [(f"spread>{t}", ("spread", t / 100)) for t in (0, 2, 4)] + \
          [(f"ratio>{t}", ("ratio", t)) for t in (1.0, 1.2, 1.4)] + [(f"ivpct>{t}", ("ivpct", t)) for t in (30, 50, 70)]
    TRF = [("none", None, None)] + [(f"{k}:{act}", k, act) for k in ("up20", "up60", "m20>0", "m60>0") for act in ("skip", "otm10")]
    def trend_on(s, k):
        return bool(s[k]) if k.startswith("up") else (s[k[:3]] > 0 if not np.isnan(s[k[:3]]) else False)
    def decide(cy, otm, ivf, trk, act):
        s = cy["s"]
        if ivf is not None:
            v = s[ivf[0]]
            if np.isnan(v) or not v > ivf[1]: return None
        if trk is not None and trend_on(s, trk):
            return None if act == "skip" else max(otm, 10)
        return otm
    def run(fn):
        def pick(Fd, Od, cyc, mode, x, optdays_, log):
            o_ = fn(cyc)
            return None if o_ is None else vf.make_pick(lambda c: True)(Fd, Od, cyc, mode, o_ / 100, optdays_, log)
        cc.pick_leg = pick
        return cc.simulate(F, O, cycles, optdays, ("x", "pct", 0), 1.0, [])
    res = {}; meta = {}
    res["B&H"] = cc.simulate(F, O, cycles, optdays); meta["B&H"] = dict(kind="bench")
    res["RVfilter 5%"] = run(lambda cy: 5 if (not np.isnan(cy["s"].rvmed) and cy["s"].rv > cy["s"].rvmed) else None); meta["RVfilter 5%"] = dict(kind="bench")
    for otm in (5, 7, 10):
        for inm, ivf in IVF:
            for tnm, trk, act in TRF:
                if otm == 10 and act == "otm10": continue
                nm = f"otm{otm}|{inm}|{tnm}"
                res[nm] = run(lambda cy, otm=otm, ivf=ivf, trk=trk, act=act: decide(cy, otm, ivf, trk, act))
                meta[nm] = dict(kind="grid", otm=otm, iv=inm, trend=tnm)
    print("variants", len(res))
    R = pd.DataFrame({k: v[0].pct_change() for k, v in res.items()}).dropna(how="all")

    def metr(k, lo, hi):
        e, w, L, w2 = res[k]; st = cc.stats(e, w, lo, hi, w2)
        n = int(((L.settle >= lo) & (L.settle < hi)).sum()) if len(L) else 0
        return dict(CAGR=st["CAGR"], vol=st["vol"], Sharpe=st["Sharpe"], MDD=st["MDD_intraday"], months=n)
    rows = []
    for k in res:
        row = dict(name=k, **meta[k])
        for pn, lo, hi in cc.PERIODS: row.update({f"{pn.split()[0]}_{m}": v for m, v in metr(k, lo, hi).items()})
        rows.append(row)
    T = pd.DataFrame(rows); T.to_csv(OUT / "cc_ivtrend_grid.csv", index=False)
    G = T[T.kind == "grid"].copy()
    sp = lambda a, b: a.rank().corr(b.rank())
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    pc = lambda d: d.assign(**{c: (d[c] * 100).round(2) for c in d.columns if c.endswith(("CAGR", "MDD", "vol"))}).round(3)
    cols = ["name", "IS_CAGR", "IS_Sharpe", "IS_MDD", "IS_months", "OOS_CAGR", "OOS_Sharpe", "OOS_MDD", "OOS_months"]
    G["OOS_rank"] = G.OOS_Sharpe.rank(ascending=False, method="min").astype(int)
    print("\n== benchmarks =="); print(pc(T[T.name.isin(["B&H", "RVfilter 5%", "otm5|none|none", "otm10|none|none"])][cols + ["FULL_CAGR", "FULL_Sharpe", "FULL_MDD"]]).to_string(index=False))
    print("\n== IS top10 (Sharpe) =="); print(pc(G.sort_values("IS_Sharpe", ascending=False).head(10)[cols + ["OOS_rank"]]).to_string(index=False))
    print(f"\nSpearman IS/OOS Sharpe all {sp(G.IS_Sharpe, G.OOS_Sharpe):.3f} | CAGR {sp(G.IS_CAGR, G.OOS_CAGR):.3f}  N={len(G)}")
    bh = T.set_index("name").loc["B&H"]
    print("beat B&H Sharpe IS %.3f OOS %.3f" % ((G.IS_Sharpe > bh.IS_Sharpe).mean(), (G.OOS_Sharpe > bh.OOS_Sharpe).mean()))
    print("\n== 平均 by IV filter =="); print(G.groupby("iv")[["IS_Sharpe", "OOS_Sharpe", "IS_CAGR", "OOS_CAGR", "IS_months", "OOS_months"]].mean().round(3))
    print("\n== 平均 by trend filter =="); print(G.groupby("trend")[["IS_Sharpe", "OOS_Sharpe", "IS_CAGR", "OOS_CAGR", "IS_months", "OOS_months"]].mean().round(3))
    # 單獨 vs 組合（otm5 基底）
    sub = G[(G.otm == 5) & (G.iv.isin(["none", "spread>2", "ratio>1.2", "ivpct>50"])) & (G.trend.isin(["none", "up60:skip", "up60:otm10", "m20>0:skip"]))]
    print("\n== otm5：IV 單獨 / 趨勢單獨 / 組合 =="); print(pc(sub[cols + ["OOS_rank"]]).to_string(index=False))

    # IV−RV 對短 call 損益的預測力（always 5% legs）
    L = res["otm5|none|none"][2].merge(sig[["expiry", "spread", "ratio", "ivpct"]], on="expiry")
    L["ret"] = L.pnl_pts / L.F_sig * 100
    print("\n== IV−RV spread → 短 call 損益(%F) ==")
    for pn, lo, hi in (IS_, OOS_, cc.PERIODS[0]):
        d = L[(L.settle >= lo) & (L.settle < hi)].dropna(subset=["spread"])
        q = pd.qcut(d.spread, 5, labels=False)
        print(pn, "N", len(d), "Pearson %.3f Spearman %.3f" % (d.spread.corr(d.ret), sp(d.spread, d.ret)))
        print(d.groupby(q).agg(spread=("spread", "mean"), n=("ret", "size"), mean_ret=("ret", "mean"), win=("ret", lambda x: (x > 0).mean()), itm=("itm", "mean")).round(3).to_string())

    # walk-forward
    names = list(G.name); wf = []; chosen = []
    for Y in range(2015, cc.END.year + 1):
        tr = R.loc[f"{Y-5}-01-01":f"{Y-1}-12-31", names]
        x = tr - RF / 252; sh = x.mean() / x.std() * math.sqrt(252); best = sh.idxmax()
        te = R.loc[f"{Y}-01-01":f"{Y}-12-31", best]; wf.append(te); chosen.append(dict(year=Y, pick=best, train_sharpe=round(sh.max(), 3)))
    wfr = pd.concat(wf); print("\n== walk-forward picks =="); print(pd.DataFrame(chosen).to_string(index=False))
    def rstat(rr):
        e = (1 + rr.fillna(0)).cumprod(); y = len(rr) / 252; x = rr - RF / 252
        return dict(CAGR=e.iloc[-1] ** (1 / y) - 1, vol=rr.std() * math.sqrt(252), Sharpe=x.mean() / x.std() * math.sqrt(252), MDD_close=(e / e.cummax() - 1).min())
    idx = wfr.index
    rows = [dict(name="WF stitched", **rstat(wfr))]
    for k in ("B&H", "otm5|none|none", "otm10|none|none", "RVfilter 5%"):
        rows.append(dict(name=k, **rstat(R.loc[idx, k])))
    k_ = rstat(wfr)["vol"] / rstat(R.loc[idx, "B&H"])["vol"]
    rows.append(dict(name=f"B&H vol-matched (x{k_:.2f} + cash)", **rstat(k_ * R.loc[idx, "B&H"] + (1 - k_) * RF / 252)))
    beta = np.cov(wfr, R.loc[idx, "B&H"])[0, 1] / R.loc[idx, "B&H"].var()
    W = pd.DataFrame(rows); print(f"\n== walk-forward 2015-01 ~ {cc.END.date()} (beta of WF vs B&H {beta:.2f}) =="); print(pc(W).to_string(index=False))
    # 盤中最差 MDD for WF: 以各年選中變體之 worst/close 比例近似
    wq = []; 
    for c in chosen:
        e, w, _, _ = res[c["pick"]]; s = slice(f"{c['year']}-01-01", f"{c['year']}-12-31"); wq.append((w[s] / e[s]))
    wratio = pd.concat(wq); ewf = (1 + wfr).cumprod(); ww = ewf * wratio.reindex(ewf.index).values / (1 + wfr)  # worst relative to prev close equity... approx
    print("WF MDD intraday ≈ %.2f%%" % (((ewf.shift().fillna(1) * (wratio.reindex(ewf.index) * (1 + wfr)) / ewf.shift().fillna(1).cummax()).min() - 1) * 100 if False else ((ewf * wratio.reindex(ewf.index).values / ewf.cummax().shift().fillna(1)).min() - 1) * 100))
    pd.DataFrame(chosen).to_csv(OUT / "cc_ivtrend_wf_picks.csv", index=False); S.to_csv(OUT / "cc_ivtrend_signals.csv")
    chart(R, wfr, iv, rv20, G, idx)


def chart(R, wfr, iv, rv20, G, idx):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw=dict(height_ratios=[3, 1.5]), sharex=True)
    e = (1 + wfr).cumprod(); ax[0].plot(e.index, e, lw=1.8, label="Walk-forward (5y Sharpe pick, yearly)")
    for k in ("B&H", "otm5|none|none", "otm10|none|none", "RVfilter 5%"):
        x = (1 + R.loc[idx, k].fillna(0)).cumprod(); ax[0].plot(x.index, x, lw=1, label=k)
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Walk-forward OOS equity 2015+ (log)")
    for k, s in (("WF", e), ("B&H", (1 + R.loc[idx, "B&H"].fillna(0)).cumprod())):
        ax[1].plot(s.index, (s / s.cummax() - 1) * 100, lw=.8, label=k)
    ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title("Close drawdown %")
    fig.tight_layout(); fig.savefig(OUT / "cc_ivtrend_wf_equity.png", dpi=110)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(iv.index, iv * 100, lw=.7, label="ATM IV (Black-76)"); ax[0].plot(rv20.index, rv20 * 100, lw=.7, label="RV20"); ax[0].legend(); ax[0].set_ylim(0, 80); ax[0].grid(alpha=.3); ax[0].set_title("TXO ATM IV vs RV20 (%)")
    for t, c in (("none", "k"),):
        pass
    sc = ax[1].scatter(G.IS_Sharpe, G.OOS_Sharpe, c=G.otm, cmap="viridis", s=14); fig.colorbar(sc, ax=ax[1], label="base OTM%")
    ax[1].set_xlabel("IS Sharpe 2010-19"); ax[1].set_ylabel("OOS Sharpe 2020+"); ax[1].grid(alpha=.3); ax[1].set_title("IS vs OOS Sharpe (grid)")
    fig.tight_layout(); fig.savefig(OUT / "cc_ivtrend_iv_scatter.png", dpi=110)
    print("charts results/cc_ivtrend_wf_equity.png results/cc_ivtrend_iv_scatter.png")


if __name__ == "__main__":
    main()
