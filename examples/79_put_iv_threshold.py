"""79 月選 put 賣方：只在隱含波動率高於門檻時進場（承 76/77 引擎與交易快取）。

IV：_rqo.load_atm_iv() 的 atm_iv_30d（ok 旗標），取訊號日 s（前一月選結算日）收盤值，無前視；IV 資料止於 2026-08-17，
  之後的交易（s > IV 末日）不納入，OOS 截至最後一筆納入交易的結算日。
濾網（濾網說否 → 該月空手）：
  絕對 IV ≥ {none,10,12,14,16,18,20,22,25,30}%；IV 近 252 日百分位（min 60 日）≥ {20,40,50,60,80}；IV / RV20 ≥ {1.0,1.1,1.2,1.3}。
  RV20 = TXF 換月調整日報酬 20 日標準差 ×√252（至 s 收盤）。百分位不足 60 日 → 視為通過（2017 年初）。
結構：價差 x2% w4% SL2x、x3% w4% SL2x、x2% w2% SL2x；裸賣 x2%、x3%、ATM 無停損。
Sharpe：月損益序列（含空手月 = 0）×√12。選參：IS 2017–2021 Sharpe 最高且 IS 交易 ≥ 20；OOS 2022+。
驗證：Spearman IS→OOS（門檻間）；隨機略過同數月 500 次；逐年 WF（擴張窗自 2017，每年重選門檻）；IV 分組。
輸出：results/dd_put_iv_{grid,random,wf,bucket,trades}.csv、dd_put_iv.png；不含 TAIFEX 原始資料。
"""
import importlib.util, pickle, sys
import numpy as np, pandas as pd
from _rqo import *
spec = importlib.util.spec_from_file_location("m76", Path(__file__).parent / "76_put_tp_sl.py"); m = importlib.util.module_from_spec(spec); sys.modules["m76"] = m; spec.loader.exec_module(m)

CACHE = ROOT / "data/cache/dd_put_strike_trades.pkl"; IS0 = m.IS0
STRS = [(("x", .02, .04), 2.0), (("x", .03, .04), 2.0), (("x", .02, .02), 2.0), (("x", .02, 0), None), (("x", .03, 0), None), (("x", 0.0, 0), None)]
FILTERS = [("none", None)] + [("IV", v / 100) for v in (10, 12, 14, 16, 18, 20, 22, 25, 30)] + [("IVpct", v / 100) for v in (20, 40, 50, 60, 80)] + [("VRP", v) for v in (1.0, 1.1, 1.2, 1.3)]
rng = np.random.default_rng(79)
def fname(f): return "none" if f[0] == "none" else f"{f[0]}≥{f[1]:.0%}" if f[0] != "VRP" else f"VRP≥{f[1]:.1f}"
def lbl(st, sl): return m.sname(st) + (" SL2x" if sl else " 無SL")


def msharpe(x): x = np.asarray(x, float); return float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else 0.0


def main():
    T = [t for t in pickle.load(open(CACHE, "rb")) if t["s"] >= IS0]
    iv = load_atm_iv().atm_iv_30d.sort_index(); IVEND = iv.index.max()
    T = [t for t in T if t["s"] <= IVEND]
    ivp = iv.rolling(252, min_periods=60).apply(lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
    F = load_fut("2016-01-01"); rv = (fut_ret(F).rolling(20).std() * np.sqrt(252))
    F2 = load_fut("2016-06-01"); days = F2.index[(F2.index <= m.END) & (F2.index >= IS0)]
    m.ensure_minutes(T, days)
    G = {}
    for t in T: G.setdefault(t["st"], []).append(t)
    OOSEND = max(t["hold"][-1] for t in T); print("IV 末日", IVEND.date(), "OOS 截至", OOSEND.date())
    months = pd.period_range("2017-01", OOSEND.to_period("M"), freq="M")
    MIS = months[months < pd.Period("2022-01")]; MOO = months[months >= pd.Period("2022-01")]
    rows, trows, rrows, wrows, brows, ser = [], [], [], [], [], {}
    for st, sl in STRS:
        TT = G[st]; L = lbl(st, sl)
        tr = []
        for t in TT:
            r = m.sim(t, None, sl, {}); d = pd.Series(r["pnl"] * MULT, t["hold"][:r["L"]])
            tr.append(dict(s=t["s"], X=t["X"], iv=iv.asof(t["s"]), ivp=ivp.asof(t["s"]), rv=rv.asof(t["s"]), net=r["net"] * MULT, why=r["why"], C0=t["C0"],
                           mu=m.margin_unit(t), mon=d.groupby(d.index.to_period("M")).sum()))
        D = pd.DataFrame(tr); D["vrp"] = D.iv / D.rv
        MON = np.array([D.mon[i].reindex(months, fill_value=0).values for i in range(len(D))])  # trade × month
        isIS = (D.s < IS_END).values
        def passmask(f):
            if f[0] == "none": return np.ones(len(D), bool)
            if f[0] == "IV": return (D.iv >= f[1]).values
            if f[0] == "IVpct": return (D.ivp.isna() | (D.ivp >= f[1])).values
            return (D.vrp >= f[1]).values
        for f in FILTERS:
            pm = passmask(f); ms = MON[pm].sum(0); ser[(L, fname(f))] = pd.Series(ms, months)
            r = dict(struct=L, filt=fname(f))
            for nm, sel, mm in (("IS", isIS, months < pd.Period("2022-01")), ("OOS", ~isIS, months >= pd.Period("2022-01"))):
                g = D[pm & sel]; r.update({f"{nm}_n": len(g), f"{nm}_pct_months": len(g) / max((sel).sum(), 1), f"{nm}_win": (g.net > 0).mean() if len(g) else np.nan,
                                           f"{nm}_avg": g.net.mean() if len(g) else np.nan, f"{nm}_net": g.net.sum(), f"{nm}_Sharpe": msharpe(ms[mm]), f"{nm}_worst": g.net.min() if len(g) else np.nan,
                                           f"{nm}_stop_rate": (g.why == "stop").mean() if len(g) else np.nan})
            keep = [t for t, p in zip(TT, pm) if p]
            pc, wc, rec = m.run_variant(keep, None, sl, days, {})
            r["OOS_MDDw"] = m.stats(pc, wc, rec, IS_END, OOSEND + pd.Timedelta(days=1))["MDDw"] if len(keep) else 0
            r["IS_MDDw"] = m.stats(pc, wc, rec, IS0, IS_END)["MDDw"] if len(keep) else 0
            for u in (.10, .25):
                pe, we, re = m.run_variant(keep, None, sl, days, {}, u=u)
                for nm, a, b in (("IS", IS0, IS_END), ("OOS", IS_END, OOSEND + pd.Timedelta(days=1))):
                    s_ = m.stats(pe, we, re, a, b, eq0=m.CAP0); r[f"{nm}_CAGR_u{int(u*100)}"] = s_["CAGR"]; r[f"{nm}_MDD_u{int(u*100)}"] = s_["MDDw_pct"]
            rows.append(r)
        R = pd.DataFrame([x for x in rows if x["struct"] == L])
        Q = R[R.IS_n >= 20]; best = Q.loc[Q.IS_Sharpe.idxmax()]
        # 隨機略過：保留與 best 同數量的 IS / OOS 交易
        for fb in dict.fromkeys([best.filt, "IV≥16%", "IV≥20%", "IVpct≥50%"]):
            b = R[R.filt == fb].iloc[0]; pm = passmask([f for f in FILTERS if fname(f) == fb][0])
            sh = {"IS": [], "OOS": []}
            for _ in range(500):
                msk = np.zeros(len(D), bool)
                for sel in (isIS, ~isIS):
                    idx = np.flatnonzero(sel); msk[rng.choice(idx, (pm & sel).sum(), replace=False)] = True
                s_ = MON[msk].sum(0); sh["IS"].append(msharpe(s_[months < pd.Period("2022-01")])); sh["OOS"].append(msharpe(s_[months >= pd.Period("2022-01")]))
            rrows.append(dict(struct=L, filt=fb, is_IS_choice=fb == best.filt, IS_Sharpe=b.IS_Sharpe, IS_pct=(np.array(sh["IS"]) < b.IS_Sharpe).mean(),
                              OOS_Sharpe=b.OOS_Sharpe, OOS_pct=(np.array(sh["OOS"]) < b.OOS_Sharpe).mean(), OOS_rand_med=np.median(sh["OOS"])))
        # WF：逐年擴張窗重選
        wf = []
        for Y in range(2019, OOSEND.year + 1):
            fit = months < pd.Period(f"{Y}-01"); test = (months >= pd.Period(f"{Y}-01")) & (months < pd.Period(f"{Y+1}-01"))
            sc = {}
            for f in FILTERS:
                pm = passmask(f)
                if (pm & (D.s < pd.Timestamp(f"{Y}-01-01")).values).sum() >= 12: sc[fname(f)] = msharpe(ser[(L, fname(f))].values[fit])
            kb = max(sc, key=sc.get); v = ser[(L, kb)].values[test]; wf.append(pd.Series(v, months[test]))
            wrows.append(dict(struct=L, year=Y, pick=kb, fit_Sharpe=sc[kb], year_net=v.sum(), none_year_net=ser[(L, "none")].values[test].sum()))
        wf = pd.concat(wf); wo = wf[wf.index >= pd.Period("2022-01")]
        wrows.append(dict(struct=L, year="WF 2022+", pick="", fit_Sharpe=msharpe(wo.values), year_net=wo.sum(), none_year_net=ser[(L, "none")][MOO].sum()))
        # IV 分組
        D["bucket"] = pd.cut(D.iv * 100, [0, 12, 16, 20, 25, 200], right=False, labels=["<12", "12–16", "16–20", "20–25", "≥25"])
        for bk, g in D.groupby("bucket", observed=False):
            brows.append(dict(struct=L, bucket=bk, n=len(g), n_IS=(g.s < IS_END).sum(), n_OOS=(g.s >= IS_END).sum(), avg_credit_pts=g.C0.mean(), avg_net=g.net.mean(),
                              loss_freq=(g.net <= 0).mean(), avg_net_per_margin=(g.net / g.mu).mean(), worst=g.net.min(), stop_rate=(g.why == "stop").mean()))
        trows += [dict(struct=L, **{k: v for k, v in x.items() if k != "mon"}) for x in tr]
        print(L, "IS 選擇", best.filt, flush=True)
    R = pd.DataFrame(rows); R.to_csv(RES / "dd_put_iv_grid.csv", index=False)
    RR = pd.DataFrame(rrows); RR.to_csv(RES / "dd_put_iv_random.csv", index=False)
    W = pd.DataFrame(wrows); W.to_csv(RES / "dd_put_iv_wf.csv", index=False)
    B = pd.DataFrame(brows); B.to_csv(RES / "dd_put_iv_bucket.csv", index=False)
    TR = pd.DataFrame(trows); TR.to_csv(RES / "dd_put_iv_trades.csv", index=False)
    pd.set_option("display.width", 400); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 300)
    for L, g in R.groupby("struct", sort=False):
        print(f"{L}: Spearman IS→OOS（門檻間）{g.IS_Sharpe.rank().corr(g.OOS_Sharpe.rank()):.2f}")
    print(RR.round(2).to_string(index=False)); print(W.round(2).to_string(index=False)); print(B.round(3).to_string(index=False))
    chart(R, B)


def chart(R, B):
    plt = plt_style(); structs = list(dict.fromkeys(R.struct)); cols = [GOLD, BLUE, "#e07b39", GR, WH, "#9ad36a"]
    fig, axs = plt.subplots(2, 1, figsize=(16, 11), facecolor=BG)
    ax = axs[0]; sty(ax); fl = list(dict.fromkeys(R.filt)); xx = np.arange(len(fl))
    for c, L in zip(cols, structs):
        g = R[R.struct == L].set_index("filt").reindex(fl)
        ax.plot(xx, g.IS_Sharpe, "x--", color=c, alpha=.6); ax.plot(xx, g.OOS_Sharpe, "o-", color=c, label=L)
    ax.set_xticks(xx); ax.set_xticklabels(fl, rotation=40, color=WH, fontsize=8); ax.axhline(0, color=GR)
    ax.set_title("月 Sharpe：IS 2017–21（虛線 x）vs OOS 2022+（實線 o），依 IV 門檻"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=8)
    ax = axs[1]; sty(ax); bk = ["<12", "12–16", "16–20", "20–25", "≥25"]; xx = np.arange(len(bk))
    for i, (c, L) in enumerate(zip(cols, structs)):
        g = B[B.struct == L].set_index("bucket").reindex(bk)
        ax.bar(xx + (i - 2.5) * .13, g.avg_net_per_margin * 100, .13, color=c, label=L)
        for x, n in zip(xx, g.n): ax.text(x + (i - 2.5) * .13, 0, f"{int(n) if n == n else 0}", color=WH, fontsize=6, ha="center", va="bottom")
    ax.set_xticks(xx); ax.set_xticklabels([f"IV {b}%" for b in bk], color=WH); ax.axhline(0, color=GR)
    ax.set_title("每筆平均淨損益 / 單位保證金（%），依訊號日 ATM IV 分組（數字 = 筆數）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=8)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "dd_put_iv.png", dpi=110, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
