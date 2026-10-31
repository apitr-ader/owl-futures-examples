"""80 月選 put 賣方：只在 IV 下降或偏低時賣出（承 76/77/79 引擎）。

IV：_rqo.load_atm_iv() atm_iv_30d，訊號日 s' 收盤值（入場日 s'+1 開盤成交），無前視；IV 止於 2026-08-17，之後的循環不納入。
規則（s' 收盤判定）：d1: IV<前一日；maN: IV<過去 N 日均值（含當日，N=5,10,20,60,120）；d1&maN；dnk: IV<k 日前（k=3,5）；none。
模式：
  M1「只看正常入場日」：s' = 前一月選結算日 s；條件不成立 → 當月空手。
  M2「等待」：s' = s 起第 0..10 個交易日中第一個條件成立日（且入場日距結算 ≥7 曆日），以該日收盤重選履約價、次日開盤成交；皆不成立 → 空手。
結構：價差 x2% w4% SL2x、x3% w4% SL2x；裸賣 x2%、ATM 無停損。成本/1 分鐘停損判定同 76。
Sharpe：月損益（含空手月 0）×√12。IS 2017–2021 選規則（IS 交易 ≥20），OOS 2022+。
驗證：Spearman IS→OOS（規則間，各模式）；隨機基準 500 次：M1 隨機略過同數月；M2 同數月 + 隨機延遲（同可用天數範圍）；逐年 WF；與 none 比較。
條件統計：正常入場日 IV 下降 vs 上升的每筆損益。
輸出：results/dd_put_ivfall_{grid,random,wf,cond,trades}.csv、dd_put_ivfall.png；快取 data/cache/（不入 repo）。
"""
import importlib.util, pickle, sys
import numpy as np, pandas as pd
from _rqo import *
spec = importlib.util.spec_from_file_location("m76", Path(__file__).parent / "76_put_tp_sl.py"); m = importlib.util.module_from_spec(spec); sys.modules["m76"] = m; spec.loader.exec_module(m)
m74 = m.m74; IS0 = m.IS0; END = m.END
CACHE = ROOT / "data/cache/dd_put_ivfall_trades.pkl"
STRS = [(("x", .02, .04), 2.0), (("x", .03, .04), 2.0), (("x", .02, 0), None), (("x", 0.0, 0), None)]
MAXOFF, MINCAL = 10, 7
RULES = ["none", "d1"] + [f"ma{n}" for n in (5, 10, 20, 60, 120)] + [f"d1&ma{n}" for n in (5, 10, 20, 60, 120)] + ["dn3", "dn5"]
rng = np.random.default_rng(80)
def lbl(st, sl): return m.sname(st) + (" SL2x" if sl else " 無SL")
def msharpe(x): x = np.asarray(x, float); return float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else 0.0


def build():
    F = load_fut("2016-06-01"); days = F.index[F.index <= END]
    E = expiries(); allm = E[E.kind == "monthly"].sort_values("expiry_date").reset_index(drop=True)
    M = allm[(allm.expiry_date <= END) & (allm.expiry_date >= IS0)]
    O0 = load_opt("2016-11-01", END, cp="P"); O0 = O0[O0.expiry.isin(M.expiry)]
    T = []
    for X, xd in zip(M.expiry, M.expiry_date):
        if xd not in days: continue
        g = O0[O0.expiry == X]
        pv = {f: g.pivot_table(index="date", columns="strike", values=f) for f in ("day_open", "day_high", "day_low", "day_close")}
        s0 = allm[allm.expiry_date < xd].expiry_date.iloc[-1]
        if s0 not in days: continue
        i0 = days.get_loc(s0); ix = days.get_loc(xd)
        for j in range(MAXOFF + 1):
            si = i0 + j; s = days[si]; d1 = days[si + 1]
            if (xd - d1).days < MINCAL: break
            if s not in pv["day_close"].index or d1 not in pv["day_close"].index: continue
            hold = days[si + 1: ix + 1]
            Fs = F.close[s]; cl = pv["day_close"].loc[s].dropna(); cl = cl[(cl.index < Fs) & (cl > 0.5)]
            if len(cl) < 3: continue
            Fc, Fl, Fh = F.close[hold].values, F.low[hold].values, F.high[hold].values
            px = lambda K: (pv["day_open"].at[d1, K] if not np.isnan(pv["day_open"].at[d1, K]) else pv["day_close"].at[d1, K]) if K in pv["day_open"].columns else np.nan
            for st, _ in STRS:
                K = cl.index[np.abs(cl.index - Fs * (1 - st[1])).argmin()]; ps = px(K)
                if np.isnan(ps) or ps < 1: continue
                S = m74.leg_path(pv, K, hold, Fc, Fl, Fh, True); Kl, pl, L = None, 0.0, [np.zeros(len(hold))] * 4
                if st[2]:
                    cand = pv["day_close"].columns[pv["day_close"].columns < K]
                    if len(cand) == 0: continue
                    Kl = cand[np.abs(cand - (K - st[2] * Fs)).argmin()]; pl = px(Kl)
                    if np.isnan(pl): continue
                    L = m74.leg_path(pv, Kl, hold, Fc, Fl, Fh, False)
                T.append(dict(st=st, X=X, s=s, s0=s0, j=j, d1=d1, hold=hold, Fs=Fs, K=K, Kl=Kl, ps=ps, pl=pl, S=S, L=L, Fc=Fc, Fl=Fl, w=st[2], C0=ps - pl))
    pickle.dump(T, open(CACHE, "wb")); return T


def main():
    T = pickle.load(open(CACHE, "rb")) if CACHE.exists() else build()
    iv = load_atm_iv().atm_iv_30d.sort_index(); IVEND = iv.index.max()
    T = [t for t in T if t["s0"] >= IS0 and t["s"] <= IVEND]
    sig = pd.DataFrame({"iv": iv, "d1": iv < iv.shift(1), "dn3": iv < iv.shift(3), "dn5": iv < iv.shift(5)})
    for n in (5, 10, 20, 60, 120):
        sig[f"ma{n}"] = iv < iv.rolling(n, min_periods=n).mean(); sig[f"d1&ma{n}"] = sig.d1 & sig[f"ma{n}"]
    sig["none"] = True
    F2 = load_fut("2016-06-01"); days = F2.index[(F2.index <= END) & (F2.index >= IS0)]
    m.TPS = [None]; m.SLS = [2.0]; m.ensure_minutes([t for t in T if t["w"] > 0], days)
    OOSEND = max(t["hold"][-1] for t in T if t["j"] == 0); print("IV 末日", IVEND.date(), "OOS 截至", OOSEND.date(), flush=True)
    months = pd.period_range("2017-01", OOSEND.to_period("M"), freq="M"); mIS = months < pd.Period("2022-01")
    rows, rrows, wrows, crows, trows, ser = [], [], [], [], [], {}
    for st, sl in STRS:
        Lb = lbl(st, sl); TT = sorted([t for t in T if t["st"] == st], key=lambda t: (t["s0"], t["j"]))
        recs = []
        for t in TT:
            r = m.sim(t, None, sl, {}); d = pd.Series(r["pnl"] * MULT, t["hold"][:r["L"]])
            rw = dict(X=t["X"], s0=t["s0"], j=t["j"], s=t["s"], iv=iv.get(t["s"], np.nan), net=r["net"] * MULT, why=r["why"], C0=t["C0"], mu=m.margin_unit(t),
                      mon=d.groupby(d.index.to_period("M")).sum().reindex(months, fill_value=0).values)
            for k in RULES: rw[k] = bool(sig[k].get(t["s"], False)) if t["s"] in sig.index else False
            recs.append(rw)
        D = pd.DataFrame(recs); D["ti"] = range(len(D)); D["isIS"] = D.s0 < IS_END; MON = np.vstack(D.mon.values)
        D0 = D[D.j == 0]; cyc = D0.X.values  # 正常入場可交易的循環（基準）
        trows += [dict(struct=Lb, **{k: v for k, v in x.items() if k != "mon"}) for x in recs]
        def pick(rule, mode):
            if mode == "M1": g = D0[D0[rule]]
            else: g = D[D.X.isin(cyc) & D[rule]].groupby("X").head(1)
            return g
        for mode in ("M1", "M2"):
            for rule in RULES:
                if mode == "M2" and rule == "none": continue
                g = pick(rule, mode); ms = MON[g.ti.values].sum(0); ser[(Lb, mode, rule)] = ms
                r = dict(struct=Lb, mode=mode, rule=rule)
                for nm, sel, mm in (("IS", g.isIS, mIS), ("OOS", ~g.isIS, ~mIS)):
                    h = g[sel]; base = (D0.s0 < IS_END).sum() if nm == "IS" else (D0.s0 >= IS_END).sum()
                    r.update({f"{nm}_n": len(h), f"{nm}_pct_months": len(h) / base, f"{nm}_win": (h.net > 0).mean() if len(h) else np.nan, f"{nm}_avg": h.net.mean() if len(h) else np.nan,
                              f"{nm}_net": h.net.sum(), f"{nm}_Sharpe": msharpe(ms[mm]), f"{nm}_worst": h.net.min() if len(h) else np.nan, f"{nm}_avg_delay": h.j.mean() if len(h) else np.nan})
                keep = [TT[i] for i in g.ti.values]
                if keep:
                    pc, wc, rec = m.run_variant(keep, None, sl, days, {})
                    r["IS_MDDw"] = m.stats(pc, wc, rec, IS0, IS_END)["MDDw"]; r["OOS_MDDw"] = m.stats(pc, wc, rec, IS_END, OOSEND + pd.Timedelta(days=1))["MDDw"]
                    for u in (.10, .25):
                        pe, we, re_ = m.run_variant(keep, None, sl, days, {}, u=u)
                        for nm, a, b in (("IS", IS0, IS_END), ("OOS", IS_END, OOSEND + pd.Timedelta(days=1))):
                            s_ = m.stats(pe, we, re_, a, b, eq0=m.CAP0); r[f"{nm}_CAGR_u{int(u*100)}"] = s_["CAGR"]; r[f"{nm}_MDD_u{int(u*100)}"] = s_["MDDw_pct"]
                rows.append(r)
        R = pd.DataFrame([x for x in rows if x["struct"] == Lb])
        # 隨機基準
        opts = {X: g.ti.values for X, g in D[D.X.isin(cyc)].groupby("X")}
        isc = D0.isIS.values
        for mode in ("M1", "M2"):
            Q = R[(R["mode"] == mode) & (R.IS_n >= 20)]; best = Q.loc[Q.IS_Sharpe.idxmax()].rule
            for rule in dict.fromkeys([best, "d1", "ma20", "d1&ma20"]):
                g = pick(rule, mode); b = R[(R["mode"] == mode) & (R.rule == rule)].iloc[0]
                nIS, nO = g.isIS.sum(), (~g.isIS).sum(); sh = {"IS": [], "OOS": []}
                for _ in range(500):
                    ch = list(rng.choice(np.flatnonzero(isc), nIS, replace=False)) + list(rng.choice(np.flatnonzero(~isc), nO, replace=False))
                    if mode == "M1": ti = D0.ti.values[ch]
                    else: ti = [rng.choice(opts[cyc[c]]) for c in ch]
                    s_ = MON[ti].sum(0); sh["IS"].append(msharpe(s_[mIS])); sh["OOS"].append(msharpe(s_[~mIS]))
                rrows.append(dict(struct=Lb, mode=mode, rule=rule, is_IS_choice=rule == best, IS_Sharpe=b.IS_Sharpe, IS_pct=(np.array(sh["IS"]) < b.IS_Sharpe).mean(),
                                  OOS_Sharpe=b.OOS_Sharpe, OOS_pct=(np.array(sh["OOS"]) < b.OOS_Sharpe).mean(), OOS_rand_med=np.median(sh["OOS"]), none_OOS=R[(R["mode"] == "M1") & (R.rule == "none")].OOS_Sharpe.iloc[0]))
            # WF
            wf = []
            for Y in range(2019, OOSEND.year + 1):
                fit = months < pd.Period(f"{Y}-01"); test = (months >= pd.Period(f"{Y}-01")) & (months < pd.Period(f"{Y+1}-01")); sc = {}
                for rule in RULES:
                    key = (Lb, mode, rule) if (mode == "M1" or rule != "none") else (Lb, "M1", "none")
                    if (pick(rule, mode).s0 < pd.Timestamp(f"{Y}-01-01")).sum() >= 12: sc[rule] = msharpe(ser[key][fit])
                kb = max(sc, key=sc.get); key = (Lb, mode, kb) if (mode == "M1" or kb != "none") else (Lb, "M1", "none")
                v = ser[key][test]; wf.append(v); wrows.append(dict(struct=Lb, mode=mode, year=Y, pick=kb, fit_Sharpe=sc[kb], year_net=v.sum(), none_year_net=ser[(Lb, "M1", "none")][test].sum()))
            wf = np.concatenate(wf); wo = wf[-(~mIS).sum():]
            wrows.append(dict(struct=Lb, mode=mode, year="WF 2022+", pick="", fit_Sharpe=msharpe(wo), year_net=wo.sum(), none_year_net=ser[(Lb, "M1", "none")][~mIS].sum()))
        # 條件統計（正常入場日）
        for nm, sel in (("ALL", D0.s0 > "1900"), ("IS", D0.isIS), ("OOS", ~D0.isIS)):
            for lab, c in (("IV falling (d1)", D0.d1), ("IV rising/flat", ~D0.d1), ("IV<MA20", D0.ma20), ("IV≥MA20", ~D0.ma20)):
                h = D0[sel & c]
                crows.append(dict(struct=Lb, period=nm, cond=lab, n=len(h), win=(h.net > 0).mean(), avg_net=h.net.mean(), med_net=h.net.median(), avg_credit_pts=h.C0.mean(),
                                  avg_net_per_margin=(h.net / h.mu).mean(), worst=h.net.min(), stop_rate=(h.why == "stop").mean()))
        for mode in ("M1", "M2"):
            g = R[R["mode"] == mode]; print(f"{Lb} {mode}: IS 選擇 {R.loc[g[g.IS_n >= 20].IS_Sharpe.idxmax()].rule}  Spearman {g.IS_Sharpe.rank().corr(g.OOS_Sharpe.rank()):.2f}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(RES / "dd_put_ivfall_grid.csv", index=False)
    RR = pd.DataFrame(rrows); RR.to_csv(RES / "dd_put_ivfall_random.csv", index=False)
    W = pd.DataFrame(wrows); W.to_csv(RES / "dd_put_ivfall_wf.csv", index=False)
    C = pd.DataFrame(crows); C.to_csv(RES / "dd_put_ivfall_cond.csv", index=False)
    pd.DataFrame(trows).to_csv(RES / "dd_put_ivfall_trades.csv", index=False)
    pd.set_option("display.width", 400); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
    print(R[["struct", "mode", "rule", "IS_n", "OOS_n", "IS_Sharpe", "OOS_Sharpe", "OOS_avg", "OOS_net", "OOS_worst", "OOS_MDDw", "OOS_CAGR_u25", "OOS_MDD_u25"]].round(2).to_string(index=False))
    print(RR.round(2).to_string(index=False)); print(W.round(2).to_string(index=False)); print(C.round(3).to_string(index=False))
    chart(R, C)


def chart(R, C):
    plt = plt_style(); structs = list(dict.fromkeys(R.struct)); cols = [GOLD, BLUE, "#e07b39", "#9ad36a"]
    fig, axs = plt.subplots(3, 1, figsize=(16, 15), facecolor=BG)
    for ax, mode in zip(axs[:2], ("M1", "M2")):
        sty(ax); fl = [r for r in RULES if not (mode == "M2" and r == "none")]; xx = np.arange(len(fl))
        for c, L in zip(cols, structs):
            g = R[(R.struct == L) & (R["mode"] == mode)].set_index("rule").reindex(fl)
            ax.plot(xx, g.IS_Sharpe, "x--", color=c, alpha=.6); ax.plot(xx, g.OOS_Sharpe, "o-", color=c, label=L)
            ax.axhline(R[(R.struct == L) & (R["mode"] == "M1") & (R.rule == "none")].OOS_Sharpe.iloc[0], color=c, lw=.7, ls=":")
        ax.set_xticks(xx); ax.set_xticklabels(fl, rotation=30, color=WH, fontsize=8); ax.axhline(0, color=GR)
        ax.set_title(f"{mode}{'（只看正常入場日）' if mode == 'M1' else '（等待至多 10 日）'} 月 Sharpe：IS 虛線 x / OOS 實線 o；點線 = 無濾網 OOS")
        ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=8)
    ax = axs[2]; sty(ax); cd = ["IV falling (d1)", "IV rising/flat", "IV<MA20", "IV≥MA20"]; xx = np.arange(len(cd))
    for i, (c, L) in enumerate(zip(cols, structs)):
        for k, (per, a) in enumerate((("IS", .5), ("OOS", 1.0))):
            g = C[(C.struct == L) & (C.period == per)].set_index("cond").reindex(cd)
            ax.bar(xx + (i * 2 + k - 3.5) * .1, g.avg_net, .1, color=c, alpha=a, label=f"{L} {per}")
    ax.set_xticks(xx); ax.set_xticklabels(cd, color=WH); ax.axhline(0, color=GR)
    ax.set_title("正常入場日條件：每筆平均淨損益（元，1 口）— 淡 = IS，深 = OOS"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=7, ncol=4)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "dd_put_ivfall.png", dpi=110, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
