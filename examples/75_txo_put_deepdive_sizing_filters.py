"""75 月選 OTM put 賣方深度回測（二）：GEX-A / IV 濾網、保證金與資金部位、尾部與追繳壓力、對照 covered call。

讀 74 的交易快取（data/cache/dd_put_trades.pkl）。同樣時序/成本（收盤訊號、次日開盤成交）。
濾網（訊號日 s 已知）：
  GEX-A（call+/put−，OI×Γ×F²×1%×50，承 71）近 252 日百分位 < q → 跳過或半倉；持有中 GEX 百分位跌破 q → 次日開盤出場。
  ATM IV（前月月選近價平 put 收盤、Black-76）近 252 日百分位 ≥ q；VRP = ATM IV − 20 日已實現波動 > 0。
保證金（SPAN 近似、TAIFEX 賣方公式）：裸賣 = 權利金市值 + max(A − 價外值, B)，A = 7.5%×F×50、B = A/2；
  put 價差 = 履約價差×50。維持 = 75%×原始；盤中權益（short 高、long 低、F 低）< 維持 → 記一次追繳。
資金：100 萬起、複利，每期口數 = floor(u×權益/單位保證金)，u∈{.05,.1,.25,.5,.75,1}；另「現金擔保」口數 = floor(權益/(K×50))（對照 covered call）。
驗證：IS 2017–2021 選參、OOS 2022+；隨機 gate（同通過率 1000 次）；逐年 WF；尾部視窗與最差月份。
"""
import importlib.util, pickle
import numpy as np, pandas as pd
from _rqo import *

spec = importlib.util.spec_from_file_location("m74", Path(__file__).parent / "74_txo_put_deepdive_grid.py"); m74 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m74)
spec = importlib.util.spec_from_file_location("m71", Path(__file__).parent / "71_txo_oi_dealer_gamma.py"); m71 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m71)
END, PRE, IS0 = m74.END, m74.PRE, m74.IS0
CAP0 = 1_000_000.0
rng = np.random.default_rng(21)
TAILS = [("2015-08 股災", "2015-08-01", "2015-09-30"), ("2018-10", "2018-10-01", "2018-11-15"), ("2020-03 疫情", "2020-02-15", "2020-04-15"),
         ("2022 空頭", "2022-01-01", "2022-12-31"), ("2024-08 日圓套利", "2024-07-10", "2024-08-31"), ("2025-04 關稅", "2025-03-25", "2025-05-10")]


def atm_iv_daily(F):
    from scipy.stats import norm
    E = expiries(); M = E[E.kind == "monthly"].set_index("expiry").expiry_date
    o = load_opt("2009-10-01", END, cp="P"); o = o[o.expiry.isin(M.index) & o.day_close.notna() & (o.day_close > 0)]
    o["xd"] = o.expiry.map(M); o["F"] = o.date.map(F.close); o = o.dropna(subset=["F"])
    o["dte"] = (o.xd - o.date).dt.days; o = o[o.dte >= 7]
    o = o[o.xd == o.groupby("date").xd.transform("min")]
    o["dist"] = (o.strike - o.F).abs(); o = o[o.dist == o.groupby("date").dist.transform("min")].drop_duplicates("date")
    iv, _ = m74.b76_put(o.day_close.values, o.F.values, o.strike.values.astype(float), o.dte.values / 365)
    return pd.Series(np.where(iv < 2.9, iv, np.nan), o.date.values).sort_index()


def margin_unit(t, L, worst=False):
    """每組保證金需求（元）陣列，長度 L。"""
    if t["w"] > 0: return np.full(L, (t["K"] - t["Kl"]) * MULT)
    Fv = (t["Fl"] if worst else t["Fc"])[:L]; pv = (t["sh"] if worst else t["sc"])[:L]
    A = .075 * Fv * MULT; otm = np.maximum(Fv - t["K"], 0) * MULT
    return pv * MULT + np.maximum(A - otm, A / 2)


def sized(TT, stop, days, u=None, cash=False, mask=None, half=None, gexexit=None):
    """複利資金模擬。回傳 日損益(元) Series、盤中最差 Series、交易紀錄、追繳次數。"""
    eq = CAP0; pc = pd.Series(0.0, days); wc = pd.Series(0.0, days); rec = []; calls = 0; calls_close = 0
    for i, t in enumerate(TT):
        if mask is not None and not mask[i]: continue
        if gexexit is not None:
            g = gexexit.reindex(t["hold"]).values
            tt = dict(t); hit = np.flatnonzero(np.nan_to_num(g, nan=1)[:-1] < .3)
        net, p, w, L, why = m74.sim(t, stop)
        if gexexit is not None and len(hit) and hit[0] + 1 < L:
            # 以 GEX 出場：截斷於 hit+1 日開盤
            e = hit[0] + 1; vo = t["so"][e] - t["lo"][e]; vc = t["sc"] - t["lc"]
            p = p[:e + 1].copy(); w = w[:e + 1].copy(); p[e] = vc[e - 1] - vo - opt_cost_pts(t["so"][e]) - (opt_cost_pts(max(t["lo"][e], .1)) if t["w"] else 0); w[e] = 0; L = e + 1; why = "gex"
        m0 = margin_unit(t, 1)[0]
        n = int(eq / (t["K"] * MULT)) if cash else int(u * eq / m0)
        if half is not None and half[i]: n //= 2
        if n <= 0: continue
        idx = t["hold"][:L]; d = p * MULT * n
        eqc = eq + np.cumsum(d); eqw = eqc + w * MULT * n
        req = margin_unit(t, L, True) * n
        calls += int((eqw < .75 * req).any()); calls_close += int((eqc < .75 * margin_unit(t, L) * n).any())
        pc[idx] += d; wc[idx] = np.minimum(wc[idx].values, w * MULT * n)
        rec.append(dict(s=t["s"], X=t["X"], n=n, net=d.sum(), ret=d.sum() / eq, why=why, eq0=eq, minratio=float((eqw / req).min())))
        eq = eqc[-1]
        if eq <= 0: break
    return pc, wc, pd.DataFrame(rec), calls, calls_close


def perf(pc, wc, rec, a, b, cap_start=None):
    m = (pc.index >= a) & (pc.index < b); p = pc[m]; w = wc[m]
    e0 = CAP0 + pc[pc.index < a].sum(); eq = e0 + p.cumsum(); prev = eq.shift(fill_value=e0)
    yrs = len(p) / 252; cagr = (eq.iloc[-1] / e0) ** (1 / yrs) - 1 if len(p) and eq.iloc[-1] > 0 else -1
    r = p / prev; dd = ((prev + w) / np.maximum.accumulate(np.r_[e0, eq.values[:-1]]) - 1).min()
    mo = (p.groupby(p.index.to_period("M")).sum() / eq.shift(fill_value=e0).groupby(p.index.to_period("M")).first())
    rr = rec[(rec.s >= a) & (rec.s < b)] if len(rec) else rec
    return dict(CAGR=cagr, Sharpe=sharpe(r), MDDw=dd, worst_mo=mo.min() if len(mo) else np.nan, n=len(rr))


def main():
    T = pickle.load(open(m74.CACHE, "rb")); S = pickle.load(open(ROOT / "data/cache/dd_put_74_summary.pkl", "rb")); R = S["R"]
    F = load_fut("2009-01-01"); days = F.index[(F.index <= END) & (F.index >= PRE)]
    groups = {}
    for t in T: groups.setdefault(m74.key_of(t), []).append(t)
    # ---- 濾網資料 ----
    o = m71.load_oi(); G, _ = m71.gex(o, F[F.index >= "2016-06-01"]); gp = m71.pct(G.A)
    iv = atm_iv_daily(F); ivp = m71.pct(iv)
    r = fut_ret(F); rv = r.rolling(20).std() * np.sqrt(252); vrp = (iv - rv.reindex(iv.index))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    best = S["best"]; print("74 IS 最佳:", best)
    CANDS = [best, ("x0.03", "E+1", 0, "none"), ("x0.05", "E+1", 0, "none"), ("x0.03", "E+1", .04, "none"), ("x0.05", "E+1", .04, "none"),
             ("d0.20", "E+1", 0, "none"), ("d0.20", "E+1", .04, "none"), ("x0.05", "E+1", .06, "none"), ("x0.03", "E+1", 0, "v3"), ("x0.04", "E+1", .02, "v4"), ("x0.04", "10", .02, "v4")]
    CANDS = list(dict.fromkeys(c for c in CANDS if c[:3] in groups))
    # ---- 1. 濾網：GEX-A / IV / VRP，以 1 組計（點→元），IS 2017–21 / OOS ----
    FILT = {"無": lambda s: True, "GEX≥.3": lambda s: gp.asof(s) >= .3 if pd.notna(gp.asof(s)) else True,
            "GEX≥.5": lambda s: gp.asof(s) >= .5 if pd.notna(gp.asof(s)) else True,
            "IVp≥.3": lambda s: ivp.asof(s) >= .3, "IVp≥.5": lambda s: ivp.asof(s) >= .5, "IVp≤.8": lambda s: ivp.asof(s) <= .8,
            "VRP>0": lambda s: vrp.asof(s) > 0, "GEX≥.3&VRP>0": lambda s: (gp.asof(s) >= .3 if pd.notna(gp.asof(s)) else True) and vrp.asof(s) > 0,
            "GEX≥.3&IVp≥.3": lambda s: (gp.asof(s) >= .3 if pd.notna(gp.asof(s)) else True) and ivp.asof(s) >= .3}
    rows = []; fser = {}
    for c in CANDS:
        TT = groups[c[:3]]; sig = [t["s"] for t in TT]
        for fn, f in FILT.items():
            m = np.array([bool(f(s)) for s in sig])
            pc, wc, rec = m74.daily(TT, c[3], days, m); fser[(c, fn)] = pc
            row = dict(cfg=str(c), filt=fn, pass_=m[np.array(sig) >= np.datetime64(IS0)].mean())
            for nm, a, b in m74.PER[1:]:
                for kk, v in m74.pstats(pc, wc, rec, a, b).items():
                    if kk in ("n", "net", "Sharpe", "MDDw", "worst"): row[f"{nm}_{kk}"] = v
            # 隨機 gate 基準（OOS）
            per = np.array([m74.sim(t, c[3])[0] for t in TT]) * MULT; oos = np.array(sig) >= np.datetime64(IS_END)
            rnd = [per[oos & (rng.random(len(TT)) < m.mean())].sum() for _ in range(1000)]
            row["OOS_rnd_pct"] = (np.array(rnd) < per[oos & m].sum()).mean(); rows.append(row)
        # 持有中 GEX 出場 + 半倉版本（用資金模擬以 u=.5 比較）
    FR = pd.DataFrame(rows); FR.to_csv(RES / "dd_put_75_filters.csv", index=False)
    print("\n== 1. 濾網（1 組，元）=="); print(FR.round(2).to_string(index=False))
    fsp = FR[FR.filt != "無"]; print(f"濾網 IS vs OOS Sharpe Spearman {fsp.IS_Sharpe.rank().corr(fsp.OOS_Sharpe.rank()):.2f}")
    dl = FR.merge(FR[FR.filt == "無"][["cfg", "IS_Sharpe", "OOS_Sharpe", "OOS_MDDw"]], on="cfg", suffixes=("", "_base"))
    dl["dIS"] = dl.IS_Sharpe - dl.IS_Sharpe_base; dl["dOOS"] = dl.OOS_Sharpe - dl.OOS_Sharpe_base; dl["dMDD_OOS"] = dl.OOS_MDDw - dl.OOS_MDDw_base
    print("\n濾網平均增量（vs 無濾網）："); print(dl.groupby("filt")[["dIS", "dOOS", "dMDD_OOS", "pass_", "OOS_rnd_pct"]].mean().round(3).to_string())
    # ---- 2. 資金/保證金 ----
    print("\n== 2. 資金部位（100 萬、複利；盤中 MDD）==")
    srows = []; keep = {}
    for c in CANDS:
        TT = groups[c[:3]]
        for lab, kw in [("u.05", dict(u=.05)), ("u.10", dict(u=.10)), ("u.25", dict(u=.25)), ("u.50", dict(u=.5)), ("u.75", dict(u=.75)), ("u1.0", dict(u=1.0)), ("現金擔保", dict(cash=True))]:
            pc, wc, rec, calls, cc = sized(TT, c[3], days, **kw); keep[(c, lab)] = (pc, wc, rec)
            row = dict(cfg=str(c), size=lab, margin_calls=calls, calls_close=cc, ruin=bool(len(rec) and (CAP0 + pc.sum()) <= 0),
                       avg_n=rec.n.mean() if len(rec) else 0, min_eq_req=rec.minratio.min() if len(rec) else np.nan)
            for nm, a, b in (("FULL", PRE, END + pd.Timedelta(days=1)), ("IS", IS0, IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1))):
                for kk, v in perf(pc, wc, rec, a, b).items(): row[f"{nm}_{kk}"] = v
            srows.append(row)
    SR = pd.DataFrame(srows); SR.to_csv(RES / "dd_put_75_sizing.csv", index=False); print(SR.round(3).to_string(index=False))
    # GEX 半倉 / 持有中出場（u=.5）
    print("\n== 2b. GEX-A 減碼/出場（u=.5 資金）==")
    grows = []
    for c in CANDS[:6]:
        TT = groups[c[:3]]; sig = [t["s"] for t in TT]
        lowg = np.array([(gp.asof(s) < .3) if pd.notna(gp.asof(s)) else False for s in sig])
        for lab, kw in [("無", {}), ("GEX<.3 跳過", dict(mask=~lowg)), ("GEX<.3 半倉", dict(half=lowg)), ("持有中 GEX<.3 出場", dict(gexexit=gp))]:
            pc, wc, rec, calls, cc = sized(TT, c[3], days, u=.5, **kw)
            row = dict(cfg=str(c), rule=lab, calls=calls)
            for nm, a, b in (("IS", IS0, IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1))):
                for kk, v in perf(pc, wc, rec, a, b).items(): row[f"{nm}_{kk}"] = v
            grows.append(row)
    GR_ = pd.DataFrame(grows); GR_.to_csv(RES / "dd_put_75_gex.csv", index=False); print(GR_.round(3).to_string(index=False))
    # ---- 3. 尾部 ----
    print("\n== 3. 尾部視窗（u=.5 資金，視窗損益 % of 期初權益；1 組點數）==")
    trows = []
    for c in CANDS:
        pc, wc, rec = keep[(c, "u.50")]; pc1, wc1, _ = m74.daily(groups[c[:3]], c[3], days)
        for nm, a, b in TAILS:
            a, b = pd.Timestamp(a), pd.Timestamp(b); e0 = CAP0 + pc[pc.index < a].sum(); seg = pc[(pc.index >= a) & (pc.index <= b)]
            eq = e0 + seg.cumsum(); wseg = wc[(wc.index >= a) & (wc.index <= b)]
            trows.append(dict(cfg=str(c), window=nm, ret=seg.sum() / e0, worst_intraday=((eq.shift(fill_value=e0) + wseg) / e0 - 1).min(),
                              pts_1unit=pc1[(pc1.index >= a) & (pc1.index <= b)].sum() / MULT))
    TR = pd.DataFrame(trows); TR.to_csv(RES / "dd_put_75_tails.csv", index=False)
    print(TR.pivot(index="window", columns="cfg", values="ret").round(3).to_string())
    print(TR.pivot(index="window", columns="cfg", values="worst_intraday").round(3).to_string())
    for c in CANDS[:6]:
        pc, wc, rec = keep[(c, "u.50")]; eq = CAP0 + pc.cumsum(); mo = pc.groupby(pc.index.to_period("M")).sum() / eq.shift(fill_value=CAP0).groupby(pc.index.to_period("M")).first()
        print(f"最差 5 個月 {c}:", ", ".join(f"{k} {v:.1%}" for k, v in mo.nsmallest(5).items()))
    # ---- 4. covered call 對照（同期：2010–2019 / 2020–2026-08）----
    cc = pd.read_csv(RES / "cc_summary.csv")
    print("\n== 4. 對照 covered call（45，名目 1 倍、無槓桿）==")
    print(cc[cc.variant.isin(["TXF B&H", "CC ATM x4", "CC ATM x1", "CC +100 x4"]) | cc.variant.str.contains(r"3%|5%", regex=True)][["variant", "period", "CAGR", "Sharpe", "MDD_intraday"]].round(3).to_string(index=False))
    for c in CANDS[:6]:
        pc, wc, rec = keep[(c, "現金擔保")]
        for nm, a, b in (("IS 2010-19", PRE, pd.Timestamp("2020-01-01")), ("OOS 2020-26/08", pd.Timestamp("2020-01-01"), pd.Timestamp("2026-08-19"))):
            p = perf(pc, wc, rec, a, b); print(f"  現金擔保 put {c} {nm}: CAGR {p['CAGR']:.3f} Sharpe {p['Sharpe']:.2f} MDDw {p['MDDw']:.3f}")
    chart(keep, CANDS, TR, FR, best)


def chart(keep, CANDS, TR, FR, best):
    plt = plt_style(); cols = [GOLD, GR, "#e07b39", BLUE, "#9ad36a", "#d36ac4", WH, "#6ad3c9", "#ff6b6b"]
    fig, axs = plt.subplots(2, 1, figsize=(13, 10), facecolor=BG)
    ax = axs[0]; sty(ax)
    for c, col in zip(CANDS, cols):
        pc, wc, rec = keep[(c, "u.50")]; ax.plot((CAP0 + pc.cumsum()) / 1e4, color=col, lw=1.3, label=str(c))
    ax.set_yscale("log"); ax.axvline(IS0, color="white", ls="--", lw=.8); ax.axvline(IS_END, color="white", ls="--", lw=.8)
    ax.set_title("資金模擬：100 萬起、保證金使用率 50%、複利（萬元，對數）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=7, ncol=2)
    ax = axs[1]; sty(ax)
    for c, col in zip(CANDS[:6], cols):
        pc, wc, rec = keep[(c, "u.50")]; eq = CAP0 + pc.cumsum(); prev = eq.shift(fill_value=CAP0)
        ax.plot(((prev + wc) / np.maximum.accumulate(prev) - 1) * 100, color=col, lw=.9, label=str(c))
    ax.set_title("盤中最差回撤 %（u=.5）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=7)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "dd_put_75_sizing.png", dpi=110, facecolor=BG); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    ax = axs[0]; sty(ax); pv = TR.pivot(index="window", columns="cfg", values="worst_intraday") * 100
    pv = pv[[str(c) for c in CANDS[:6]]]; xx = np.arange(len(pv))
    for i, (c, col) in enumerate(zip(pv.columns, cols)): ax.bar(xx + i * .16 - .32, pv[c], .16, color=col, label=c)
    ax.set_xticks(xx); ax.set_xticklabels(pv.index, color=WH, rotation=20, fontsize=8); ax.set_title("尾部視窗盤中最差 %（u=.5）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=7)
    ax = axs[1]; sty(ax); g = FR.groupby("filt")[["IS_Sharpe", "OOS_Sharpe"]].mean(); xx = np.arange(len(g))
    ax.bar(xx - .2, g.IS_Sharpe, .4, color=GR, label="IS 2017–21"); ax.bar(xx + .2, g.OOS_Sharpe, .4, color=GOLD, label="OOS 2022+")
    ax.set_xticks(xx); ax.set_xticklabels(g.index, color=WH, rotation=30, fontsize=8); ax.set_title("濾網平均 Sharpe（候選 config 平均）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "dd_put_75_tails_filters.png", dpi=110, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
