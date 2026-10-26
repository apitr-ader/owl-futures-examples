"""48 BAB v2：市場代理改用 TAIEX 報酬指數（data/index/taiex_daily.parquet tr_close，tools/fetch_taiex.py 抓取），
並加入電子業 BAB 與產業中性 BAB。β 估計、權重、成交、成本與 47 相同（引用 47 的 betas/sim/stats）。
- 全市場：(a)-(d) 全部變體，β 對 TAIEX TR；TXF 避險以 TAIEX β 計（TXF ≈ TAIEX 價格指數期貨）。首次 β 約 2011-01（個股資料自 2008）。
- 電子業：產業 = data/stocks/_industry_map.csv（現行上市櫃產業別；下市股以 FinMind TaiwanStockInfo 補，79 檔下市股無產業 → 排除）。
  電子 = 半導體/電腦及週邊/光電/通信網路/電子零組件/電子通路/資訊服務/其他電子（含 FinMind「電子工業」）。上市櫃皆含。
  β 對電子類指數（價格指數，無 TR）；穩健性：β 對 TAIEX TR。避險：放空 TXF × 低 β 腿的 TAIEX β（電子類指數不可交易）。
- 金融業（--fin-only 只跑此段）：金融保險業（含下市股 FinMind「金融保險」），β 對金融類指數（價格指數）＋穩健性對 TAIEX TR；
  樣本小 → 最少 12 檔、以中位數/三分位取代五分位；(c) 空方用個股期貨；流動性門檻敏感度。
- --beta-study：比較 β 估計法（FP / OLS 1y,6m ± Vasicek / Blume / Dimson / 週 / 月 / 下檔 β / 總波動與特質波動排序 / 事後實現 β 調整腿槓桿），
  預測力（下 63 日 Dimson 實現 β）、BAB 多空、Q1 vs Q5 alpha 差。--beta-method NAME 以該估計法取代主程式 TAIEX β。
- 產業中性：β 在各產業內百分位排名後再做 rank 權重（多空在每個產業內大致平衡）。
期間：IS 2011-2019（另報 IS 2012-2019 以便與 47 比較），OOS 2020-01 ~ 2026-10-22。
"""
import importlib.util, sys, argparse
from pathlib import Path
import numpy as np, pandas as pd

_s = importlib.util.spec_from_file_location("m47", Path(__file__).with_name("47_bab.py"))
m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m)
ROOT, OUT = m.ROOT, m.OUT
IS0, IS12, IS1 = pd.Timestamp("2011-01-01"), pd.Timestamp("2012-01-01"), pd.Timestamp("2020-01-01")
PER = (("IS", IS0, IS1), ("IS12", IS12, IS1), ("OOS", IS1, None))


def idx(path, col):
    d = pd.read_parquet(ROOT / "data/index" / path).set_index("date")[col].dropna()
    return d.pct_change().dropna()


def targets(D, B, t, mask=None, kind="rank", mode="ls", grp=None, BH=None, SF=None, liq=m.LIQ, pf=m.PFLOOR, min_n=20, sig=None, rh=None):
    """sig: optional sort signal (e.g. vol) — rank on sig, scale with B. rh: daily market returns -> scale legs by trailing realized leg beta."""
    b = B[t]; ok = D["traded"][t] & np.isfinite(b) & (D["t20"][t] >= liq) & (D["P"][t] >= pf)
    if sig is not None:
        ok &= np.isfinite(sig[t])
    if mask is not None:
        ok &= mask
    i = np.where(ok)[0]
    if len(i) < min_n:
        return None
    bb = b[i]; sb = sig[t][i] if sig is not None else bb
    if grp is not None:   # within-industry percentile ranks
        z = pd.Series(sb).groupby(grp[i]).rank(pct=True).values
    else:
        z = pd.Series(sb).rank().values
    zb = z.mean()
    if kind == "rank":
        k = 2 / np.abs(z - zb).sum(); wL = k * np.clip(zb - z, 0, None); wH = k * np.clip(z - zb, 0, None)
    else:
        q = {"ew": 0.5, "q3": 1 / 3, "q5": 0.2}[kind]
        lo, hi = sb <= np.quantile(sb, q), sb >= np.quantile(sb, 1 - q); wL, wH = lo / lo.sum(), hi / hi.sum()
    if SF is not None:
        s = SF[t, i] & (z > zb)
        wH = np.nan_to_num(np.where(s, pd.Series(np.where(s, bb, np.nan)).rank().values, 0)); wH = wH / wH.sum() if wH.sum() > 0 else wH
    bL, bH = max(wL @ bb, 0.3), max(wH @ bb, 0.3)   # floor: unshrunk OLS/Dimson leg betas can be ~0 or negative -> 1/β explodes
    if rh is not None and t >= 252:   # ex-post: trailing 1y realized beta of each (constant-weight) leg
        Rw = np.nan_to_num(D["R"][t - 251:t + 1, i]); mm = rh[t - 251:t + 1]
        lb = lambda w_: max(np.cov(Rw @ w_, mm)[0, 1] / mm.var(), 0.2)
        bL, bH = lb(wL), lb(wH)
    hb = wL @ np.nan_to_num(BH[t, i], nan=1.0) if BH is not None else bL      # low leg beta vs TAIEX for TXF hedge
    w = np.zeros(len(b)); h = 0.0
    if mode == "ls":
        w[i] = wL / bL - (wH / bH if bH > 0 else 0)
    elif mode == "lhedge":
        w[i] = wL / bL; h = -hb / bL
    elif mode == "lhedge_unlev":
        w[i] = wL; h = -hb
    elif mode == "long_lev":
        w[i] = wL / bL
    elif mode == "long":
        w[i] = wL
    elif mode == "high":
        w[i] = wH
    elif mode == "all":
        w[i] = 1 / len(i)
    return w, h, bL, bH


# ======================= alternative beta estimators (--beta-method / --beta-study) =======================
def _rs(a, w):
    c = np.cumsum(a, 0); o = c.copy(); o[w:] = c[w:] - c[:-w]; return o


def _ols(Y, x, w, minn, extra=None, resid=False):
    """rolling OLS slope of each column of Y on x (window w rows, pairwise-complete); optional residual std."""
    V = np.isfinite(Y) & np.isfinite(x)[:, None]
    if extra is not None:
        V &= extra[:, None]
    Yz = np.where(V, Y, 0.0); Xz = np.where(V, x[:, None], 0.0)
    n = _rs(V.astype(float), w); sx, sy = _rs(Xz, w), _rs(Yz, w); sxx, sxy = _rs(Xz * Xz, w), _rs(Xz * Yz, w)
    with np.errstate(all="ignore"):
        vx = sxx - sx * sx / n; b = (sxy - sx * sy / n) / vx
        b[(n < minn) | (vx <= 0)] = np.nan
        if not resid:
            return b
        vy = (_rs(Yz * Yz, w) - sy * sy / n) / (n - 1); ve = vy - b * b * vx / (n - 1)
        return b, np.sqrt(np.clip(ve, 0, None)), np.sqrt(np.clip(vy, 0, None))


def _lag(a, k):
    o = np.full_like(a, np.nan)
    if k > 0:
        o[k:] = a[:-k]
    else:
        o[:] = a
    return o


def _dimson(R, mv, w, minn, leads=1):
    b = _ols(R, mv, w, minn) + _ols(R, _lag(mv, 1), w, minn)        # lag -1 (stock reacts late to market)
    for k in range(1, leads + 1):                                     # lead +k: pair r_{t-k} with m_t (no look-ahead)
        b = b + _ols(_lag(R, k), mv, w, minn)
    return b


def _resampled(D, mv, freq, w, minn):
    days = D["days"]; key = days.to_period(freq); last = np.r_[key[1:] != key[:-1], True]
    L = np.log1p(D["R"]); grp = np.cumsum(np.r_[0, last[:-1]])
    Lp = pd.DataFrame(L).groupby(grp).sum(min_count=1).values; Mp = pd.Series(np.log1p(mv)).groupby(grp).sum(min_count=1).values
    b = _ols(np.expm1(Lp), np.expm1(Mp), w, minn)
    out = np.full(D["R"].shape, np.nan); out[last] = b
    return pd.DataFrame(out).ffill().values


def beta_methods(D, rT):
    mv = rT.reindex(D["days"]).values; R = D["R"]
    ols1, ivol, tvol = _ols(R, mv, 252, 120, resid=True)
    M = {"FP (5y3d corr×1y vol, Vasicek .6)": lambda: m.betas(D, rT),
         "OLS daily 1y": lambda: ols1, "OLS daily 1y Vasicek .6": lambda: 0.6 * ols1 + 0.4, "OLS daily 1y Blume .67": lambda: 0.67 * ols1 + 0.33,
         "OLS daily 6m": lambda: _ols(R, mv, 126, 60), "OLS daily 6m Vasicek .6": lambda: 0.6 * _ols(R, mv, 126, 60) + 0.4,
         "Dimson(-1,0,+1) 1y": lambda: _dimson(R, mv, 252, 120, 1), "Dimson(-1..+2) 1y": lambda: _dimson(R, mv, 252, 120, 2),
         "Dimson(-1,0,+1) 1y Vasicek .6": lambda: 0.6 * _dimson(R, mv, 252, 120, 1) + 0.4,
         "Weekly 2y": lambda: _resampled(D, mv, "W-FRI", 104, 52), "Weekly 3y": lambda: _resampled(D, mv, "W-FRI", 156, 78),
         "Monthly 60m": lambda: _resampled(D, mv, "M", 60, 36),
         "Downside 1y (mkt<0 days)": lambda: _ols(R, mv, 252, 50, extra=np.nan_to_num(mv) < 0),
         "SORT total vol 1y": lambda: tvol, "SORT idio vol 1y (CAPM resid)": lambda: ivol}
    return M


def _alpha(o, rT):
    r = o["ret"] - m.RF / 252; x = rT.reindex(r.index).fillna(0) - m.RF / 252
    r3, x3 = r.rolling(3).sum()[2::3], x.rolling(3).sum()[2::3]
    b, a = np.polyfit(x3, r3, 1); return a * 252 / 3, b


def beta_study(D, X, rT, BT, SF, EL, FN, reb):
    days = D["days"]; mv = rT.reindex(days).values
    d63 = _dimson(D["R"], mv, 63, 40, 1); RB = np.full_like(d63, np.nan); RB[:-63] = d63[63:]   # realized Dimson beta over next 63 days
    MS = beta_methods(D, rT); rows, curves, pred = [], {}, []
    for name, f in MS.items():
        Bx = f(); is_sort = name.startswith("SORT")
        Bsc, sig = (BT, Bx) if is_sort else (Bx, None)
        # (i) predictive accuracy on liquid universe
        if not is_sort:
            for per, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, pd.Timestamp("2099-01-01"))):
                sl, r2, q1, q5, r1, r5 = [], [], [], [], [], []
                for t in reb:
                    if not (lo <= days[t] < hi):
                        continue
                    ok = D["traded"][t] & (D["t20"][t] >= m.LIQ) & (D["P"][t] >= m.PFLOOR) & np.isfinite(Bx[t]) & np.isfinite(RB[t])
                    if ok.sum() < 30:
                        continue
                    xb, yb = Bx[t][ok], RB[t][ok]; yb = np.clip(yb, -2, 5)
                    c = np.polyfit(xb, yb, 1); sl.append(c[0]); r2.append(np.corrcoef(xb, yb)[0, 1] ** 2)
                    qa, qb = np.quantile(xb, [.2, .8]); q1.append(xb[xb <= qa].mean()); r1.append(yb[xb <= qa].mean()); q5.append(xb[xb >= qb].mean()); r5.append(yb[xb >= qb].mean())
                pred.append(dict(method=name, period=per, slope=np.mean(sl), R2=np.mean(r2), Q1_exante=np.mean(q1), Q1_real=np.mean(r1), Q5_exante=np.mean(q5), Q5_real=np.mean(r5)))
        for uni, mk in (("ALL", dict()), ("ELEC", dict(mask=EL)), ("FIN", dict(mask=FN, min_n=12))):
            qk = "q3" if uni == "FIN" else "q5"
            def run(**kw):
                tg = {}
                for t in reb:
                    x = targets(D, Bsc, t, sig=sig, **mk, **kw)
                    if x is not None:
                        tg[t] = x[:2]
                return m.sim(D, X, tg)
            O = {"BAB": run(), "Qlow": run(kind=qk, mode="long"), "Qhigh": run(kind=qk, mode="high")}
            if uni == "ALL" and name.startswith("FP"):
                O["BAB ex-post leg-β scaling"] = run(rh=np.nan_to_num(mv))
            for p, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, None)):
                st = {k: m.stats(m.win(o, lo, hi), rT) for k, o in O.items()}; al = {k: _alpha(m.win(o, lo, hi), rT) for k, o in O.items()}
                for k in O:
                    if k.startswith("BAB"):
                        rows.append(dict(method=name if k == "BAB" else name + " | ex-post leg-β scaling", univ=uni, period=p, CAGR=st[k]["CAGR"], sharpe=st[k]["sharpe"], MDD=st[k]["MDD"],
                                         beta3d=st[k]["beta3d"], alpha=al[k][0], cost=st[k]["cost_yr"],
                                         Qlow_CAGR=st["Qlow"]["CAGR"], Qhigh_CAGR=st["Qhigh"]["CAGR"], Qlow_beta=al["Qlow"][1], Qhigh_beta=al["Qhigh"][1],
                                         Qlow_alpha=al["Qlow"][0], Qhigh_alpha=al["Qhigh"][0], alpha_diff=al["Qlow"][0] - al["Qhigh"][0]))
            if uni == "ALL":
                curves[name] = (1 + m.win(O["BAB"], IS0, None).ret).cumprod()
                if "BAB ex-post leg-β scaling" in O:
                    curves[name + " | ex-post leg-β"] = (1 + m.win(O["BAB ex-post leg-β scaling"], IS0, None).ret).cumprod()
        print("done", name, flush=True)
    P = pd.DataFrame(pred); S = pd.DataFrame(rows)
    P.to_csv(OUT / "bab3_beta_pred.csv", index=False); S.to_csv(OUT / "bab3_beta_methods.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    print("\n== (i) predictive accuracy: next-63d realized Dimson beta on ex-ante beta (monthly cross-sections, means)"); print(P.round(3).to_string(index=False))
    for u in ("ALL", "ELEC", "FIN"):
        for p in ("IS", "OOS"):
            print(f"\n== {u} {p}"); print(S[(S.univ == u) & (S.period == p)].drop(columns=["univ", "period"]).round(3).to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    for k, e in curves.items():
        ax[0].plot(e.index, e, lw=2.2 if "ex-post" in k else 1, label=k)
    ax[0].set_yscale("log"); ax[0].axvline(IS1, c="gray"); ax[0].legend(fontsize=7); ax[0].set_title("BAB rank L/S (ALL), by beta estimator (TAIEX TR)")
    a = S[(S.univ == "ALL") & ~S.method.str.contains("ex-post")].pivot(index="method", columns="period", values="alpha_diff")
    a.plot.barh(ax=ax[1]); ax[1].axvline(0, c="k"); ax[1].set_title("ALL: alpha(Q1 low) − alpha(Q5 high), ann., EW long-only")
    plt.tight_layout(); plt.savefig(OUT / "bab3_beta_methods.png", dpi=110); plt.close()



def main():
    pd.set_option("display.width", 260)
    D = m.ss.prepare(m.sd); days = D["days"]; X = m.m37.load()
    rT, rE, r50 = idx("taiex_daily.parquet", "tr_close"), idx("sector_electronics_daily.parquet", "close"), m.bench()
    rF = idx("sector_finance_daily.parquet", "close")
    BT, BE, B50, BF = m.betas(D, rT), m.betas(D, rE), m.betas(D, r50), m.betas(D, rF)
    SF = m.sf_mask(D)
    im = pd.read_csv(ROOT / "data/stocks/_industry_map.csv", dtype={"code": str}).set_index("code").reindex(D["cols"])
    EL = im.elec.fillna(False).astype(bool).values; GRP = im.ind.fillna("NA").values; KNOWN = im.ind.notna().values
    reb = [t for t in np.where(np.r_[days[1:].month != days[:-1].month, False] & (days >= IS0 - pd.Timedelta(days=40)))[0] if t < len(days) - 1]
    ap = argparse.ArgumentParser(); ap.add_argument("--fin-only", action="store_true"); ap.add_argument("--beta-study", action="store_true")
    ap.add_argument("--beta-method", default=None, help="replace TAIEX-TR FP beta in main runs with one of beta_methods() keys")
    A = ap.parse_args()
    if A.beta_study:
        return beta_study(D, X, rT, BT, SF, EL, FN_ := pd.Series(GRP).str.contains("金融").values, reb)
    if A.beta_method:
        BT = beta_methods(D, rT)[A.beta_method]()

    def run(B, name, sim_kw=None, **kw):
        tg, info = {}, []
        for t in reb:
            x = targets(D, B, t, **kw)
            if x is not None:
                tg[t] = x[:2]; info.append((days[t], x[2], x[3], int((x[0] > 0).sum()), int((x[0] < 0).sum())))
        o = m.sim(D, X, tg, **(sim_kw or {})); o.attrs["info"] = pd.DataFrame(info, columns=["date", "bL", "bH", "nL", "nS"]).set_index("date")
        V[name] = o

    V = {}
    FINONLY = "--fin-only" in sys.argv
    for tag, B in (() if FINONLY else (("T", BT), ("50", B50))):
        run(B, f"[{tag}] (a) BAB rank (paper)")
        if tag == "50":
            continue
        run(B, f"[{tag}] (a) BAB EW median", kind="ew"); run(B, f"[{tag}] (a) BAB EW quintile", kind="q5")
        for nm, lq in (("liq0", 0), ("liq20M", 2e7), ("liq100M", 1e8)):
            run(B, f"[{tag}] (a) BAB rank {nm}", liq=lq)
        run(B, f"[{tag}] (a) BAB rank borrow 4%", sim_kw=dict(borrow=0.04))
        run(B, f"[{tag}] (b) low-β ×1/βL − TXF", mode="lhedge")
        run(B, f"[{tag}] (b2) low-β unlev − TXF×βL", mode="lhedge_unlev")
        run(B, f"[{tag}] (c) long spot / short SF", SF=SF, sim_kw=dict(sfshort=True))
        run(B, f"[{tag}] (d) long low-β unlev", mode="long")
        run(B, f"[{tag}] (d) long low-β lev (fin 6.5%)", mode="long_lev")
        run(B, f"[{tag}] (d') long high-β unlev", mode="high")
        run(B, f"[{tag}] sector-neutral BAB rank", mask=KNOWN, grp=GRP)
        run(B, f"[{tag}] sector-neutral (b2) − TXF×βL", mask=KNOWN, grp=GRP, mode="lhedge_unlev")
    for tag, B in (() if FINONLY else (("E", BE), ("E|T", BT))):
        run(B, f"[{tag}] ELEC BAB rank", mask=EL)
        run(B, f"[{tag}] ELEC BAB EW quintile", mask=EL, kind="q5")
        run(B, f"[{tag}] ELEC (b2) low-β unlev − TXF×βL(TAIEX)", mask=EL, mode="lhedge_unlev", BH=BT)
        run(B, f"[{tag}] ELEC (d) long low-β unlev", mask=EL, mode="long")
        run(B, f"[{tag}] ELEC (d') long high-β unlev", mask=EL, mode="high")
    if not FINONLY:
        run(BE, "[E] ELEC EW all (liq-filtered)", mask=EL, mode="all")
        run(BE, "[E] ELEC sub-industry-neutral BAB", mask=EL, grp=GRP)
    # ---- FINANCE-ONLY BAB (金融保險業; small universe -> min 12 names, median/tercile instead of quintile)
    FN = pd.Series(GRP).str.contains("金融").values
    nf = [int((FN & D["traded"][t] & np.isfinite(BF[t]) & (D["t20"][t] >= m.LIQ) & (D["P"][t] >= m.PFLOOR)).sum()) for t in reb]
    print("finance names passing filter per rebalance: min/median/max", min(nf), int(np.median(nf)), max(nf), "| in panel", int(FN.sum()))
    fk = dict(mask=FN, min_n=12)
    for tag, B in (("F", BF), ("F|T", BT)):
        run(B, f"[{tag}] FIN BAB rank", **fk)
        run(B, f"[{tag}] FIN BAB EW median", kind="ew", **fk)
        run(B, f"[{tag}] FIN BAB EW tercile", kind="q3", **fk)
        run(B, f"[{tag}] FIN (b2) low-β unlev − TXF×βL(TAIEX)", mode="lhedge_unlev", BH=BT, **fk)
        run(B, f"[{tag}] FIN (c) long spot / short SF", SF=SF, sim_kw=dict(sfshort=True), **fk)
        run(B, f"[{tag}] FIN (d) long low-β unlev", mode="long", **fk)
        run(B, f"[{tag}] FIN (d') long high-β unlev", mode="high", **fk)
    for nm, lq in (("liq0", 0), ("liq20M", 2e7), ("liq100M", 1e8), ("liq200M", 2e8)):
        run(BF, f"[F] FIN BAB rank {nm}", mask=FN, min_n=12, liq=lq)
    run(BF, "[F] FIN EW all (liq-filtered)", mode="all", **fk)
    for k, r in (("TAIEX TR B&H", rT), ("ELEC index B&H (price)", rE), ("FIN index B&H (price)", rF), ("0050 B&H", r50)):
        V[k] = pd.DataFrame({"ret": r.reindex(days).fillna(0)}).assign(long=lambda d: d.ret, short=0., hedge=0., cash=0., cost=0., borrow=0., turn=0.)
    rows = []
    for k, o in V.items():
        for p, lo, hi in PER:
            st = m.stats(m.win(o, lo, hi), rT)
            if k.startswith("[F") or "FIN" in k:
                st["beta_sect"] = m.stats(m.win(o, lo, hi), rF)["beta3d"]
            if k.startswith("[E") or "ELEC" in k:
                st["beta_elec"] = m.stats(m.win(o, lo, hi), rE)["beta3d"]
            rows.append(dict(name=k, period=p, **st))
    S = pd.DataFrame(rows); S.to_csv(OUT / ("bab2_fin_summary.csv" if FINONLY else "bab2_summary.csv"), index=False)
    cols = ["name", "CAGR", "vol", "sharpe", "MDD", "beta3d", "beta_elec", "beta_sect", "long_yr", "short_yr", "hedge_yr", "cost_yr", "turn_yr"]
    for p in ("IS", "IS12", "OOS"):
        print(f"\n== {p}"); print(S[S.period == p][[c for c in cols if c in S]].round(3).to_string(index=False))
    Y = pd.DataFrame({k: (1 + m.win(o, IS0, None).ret).groupby(m.win(o, IS0, None).index.year).prod() - 1 for k, o in V.items()})
    Y.to_csv(OUT / ("bab2_fin_yearly.csv" if FINONLY else "bab2_yearly.csv"))
    print(Y[[c for c in Y if "liq" not in c and "borrow" not in c]].round(3).T.to_string())
    for k in ("[T] (a) BAB rank (paper)", "[E] ELEC BAB rank", "[F] FIN BAB rank", "[F|T] FIN BAB rank"):
        if k not in V:
            continue
        i = V[k].attrs["info"]; print(k, "first reb", i.index[0].date(), i[["bL", "bH", "nL"]].mean().round(2).to_dict())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for fn, keys in (("bab2_market", ["[T] (a) BAB rank (paper)", "[50] (a) BAB rank (paper)", "[T] (b2) low-β unlev − TXF×βL", "[T] (c) long spot / short SF",
                                       "[T] (d) long low-β unlev", "[T] sector-neutral BAB rank", "TAIEX TR B&H", "0050 B&H"]),
                     ("bab2_elec", ["[E] ELEC BAB rank", "[E|T] ELEC BAB rank", "[E] ELEC sub-industry-neutral BAB", "[E] ELEC (b2) low-β unlev − TXF×βL(TAIEX)",
                                    "[E] ELEC (d) long low-β unlev", "[E] ELEC (d') long high-β unlev", "[E] ELEC EW all (liq-filtered)", "ELEC index B&H (price)", "TAIEX TR B&H"]),
                     ("bab2_fin", ["[F] FIN BAB rank", "[F|T] FIN BAB rank", "[F] FIN BAB EW tercile", "[F] FIN (b2) low-β unlev − TXF×βL(TAIEX)", "[F] FIN (c) long spot / short SF",
                                   "[F] FIN (d) long low-β unlev", "[F] FIN (d') long high-β unlev", "[F] FIN EW all (liq-filtered)", "FIN index B&H (price)", "TAIEX TR B&H"])):
        if not all(k in V for k in keys):
            continue
        fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        for k in keys:
            e = (1 + m.win(V[k], IS0, None).ret).cumprod(); st = dict(c="k", lw=2) if "B&H" in k else dict(lw=1.2)
            ax[0].plot(e.index, e, label=k, **st); ax[1].plot(e.index, e / e.cummax() - 1, **st)
        ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[1].set_ylabel("drawdown"); ax[0].set_title(f"48 BAB v2 ({fn}), next-open fills, costs incl.")
        for a in ax:
            a.axvline(IS1, c="gray", lw=.8)
        plt.tight_layout(); plt.savefig(OUT / f"{fn}.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
