"""49 低特質波動（IVOL）純多與 TXF 避險版本。
IVOL = 每檔股票日報酬對 TAIEX TR（data/index/taiex_daily.parquet tr_close）滾動 CAPM 迴歸殘差標準差；主視窗 252 日（至少 120 日，
與 48 --beta-study 相同），另報 21/63/126 日（至少 15/40/60 日）。
規則同 47/48：每月最後交易日收盤算訊號，t+1 開盤成交（spotsim 資料；漲停不能買、跌停不能賣、停牌順延、下市以最後價出清），
手續費 0.1425%×0.5、證交稅 0.3%、滑價 1 tick；流動性門檻 20 日均成交值 ≥ 5,000 萬（敏感度 0/2,000 萬/1 億）、股價 ≥ 10；rf 1.5%。
TXF：50 元/口 + 稅 2e-5 + 1 點、換月雙邊成本；TXF 報酬 ≈ TAIEX 價格指數期貨（m37.load）。
變體：
 (1) Q1（最低 IVOL 五分位）等權純多；十分位、三分位；IVOL 視窗 1m/3m/6m/12m。
 (2) Q1 − TXF 至 β=0：避險比率 = 目前持股過去 60/120 日實際 β（持股固定權重回推），對照 ex-ante FP β（48 的 TAIEX TR β）。
 (3) 部分避險至 β=0.5。
 (4) 狀態避險：TAIEX TR 收盤 < 200/250 日均線時才放空 TXF×β（每日檢查，訊號收盤、次日開盤調整避險）。
 (5) 波動目標 15%：每月以持股過去 60 日波動縮放（純多上限 1×；β0 避險版上限 2×，融資 6.5%）。
 (6) 動能過濾：排除 12-1 月動能最低 30%（在流動性合格股內）後再取 IVOL Q1。
比較：TAIEX TR B&H、0050 B&H、Q5 高 IVOL、等權全市場（流動性合格）。電子 / 金融（金融用三分位、至少 12 檔）簡報。
產業別 = data/stocks/_industry_map.csv 現行分類（下市股以 FinMind 補，無產業者排除於產業版）。
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd

_s = importlib.util.spec_from_file_location("m48", Path(__file__).with_name("48_bab_v2.py"))
m48 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m48)
m = m48.m; ROOT, OUT = m.ROOT, m.OUT
IS0, IS1 = m48.IS0, m48.IS1
PER = (("IS", IS0, IS1), ("OOS", IS1, None))


def sim(D, X, tg, hd=None, fin=m.FIN):
    """m47.sim + hd: {t: hedge ratio} hedge-only changes on non-rebalance days (filled next open)."""
    days, A, AO, P, PO, traded, lu, ld, last = (D[k] for k in ("days", "A", "AO", "P", "PO", "traded", "lu", "ld", "last"))
    nd, nS = A.shape; Xr = X.reindex(days)
    xo, xc, xg, xi, xroll = (Xr[k].values for k in ("o", "c", "gap", "intr", "roll"))
    w = np.zeros(nS); h = 0.0; pend = np.full(nS, np.nan); hp = np.nan; hd = hd or {}
    cols = ("ret", "long", "short", "hedge", "cash", "cost", "borrow", "turn")
    O = {k: np.zeros(nd) for k in cols}
    for t in range(1, nd):
        ro = np.nan_to_num(np.where(traded[t], AO[t] / A[t - 1] - 1, 0.0))
        hg = 0.0 if np.isnan(xg[t]) else xg[t]
        pl, ps = (w * ro)[w > 0].sum(), (w * ro)[w < 0].sum(); ph = h * hg
        c = 0.0; trn = 0.0
        g1 = 1 + pl + ps + ph; w = w * (1 + ro) / g1; h = h * (1 + hg) / g1
        if xroll[t] == True and h != 0 and np.isfinite(xo[t]):
            c += abs(h) * 2 * (2e-5 + (50 + 200) / (200 * xo[t]))
        ip = np.where(~np.isnan(pend))[0]
        if len(ip):
            d = pend[ip] - w[ip]; buy = d > 0; dl = t > last[ip]
            okx = (traded[t, ip] & ~(buy & lu[t, ip]) & ~(~buy & ld[t, ip])) | dl
            ii = ip[okx]
            if len(ii):
                px = np.where(traded[t, ii], PO[t, ii], P[t, ii]); px = np.where(np.isfinite(px) & (px > 0), px, 1.0)
                dd = pend[ii] - w[ii]; sl = m.tick(px) / px
                sc = np.where(dd > 0, m.COMM + sl, m.COMM + m.TAX + sl)
                c += (np.abs(dd) * sc).sum(); trn += np.abs(dd).sum()
                w[ii] = pend[ii]; pend[ii] = np.nan
        if np.isfinite(hp) and np.isfinite(xo[t]):
            c += abs(hp - h) * (2e-5 + (50 + 200) / (200 * xo[t])); h = hp; hp = np.nan
        rc = np.nan_to_num(np.where(traded[t], A[t] / AO[t] - 1, 0.0))
        hi_ = xi[t] * (xc[t - 1] if np.isfinite(xc[t - 1]) else xo[t]) / xo[t] if (np.isfinite(xi[t]) and np.isfinite(xo[t])) else 0.0
        hi_ = 0.0 if not np.isfinite(hi_) else hi_
        cash = 1 - w.sum(); cr = cash * (m.RF if cash > 0 else fin) / 252
        ql = (w * rc).sum(); qh = h * hi_
        g2 = 1 + ql + qh
        r = (1 + pl + ps + ph) * (1 - c) * g2 - 1 + cr
        w = w * (1 + rc) / g2; h = h * (1 + hi_) / g2
        for k, v in zip(cols, (r, pl + ql, ps, ph + qh, cr, -c, 0.0, trn)):
            O[k][t] = v
        if t in tg:
            tw, th = tg[t]; ch = (np.abs(tw - w) > 1e-12) | ~np.isnan(pend); pend = np.where(ch, tw, np.nan); hp = th
        elif t in hd:
            hp = hd[t]
    return pd.DataFrame(O, days)


def stats(o, rT):
    s = m.stats(o, rT); r = o["ret"] - m.RF / 252; x = rT.reindex(r.index).fillna(0) - m.RF / 252
    r3, x3 = r.rolling(3).sum()[2::3].values, x.rolling(3).sum()[2::3].values
    Xm = np.c_[np.ones_like(x3), x3]; cf, *_ = np.linalg.lstsq(Xm, r3, rcond=None); e = r3 - Xm @ cf
    se = np.sqrt(e.var(ddof=2) * np.linalg.inv(Xm.T @ Xm)[0, 0])
    s.update(Calmar=s["CAGR"] / -s["MDD"] if s["MDD"] < 0 else np.nan, alpha=cf[0] * 252 / 3, alpha_t=cf[0] / se, beta_reg=cf[1])
    return s


def main():
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    D = m.ss.prepare(m.sd); days = D["days"]; X = m.m37.load(); R = D["R"]
    rT, r50 = m48.idx("taiex_daily.parquet", "tr_close"), m.bench(); mv = rT.reindex(days).values
    BT = m.betas(D, rT)
    IV = {w: m48._ols(R, mv, w, mn, resid=True)[1] for w, mn in ((21, 15), (63, 40), (126, 60), (252, 120))}
    A = pd.DataFrame(D["A"]); MOM = (A.shift(21) / A.shift(252) - 1).values
    im = pd.read_csv(ROOT / "data/stocks/_industry_map.csv", dtype={"code": str}).set_index("code").reindex(D["cols"])
    EL = im.elec.fillna(False).astype(bool).values; FN = im.ind.fillna("NA").str.contains("金融").values
    reb = [t for t in np.where(np.r_[days[1:].month != days[:-1].month, False] & (days >= IS0 - pd.Timedelta(days=40)))[0] if t < len(days) - 1]
    Rn = np.nan_to_num(R)
    tr = pd.Series(np.nan_to_num(mv)).add(1).cumprod().values

    def pick(t, iv=252, q=0.2, side="low", mask=None, liq=m.LIQ, momx=None, min_n=20):
        s = IV[iv][t]; ok = D["traded"][t] & np.isfinite(s) & (D["t20"][t] >= liq) & (D["P"][t] >= m.PFLOOR)
        if mask is not None:
            ok &= mask
        if momx is not None:
            ok &= np.isfinite(MOM[t]); i0 = np.where(ok)[0]
            if len(i0) >= min_n:
                ok[i0[MOM[t][i0] < np.quantile(MOM[t][i0], momx)]] = False
        i = np.where(ok)[0]
        if len(i) < min_n:
            return None
        if side == "all":
            sel = i
        else:
            v = s[i]; sel = i[v <= np.quantile(v, q)] if side == "low" else i[v >= np.quantile(v, 1 - q)]
        w = np.zeros(R.shape[1]); w[sel] = 1 / len(sel); return w

    def pbeta(w, t, n):   # holdings-based trailing realized beta / vol
        if t < n:
            return np.nan, np.nan
        p = Rn[t - n + 1:t + 1] @ w; x = np.nan_to_num(mv[t - n + 1:t + 1])
        return np.cov(p, x)[0, 1] / x.var(), p.std() * 252 ** .5

    V, HI = {}, {}

    def run(name, hedge=None, bwin=120, tgt=0.0, regime=None, vt=None, cap=1.0, **pk):
        tg, hd, info = {}, {}, []
        bcur = 0.0
        for k, t in enumerate(reb):
            w = pick(t, **pk)
            if w is None:
                continue
            bx = np.nansum(w * np.nan_to_num(BT[t], nan=1.0)); br, pv = pbeta(w, t, bwin)
            b = bx if hedge == "exante" else br
            sc = 1.0
            if vt is not None:
                hv = pbeta(w, t, 60)[1] if hedge is None else np.std(Rn[t - 59:t + 1] @ w - b * np.nan_to_num(mv[t - 59:t + 1])) * 252 ** .5
                sc = min(cap, vt / hv) if hv > 0 else 1.0
            w = w * sc; b *= sc; bcur = b
            h = 0.0
            if hedge is not None:
                h = -(b - tgt * sc)
            if regime is not None:
                h = -b if tr[t] < pd.Series(tr[:t + 1]).iloc[-regime:].mean() else 0.0
            tg[t] = (w, h); info.append((days[t], int((w > 0).sum()), bx, br, h, sc))
            if regime is not None:   # daily regime checks until next rebalance
                t2 = reb[k + 1] if k + 1 < len(reb) else len(days) - 1
                st = h != 0
                for u in range(t + 1, t2):
                    if u < regime:
                        continue
                    on = tr[u] < tr[u - regime + 1:u + 1].mean()
                    if on != st:
                        hd[u] = -bcur if on else 0.0; st = on
        o = sim(D, X, tg, hd); V[name] = o
        HI[name] = pd.DataFrame(info, columns=["date", "n", "beta_exante", "beta_real", "hedge", "scale"]).set_index("date")

    # ---------------- full market
    run("Q1 low-IVOL 12m (main)"); run("Q5 high-IVOL 12m", side="high"); run("EW market (liq)", side="all")
    run("D1 low-IVOL decile", q=0.1); run("T1 low-IVOL tercile", q=1 / 3)
    for iv, nm in ((21, "1m"), (63, "3m"), (126, "6m")):
        run(f"Q1 low-IVOL {nm}", iv=iv)
    for nm, lq in (("liq0", 0), ("liq20M", 2e7), ("liq100M", 1e8)):
        run(f"Q1 low-IVOL {nm}", liq=lq)
    run("(2) Q1 − TXF β0 (real 60d)", hedge="real", bwin=60); run("(2) Q1 − TXF β0 (real 120d)", hedge="real", bwin=120)
    run("(2) Q1 − TXF β0 (ex-ante FP)", hedge="exante")
    run("(3) Q1 − TXF β0.5 (real 120d)", hedge="real", tgt=0.5)
    run("(4) Q1 regime hedge MA200", regime=200); run("(4) Q1 regime hedge MA250", regime=250)
    run("(5) Q1 vol-target 15% (cap 1)", vt=0.15); run("(5) Q1 − TXF β0 vol-target 15% (cap 2)", hedge="real", vt=0.15, cap=2.0)
    run("(5) Q1 vol-target 10% (cap 1)", vt=0.10)
    run("(6) Q1 ex bottom-30% mom", momx=0.3); run("(6) Q1 ex mom − TXF β0", momx=0.3, hedge="real")
    run("(6) Q1 ex mom regime MA250", momx=0.3, regime=250)
    # ---------------- sectors
    for u, mk, q, mn in (("ELEC", EL, 0.2, 20), ("FIN", FN, 1 / 3, 12)):
        run(f"[{u}] Q low-IVOL", mask=mk, q=q, min_n=mn); run(f"[{u}] Q high-IVOL", mask=mk, q=q, side="high", min_n=mn)
        run(f"[{u}] EW (liq)", mask=mk, side="all", min_n=mn); run(f"[{u}] Q low − TXF β0", mask=mk, q=q, min_n=mn, hedge="real")
        run(f"[{u}] Q low regime MA250", mask=mk, q=q, min_n=mn, regime=250)
    for k, r in (("TAIEX TR B&H", rT), ("0050 B&H", r50)):
        V[k] = pd.DataFrame({"ret": r.reindex(days).fillna(0)}).assign(long=lambda d: d.ret, short=0., hedge=0., cash=0., cost=0., borrow=0., turn=0.)
    rows = []
    for k, o in V.items():
        for p, lo, hi in PER + (("FULL", IS0, None),):
            rows.append(dict(name=k, period=p, **stats(m.win(o, lo, hi), rT)))
    S = pd.DataFrame(rows); S.to_csv(OUT / "ivol_summary.csv", index=False)
    cols = ["name", "CAGR", "vol", "sharpe", "MDD", "Calmar", "beta3d", "alpha", "alpha_t", "hedge_yr", "cost_yr", "turn_yr"]
    for p in ("IS", "OOS", "FULL"):
        print(f"\n== {p}"); print(S[S.period == p][cols].round(3).to_string(index=False))
    Y = pd.DataFrame({k: (1 + m.win(o, IS0, None).ret).groupby(m.win(o, IS0, None).index.year).prod() - 1 for k, o in V.items()})
    Y.to_csv(OUT / "ivol_yearly.csv")
    ky = ["Q1 low-IVOL 12m (main)", "(2) Q1 − TXF β0 (real 120d)", "(3) Q1 − TXF β0.5 (real 120d)", "(4) Q1 regime hedge MA250", "(5) Q1 vol-target 15% (cap 1)",
          "(6) Q1 ex bottom-30% mom", "Q5 high-IVOL 12m", "EW market (liq)", "TAIEX TR B&H", "0050 B&H"]
    print("\n== yearly"); print(Y[ky].round(3).to_string())
    HY = pd.DataFrame({k: m.win(V[k], IS0, None).hedge.groupby(m.win(V[k], IS0, None).index.year).sum() for k in ky[1:4]}); print("\n== yearly hedge P&L (sum of daily)"); print(HY.round(3).to_string())
    for k in ("Q1 low-IVOL 12m (main)", "(2) Q1 − TXF β0 (real 120d)", "(4) Q1 regime hedge MA250", "(5) Q1 vol-target 15% (cap 1)", "[FIN] Q low-IVOL"):
        h = HI[k]; print(k, "first", h.index[0].date(), h[["n", "beta_exante", "beta_real", "hedge", "scale"]].mean().round(3).to_dict(), "hedge-on months", round((h.hedge != 0).mean(), 2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for fn, keys in (("ivol_market", ky), ("ivol_variants", ["Q1 low-IVOL 12m (main)", "D1 low-IVOL decile", "T1 low-IVOL tercile", "Q1 low-IVOL 1m", "Q1 low-IVOL 3m", "Q1 low-IVOL 6m",
                                                         "(2) Q1 − TXF β0 (real 60d)", "(2) Q1 − TXF β0 (ex-ante FP)", "(4) Q1 regime hedge MA200", "(5) Q1 − TXF β0 vol-target 15% (cap 2)", "TAIEX TR B&H"]),
                     ("ivol_sectors", [k for k in V if k.startswith("[")] + ["TAIEX TR B&H"])):
        fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        for k in keys:
            e = (1 + m.win(V[k], IS0, None).ret).cumprod(); st = dict(c="k", lw=2) if "TAIEX" in k else dict(lw=1.2)
            ax[0].plot(e.index, e, label=k, **st); ax[1].plot(e.index, e / e.cummax() - 1, **st)
        ax[0].set_yscale("log"); ax[0].legend(fontsize=7); ax[1].set_ylabel("drawdown"); ax[0].set_title(f"49 low-IVOL ({fn}), next-open fills, costs incl.")
        for a in ax:
            a.axvline(IS1, c="gray", lw=.8)
        plt.tight_layout(); plt.savefig(OUT / f"{fn}.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
