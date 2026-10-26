"""47 Betting Against Beta（Frazzini & Pedersen 2014, JFE）台股現股回測。

Beta（論文作法）：σ_i、σ_m = 過去 1 年（252 日、至少 120 日）日 log 報酬標準差；ρ = 過去 5 年（1260 日、至少 750 筆）
重疊 3 日 log 報酬相關係數；β_TS = ρ σ_i/σ_m；Vasicek 縮減 β = 0.6 β_TS + 0.4×1。
市場代理：0050（Yahoo adjclose，data/bench/0050.json，自 2009-01 起；無 TAIEX/市值資料）→ 750 筆門檻使首次 β 約 2012-01。
組合：每月最後交易日收盤算訊號，t+1 開盤成交（漲停不能買、跌停不能賣，隔日重試；停牌隔日重試；下市以最後價出清）。
 rank 權重：z=β 排名，k=2/Σ|z−z̄|，w_H=k(z−z̄)+、w_L=k(z−z̄)−；多 w_L/β_L、空 w_H/β_H（兩腿事前 β=1）；
 現金 = 1 − Σw（含放空所得）賺 rf，<0 時以融資利率 FIN 付息（論文假設以 rf 融資）。
變體：(a) paper rank/中位數；EW 中位數；EW 五分位。(b) 低 β 現股 ×1/β_L + 放空 TXF 1×NAV；(b2) 低 β 不槓桿 + 放空 TXF β_L×NAV。
 (c) 空頭腿只用股票期貨（當時有掛牌的 S 類標的；以現股報酬近似期貨報酬、期貨成本、無借券費）。(d) 純多低 β（不槓桿 / 槓桿至 β=1）。
成本：手續費 0.1425%×0.5/邊、證交稅 0.3%（賣出與放空賣出時）、滑價 1 tick/邊；借券費 BORROW/年（預設 2%）；
 股期 50 元/口/邊（2000 股）+ 稅 2e-5 + 1 tick；TXF 50 元/口 + 稅 2e-5 + 1 點，換月雙邊成本。
未模型化：平盤下不得放空、融券強制回補（股東會/除權息停券）、券源不足、期貨基差。
IS 2012-2019（受 β 暖身限制），OOS 2020-01 ~ 2026-10-22。
"""
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/workspace/fubon-threads")
sys.path.insert(0, str(ROOT / "tools")); import stockdata as sd  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools")); import spotsim as ss  # noqa
_s = importlib.util.spec_from_file_location("m37", Path(__file__).with_name("37_txf_vol_target.py"))
m37 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m37)
OUT = ROOT / "results"
RF, FIN, BORROW, COMM, TAX, LIQ, PFLOOR = 0.015, 0.065, 0.02, 0.001425 * 0.5, 0.003, 5e7, 10.0
IS0, IS1 = pd.Timestamp("2012-01-01"), pd.Timestamp("2020-01-01")
PER = (("IS", IS0, IS1), ("OOS", IS1, None), ("FULL", IS0, None))
tick = ss.tick


def bench():
    d = json.load(open(ROOT / "data/bench/0050.json"))["chart"]["result"][0]
    s = pd.Series(d["indicators"]["adjclose"][0]["adjclose"], pd.to_datetime(d["timestamp"], unit="s") + pd.Timedelta(hours=8)).dropna()
    s.index = s.index.normalize(); s = s[~s.index.duplicated(keep="last")]
    r = s.pct_change().dropna(); return r[r.abs() < 0.4]


def betas(D, rm):
    days = D["days"]; l = np.log1p(pd.DataFrame(D["R"], days))
    lm = np.log1p(rm.reindex(days))
    l3, m3 = l.rolling(3).sum(), lm.rolling(3).sum()
    rho = l3.rolling(1260, min_periods=750).corr(m3)
    si = l.rolling(252, min_periods=120).std(); sm = lm.rolling(252, min_periods=120).std()
    return (0.6 * rho.mul(1, axis=0) * si.div(sm, axis=0) + 0.4).values


def sf_mask(D):
    m = pd.read_csv(ROOT / "data/stock_futures_daily/products_map.csv", dtype={"underlying_code": str})
    m = m[(m.underlying_type == "S") & m.underlying_code.notna()]
    cols = {c: i for i, c in enumerate(D["cols"])}; M = np.zeros((len(D["days"]), len(cols)), bool)
    for x in m.itertuples():
        if x.underlying_code in cols:
            M[(D["days"] >= x.first) & (D["days"] <= x.last), cols[x.underlying_code]] = True
    return M


def targets(D, B, t, kind="rank", liq=LIQ, pf=PFLOOR, mode="ls", SF=None):
    b = B[t]; ok = D["traded"][t] & np.isfinite(b) & (D["t20"][t] >= liq) & (D["P"][t] >= pf)
    i = np.where(ok)[0]
    if len(i) < 20:
        return None
    bb = b[i]; z = pd.Series(bb).rank().values; zb = z.mean()
    if kind == "rank":
        k = 2 / np.abs(z - zb).sum(); wL = k * np.clip(zb - z, 0, None); wH = k * np.clip(z - zb, 0, None)
    else:
        q = 0.5 if kind == "ew" else 0.2
        lo, hi = bb <= np.quantile(bb, q), bb >= np.quantile(bb, 1 - q); wL, wH = lo / lo.sum(), hi / hi.sum()
    if SF is not None:                                   # short leg restricted to stock-futures names
        s = SF[t, i] & (z > zb)  # rank weights among SF names above median
        wH = np.nan_to_num(np.clip(np.where(s, pd.Series(np.where(s, bb, np.nan)).rank().values, 0), 0, None))
        wH = wH / wH.sum() if wH.sum() > 0 else wH
    bL, bH = wL @ bb, wH @ bb
    w = np.zeros(len(b)); h = 0.0
    if mode == "ls":
        w[i] = wL / bL - (wH / bH if bH > 0 else 0)
    elif mode == "lhedge":
        w[i] = wL / bL; h = -1.0
    elif mode == "lhedge_unlev":
        w[i] = wL; h = -bL
    elif mode == "long_lev":
        w[i] = wL / bL
    elif mode == "long":
        w[i] = wL
    elif mode == "high":
        w[i] = wH
    return w, h, bL, bH


def sim(D, X, tg, borrow=BORROW, fin=FIN, sfshort=False):
    days, A, AO, P, PO, traded, lu, ld, last = (D[k] for k in ("days", "A", "AO", "P", "PO", "traded", "lu", "ld", "last"))
    nd, nS = A.shape; Xr = X.reindex(days)
    xo, xc, xg, xi, xroll = (Xr[k].values for k in ("o", "c", "gap", "intr", "roll"))
    w = np.zeros(nS); h = 0.0; pend = np.full(nS, np.nan); hp = np.nan
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
                dd = pend[ii] - w[ii]; sl = tick(px) / px
                if sfshort:   # short-side trades (w<0 before or after) priced as stock futures
                    fut = (pend[ii] < 0) | (w[ii] < 0); fc = 50 / (2000 * px) + 2e-5 + sl
                    sc = np.where(fut, fc, np.where(dd > 0, COMM + sl, COMM + TAX + sl))
                else:
                    sc = np.where(dd > 0, COMM + sl, COMM + TAX + sl)
                c += (np.abs(dd) * sc).sum(); trn += np.abs(dd).sum()
                w[ii] = pend[ii]; pend[ii] = np.nan
        if np.isfinite(hp) and np.isfinite(xo[t]):
            c += abs(hp - h) * (2e-5 + (50 + 200) / (200 * xo[t])); trn += abs(hp - h); h = hp; hp = np.nan
        rc = np.nan_to_num(np.where(traded[t], A[t] / AO[t] - 1, 0.0)); hi_ = 0.0 if np.isnan(xi[t]) else xi[t] * (xc[t - 1] if np.isfinite(xc[t - 1]) else xo[t]) / xo[t] if np.isfinite(xo[t]) else 0.0
        hi_ = 0.0 if not np.isfinite(hi_) else hi_
        cash = 1 - w.sum(); cr = cash * (RF if cash > 0 else fin) / 252
        bf = 0.0 if sfshort else -borrow / 252 * (-w[w < 0]).sum()
        ql, qs = (w * rc)[w > 0].sum(), (w * rc)[w < 0].sum(); qh = h * hi_
        g2 = 1 + ql + qs + qh
        r = (1 + pl + ps + ph) * (1 - c) * g2 - 1 + cr + bf
        w = w * (1 + rc) / g2; h = h * (1 + hi_) / g2
        for k, v in zip(cols, (r, pl + ql, ps + qs, ph + qh, cr, -c, bf, trn)):
            O[k][t] = v
        if t in tg:
            tw, th = tg[t]; ch = (np.abs(tw - w) > 1e-12) | ~np.isnan(pend); pend = np.where(ch, tw, np.nan); hp = th
    return pd.DataFrame(O, days)


def win(r, lo, hi):
    return r[(r.index >= lo) & ((r.index < hi) if hi is not None else True)]


def stats(o, rm):
    r = o["ret"]; e = (1 + r).cumprod(); y = len(r) / 252; x = r - RF / 252
    m = rm.reindex(r.index).fillna(0); m3, r3 = m.rolling(3).sum()[::3], r.rolling(3).sum()[::3]
    return dict(CAGR=e.iloc[-1] ** (1 / y) - 1, vol=r.std() * 252 ** .5, sharpe=x.mean() / x.std() * 252 ** .5, MDD=(e / e.cummax() - 1).min(),
                beta=np.cov(r, m)[0, 1] / m.var(), beta3d=np.cov(r3.dropna(), m3.dropna())[0, 1] / m3.var(),
                long_yr=o["long"].mean() * 252, short_yr=o["short"].mean() * 252, hedge_yr=o["hedge"].mean() * 252,
                cost_yr=-o["cost"].mean() * 252, borrow_yr=-o["borrow"].mean() * 252, turn_yr=o["turn"].sum() / y)


def main():
    pd.set_option("display.width", 250)
    D = ss.prepare(sd); days = D["days"]; rm = bench(); X = m37.load()
    B = betas(D, rm); SF = sf_mask(D)
    reb = np.where(np.r_[days[1:].month != days[:-1].month, False] & (days >= IS0 - pd.Timedelta(days=40)))[0]
    reb = [t for t in reb if t < len(days) - 1]

    def T(**kw):
        tg, bl = {}, []
        for t in reb:
            x = targets(D, B, t, **kw)
            if x is not None:
                tg[t] = (x[0], x[1]); bl.append((days[t], x[2], x[3], int((x[0] > 0).sum()), int((x[0] < 0).sum())))
        return tg, pd.DataFrame(bl, columns=["date", "bL", "bH", "nL", "nS"]).set_index("date")

    V = {}
    tgr, binfo = T(kind="rank")
    V["(a) BAB rank (paper)"] = sim(D, X, tgr)
    V["(a) BAB rank, borrow 1%"] = sim(D, X, tgr, borrow=0.01)
    V["(a) BAB rank, borrow 4%"] = sim(D, X, tgr, borrow=0.04)
    V["(a) BAB rank, fin@rf"] = sim(D, X, tgr, fin=RF)
    V["(a) BAB EW median"] = sim(D, X, T(kind="ew")[0])
    V["(a) BAB EW quintile"] = sim(D, X, T(kind="q5")[0])
    for nm, lq in (("liq0", 0), ("liq20M", 2e7), ("liq100M", 1e8)):
        V[f"(a) BAB rank {nm}"] = sim(D, X, T(kind="rank", liq=lq)[0])
    V["(a) BAB rank liq50M no px floor"] = sim(D, X, T(kind="rank", pf=0)[0])
    V["(b) low-β ×1/βL − TXF"] = sim(D, X, T(kind="rank", mode="lhedge")[0])
    V["(b2) low-β unlev − TXF×βL"] = sim(D, X, T(kind="rank", mode="lhedge_unlev")[0])
    V["(c) long low-β spot / short SF high-β"] = sim(D, X, T(kind="rank", SF=SF)[0], sfshort=True)
    V["(d) long low-β unlev"] = sim(D, X, T(kind="rank", mode="long")[0])
    V["(d) long low-β lev β=1 (fin 6.5%)"] = sim(D, X, T(kind="rank", mode="long_lev")[0])
    V["(d) long low-β lev β=1 (fin@rf)"] = sim(D, X, T(kind="rank", mode="long_lev")[0], fin=RF)
    V["(d') long high-β unlev"] = sim(D, X, T(kind="rank", mode="high")[0])
    m50 = pd.DataFrame({"ret": rm.reindex(days).fillna(0)}).assign(long=lambda d: d.ret, short=0., hedge=0., cash=0., cost=0., borrow=0., turn=0.)
    V["0050 B&H"] = m50
    rows = []
    for k, o in V.items():
        for p, lo, hi in PER:
            rows.append(dict(name=k, period=p, **stats(win(o, lo, hi), rm)))
    S = pd.DataFrame(rows); S.to_csv(OUT / "bab_summary.csv", index=False)
    print(S[S.period != "FULL"].round(3).to_string(index=False))
    Y = pd.DataFrame({k: (1 + win(o, IS0, None).ret).groupby(win(o, IS0, None).index.year).prod() - 1 for k, o in V.items()})
    Y.to_csv(OUT / "bab_yearly.csv"); print(Y[[c for c in Y if "borrow" not in c and "liq" not in c and "fin@rf" not in c]].round(3).T.to_string())
    binfo.to_csv(OUT / "bab_betas.csv"); print(binfo.describe().round(2).to_string()); print("first rebalance", binfo.index[0].date(), "data end", days[-1].date())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = ["(a) BAB rank (paper)", "(a) BAB EW median", "(a) BAB EW quintile", "(b) low-β ×1/βL − TXF", "(b2) low-β unlev − TXF×βL",
            "(c) long low-β spot / short SF high-β", "(d) long low-β unlev", "(d) long low-β lev β=1 (fin 6.5%)", "0050 B&H"]
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for k in show:
        e = (1 + win(V[k], IS0, None).ret).cumprod(); st = dict(c="k", lw=2) if "0050" in k else dict(lw=1.2)
        ax[0].plot(e.index, e, label=k, **st); ax[1].plot(e.index, e / e.cummax() - 1, **st)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[1].set_ylabel("drawdown"); ax[0].set_title("47 Betting Against Beta, Taiwan spot (next-open fills, costs incl.)")
    for a in ax:
        a.axvline(IS1, c="gray", lw=.8)
    plt.tight_layout(); plt.savefig(OUT / "bab_equity_dd.png", dpi=110)
    fig, ax = plt.subplots(figsize=(13, 5)); o = win(V["(a) BAB rank (paper)"], IS0, None)
    for k in ("long", "short", "cash", "cost", "borrow"):
        ax.plot(o.index, o[k].cumsum(), label=k)
    ax.plot(o.index, o.ret.cumsum(), c="k", lw=2, label="total"); ax.legend(); ax.set_title("(a) BAB rank: cumulative arithmetic contribution by leg")
    plt.tight_layout(); plt.savefig(OUT / "bab_legs.png", dpi=110)
    fig, ax = plt.subplots(figsize=(13, 4)); binfo[["bL", "bH"]].plot(ax=ax); ax.set_title("ex-ante β of low/high legs (shrunk)")
    plt.tight_layout(); plt.savefig(OUT / "bab_betas.png", dpi=110)


if __name__ == "__main__":
    main()
