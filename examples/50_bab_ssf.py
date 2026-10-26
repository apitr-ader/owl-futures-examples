"""50 BAB 限縮在有股票期貨（SSF）的標的，兩腿都用股票期貨交易（無證交稅 0.3%、無借券費、無放空限制）。
資料：data/stock_futures_daily（tools/sfdaily.py，期交所逐筆重建；含下市契約；資料止於 2026-08-17）。不 commit 任何期交所衍生資料。
- 標的池（point-in-time）：products_map underlying_type=='S'、標準契約（2000 股；小型 100 股另作離散口數可行性），
  以各月底前 20 日實際有成交（front-month volume>0 天數 ≥ 10）認定「有掛牌且有交易」，並以 SSF 20 日均量（口）與現股 20 日均成交值過濾（敏感度）。
  同一標的多個代碼（舊/新）取當日量大者。期交所資料無未平倉量 → 只能用成交量。
- 期貨報酬：近月連續（到期隔日換月）；r_t = 同一契約 close_t / close_{t-1}；換月日用新契約前一日收盤（contracts/ 表）；
  無同契約前收或 |r|>15%（如股利調整轉為 A 型契約）→ 以現股報酬遞補（計數報告）。開盤報酬同理 open_t / close_{t-1}。
  另跑「現股報酬代理」（同一權重、期貨成本）→ 差額 = 基差/股利效果。
- 成本（期貨）：手續費 50 元/口/邊、期交稅 2e-5/邊、滑價 1 tick/邊（現股 tick 表）、每月換月雙邊（持倉 × 2 × 單邊成本）。零成本情境全部歸零。
- 保證金：整個 NAV 為現金賺 rf（期貨名目不占用現金）；名目槓桿 1/β_L、1/β_H 自然由期貨達成。
- 離散口數：資本 1,000 萬 / 3,000 萬，目標名目 / (2000×價) 四捨五入；另「含小型」= 有小型契約者以 100 股為單位。
- β：FP（TAIEX TR）主版本；Dimson(-1,0,+1)+Vasicek；FP + 事後實際腿 β 調整（48 targets rh）。
- 變體：rank L/S、三分位 EW、產業中性（SSF 池內）、月 vs 季調整、(b2) 低 β SSF 多 − TXF×β_L。
期間：IS 2011-2019、OOS 2020-01 ~ 2026-08-17（SSF 資料終點）；現股代理另報到 2026-10-22。
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd

_s = importlib.util.spec_from_file_location("m48", Path(__file__).with_name("48_bab_v2.py"))
m48 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m48)
_s = importlib.util.spec_from_file_location("m49", Path(__file__).with_name("49_low_ivol.py"))
m49 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m49)
m = m48.m; ROOT, OUT = m.ROOT, m.OUT
IS0, IS1, SEND = m48.IS0, m48.IS1, pd.Timestamp("2026-08-18")
SFD = ROOT / "data/stock_futures_daily"
FC, FT = 50.0, 2e-5


def load_ssf(D):
    """futures panels aligned to D: Af (close index), AOf (open index), Pf (raw close), POf (raw open), vol, roll, fallback count; mini flag."""
    days, cols = D["days"], {c: i for i, c in enumerate(D["cols"])}; nd, nS = len(days), len(cols)
    pm = pd.read_csv(SFD / "products_map.csv", dtype={"underlying_code": str})
    pm = pm[(pm.underlying_type == "S") & pm.underlying_code.isin(cols)]
    rc, ro, Pc, Po, V, RL = (np.full((nd, nS), np.nan) for _ in range(6)); MINI = np.zeros((nd, nS), bool)
    dix = pd.Series(np.arange(nd), days)
    for x in pm.itertuples():
        j = cols[x.underlying_code]; f = SFD / "daily" / f"{x.symbol}_raw.parquet"; g = SFD / "contracts" / f"{x.symbol}.parquet"
        if not (f.exists() and g.exists()):
            continue
        d = pd.read_parquet(f); d["date"] = pd.to_datetime(d.time).dt.tz_localize(None).dt.normalize()
        if x.mini:
            ii = dix.reindex(d.date).dropna().astype(int).values; MINI[ii, j] = True; continue
        c = pd.read_parquet(g).sort_values(["contract", "trade_date"]); c["pc"] = c.groupby("contract").close.shift()
        c["pd_"] = c.groupby("contract").trade_date.shift()
        d = d.merge(c[["trade_date", "contract", "pc", "pd_"]], left_on=["date", "contract_month"], right_on=["trade_date", "contract"], how="left")
        d = d[d.date.isin(days)]; ii = dix[d.date].values
        better = ~(V[ii, j] >= d.volume.values)             # several symbols per stock: keep higher volume
        ii, d = ii[better], d[better]
        # previous trading day of contract must be previous panel day, else fallback
        prev = np.r_[pd.NaT, days[:-1]][ii]; ok = (d.pd_.values == prev)
        rc[ii, j] = np.where(ok, d.close.values / d.pc.values - 1, np.nan); ro[ii, j] = np.where(ok, d.open.values / d.pc.values - 1, np.nan)
        Pc[ii, j], Po[ii, j], V[ii, j], RL[ii, j] = d.close.values, d.open.values, d.volume.values, d.is_roll.values.astype(float)
    # TAIFEX adjusts SSF contracts for cash dividends (positions keep the dividend value, but the same-code
    # contract price drops on ex-date). Add back the spot dividend component (TR return - raw price return).
    Ps = D["P"]; rawr = np.r_[np.full((1, nS), np.nan), Ps[1:] / Ps[:-1] - 1]
    dv = np.clip(np.nan_to_num(np.nan_to_num(D["R"]) - rawr), 0, 0.2); dv[dv < 2e-3] = 0
    rc = rc + dv; ro = ro + dv
    bad = ~(np.abs(rc) < 0.15) | ~(np.abs(ro) < 0.15)
    has = np.isfinite(Pc) & (np.nan_to_num(V) > 0)
    R0 = np.nan_to_num(D["R"])                               # spot close-close
    fb = has & bad
    rcf = np.where(has & ~bad, rc, R0)                       # non-traded days: spot return keeps index moving (only matters while held)
    # spot open return for fallback
    Aspot = D["A"]; sro = np.nan_to_num(np.where(D["traded"], D["AO"] / np.r_[Aspot[:1], Aspot[:-1]] - 1, 0))
    rof = np.where(has & ~bad, ro, sro)
    Af = np.cumprod(1 + rcf, 0); AOf = np.r_[Af[:1], Af[:-1]] * (1 + rof)
    return dict(A=Af, AO=AOf, P=np.where(has, Pc, D["P"]), PO=np.where(has, Po, D["PO"]), vol=np.nan_to_num(V), roll=np.nan_to_num(RL) > 0,
                has=has, fallback=fb, MINI=MINI)


def sim_f(D, F, X, tg, hd=None, costs=True, spot=False):
    """all positions are futures notionals; NAV is cash earning rf. spot=True: spot prices/returns, futures costs."""
    days = D["days"]
    A, AO, P, PO = (D[k] for k in ("A", "AO", "P", "PO")) if spot else (F[k] for k in ("A", "AO", "P", "PO"))
    traded = D["traded"] if spot else F["has"]
    lu, ld, last, roll = D["lu"], D["ld"], D["last"], F["roll"]
    nd, nS = A.shape; Xr = X.reindex(days); xo, xc, xg, xi, xroll = (Xr[k].values for k in ("o", "c", "gap", "intr", "roll"))
    w = np.zeros(nS); h = 0.0; pend = np.full(nS, np.nan); hp = np.nan; hd = hd or {}
    k1 = 1.0 if costs else 0.0
    cols = ("ret", "long", "short", "hedge", "cash", "cost", "borrow", "turn")
    O = {k: np.zeros(nd) for k in cols}
    for t in range(1, nd):
        ro = np.nan_to_num(np.where(traded[t] | ~np.isnan(A[t]), AO[t] / A[t - 1] - 1, 0.0))
        hg = 0.0 if np.isnan(xg[t]) else xg[t]
        pl, ps = (w * ro)[w > 0].sum(), (w * ro)[w < 0].sum(); ph = h * hg; c = 0.0; trn = 0.0
        g1 = 1 + pl + ps + ph; w = w * (1 + ro) / g1; h = h * (1 + hg) / g1
        if costs:
            if xroll[t] == True and h != 0 and np.isfinite(xo[t]):
                c += abs(h) * 2 * (2e-5 + (50 + 200) / (200 * xo[t]))
            rr = np.where(roll[t] & (w != 0))[0]
            if len(rr):
                px = P[t - 1, rr]; px = np.where(np.isfinite(px) & (px > 0), px, 1.0)
                c += (np.abs(w[rr]) * 2 * (FT + FC / (2000 * px) + m.tick(px) / px)).sum()
        ip = np.where(~np.isnan(pend))[0]
        if len(ip):
            d = pend[ip] - w[ip]; buy = d > 0; dl = t > last[ip]
            okx = (traded[t, ip] & ~(buy & lu[t, ip]) & ~(~buy & ld[t, ip])) | dl
            ii = ip[okx]
            if len(ii):
                px = np.where(traded[t, ii], PO[t, ii], P[t, ii]); px = np.where(np.isfinite(px) & (px > 0), px, 1.0)
                dd = pend[ii] - w[ii]; c += k1 * (np.abs(dd) * (FT + FC / (2000 * px) + m.tick(px) / px)).sum(); trn += np.abs(dd).sum()
                w[ii] = pend[ii]; pend[ii] = np.nan
        if np.isfinite(hp) and np.isfinite(xo[t]):
            c += k1 * abs(hp - h) * (2e-5 + (50 + 200) / (200 * xo[t])); h = hp; hp = np.nan
        rc = np.nan_to_num(A[t] / AO[t] - 1)
        hi_ = xi[t] * (xc[t - 1] if np.isfinite(xc[t - 1]) else xo[t]) / xo[t] if (np.isfinite(xi[t]) and np.isfinite(xo[t])) else 0.0
        hi_ = 0.0 if not np.isfinite(hi_) else hi_
        cr = m.RF / 252
        ql, qs = (w * rc)[w > 0].sum(), (w * rc)[w < 0].sum(); qh = h * hi_
        g2 = 1 + ql + qs + qh
        r = (1 + pl + ps + ph) * (1 - c) * g2 - 1 + cr
        w = w * (1 + rc) / g2; h = h * (1 + hi_) / g2
        for k, v in zip(cols, (r, pl + ql, ps + qs, ph + qh, cr, -c, 0.0, trn)):
            O[k][t] = v
        if t in tg:
            tw, th = tg[t]; ch = (np.abs(tw - w) > 1e-12) | ~np.isnan(pend); pend = np.where(ch, tw, np.nan); hp = th
    return pd.DataFrame(O, days)


def discretize(w, px, cap, mini=None):
    lot = np.where(mini, 100, 2000) if mini is not None else 2000
    n = np.round(w * cap / (lot * px)); return np.nan_to_num(n * lot * px / cap)


def main():
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    D = m.ss.prepare(m.sd); days = D["days"]; X = m.m37.load()
    rT = m48.idx("taiex_daily.parquet", "tr_close"); mv = np.nan_to_num(rT.reindex(days).values)
    BT = m.betas(D, rT); BD = m48.beta_methods(D, rT)["Dimson(-1,0,+1) 1y Vasicek .6"]()
    F = load_ssf(D)
    print("futures-days with SSF trade:", int(F["has"].sum()), "fallback (no same-contract prev close / |r|>15%):", int(F["fallback"].sum()), flush=True)
    im = pd.read_csv(ROOT / "data/stocks/_industry_map.csv", dtype={"code": str}).set_index("code").reindex(D["cols"])
    GRP = im.ind.fillna("NA").values
    reb = [t for t in np.where(np.r_[days[1:].month != days[:-1].month, False] & (days >= IS0 - pd.Timedelta(days=40)) & (days < SEND))[0] if t < len(days) - 1]
    rebq = [t for t in reb if days[t].month in (3, 6, 9, 12)]
    H = F["has"].astype(float); V = F["vol"]
    act = pd.DataFrame(H).rolling(20, min_periods=1).sum().values >= 10
    v20 = pd.DataFrame(V).rolling(20, min_periods=1).mean().values

    def U(t, fv=10, sl=m.LIQ):
        return act[t] & (v20[t] >= fv) & (D["t20"][t] >= sl)

    usz = pd.DataFrame({"date": [days[t] for t in reb], "listed_active": [int(act[t].sum()) for t in reb],
                        "main_univ(vol>=10, spot>=50M)": [int((U(t) & np.isfinite(BT[t]) & (D["P"][t] >= m.PFLOOR) & D["traded"][t]).sum()) for t in reb]}).set_index("date")
    usz.to_csv(OUT / "bab_ssf_universe.csv"); print(usz.resample("YE").mean().round(0).to_string())

    V_, INFO = {}, {}

    def run(name, B=BT, fv=10, sl=m.LIQ, rbs=None, cap=None, mini=False, costs=True, spot=False, **kw):
        tg = {}
        for t in (rbs or reb):
            x = m48.targets(D, B, t, mask=U(t, fv, sl), min_n=kw.pop("min_n", 12) if False else 12, **kw)
            if x is None:
                continue
            w, h = x[:2]
            if cap is not None:
                px = F["P"][t]; w = discretize(w, np.where(np.isfinite(px) & (px > 0), px, 1e9), cap, F["MINI"][t] if mini else None)
            tg[t] = (w, h)
        o = sim_f(D, F, X, tg, costs=costs, spot=spot); V_[name] = o
        INFO[name] = pd.Series([int((tg[t][0] != 0).sum()) for t in tg])

    base = [("rank L/S", {}), ("tercile EW L/S", dict(kind="q3")), ("sector-neutral L/S", dict(grp=GRP)),
            ("rank L/S quarterly", dict(rbs=rebq)), ("(b2) low-β SSF − TXF×βL", dict(mode="lhedge_unlev")),
            ("rank L/S Dimson+Vasicek", dict(B=BD)), ("rank L/S FP + ex-post leg-β", dict(rh=mv))]
    for nm, kw in base:
        run(f"[F] {nm}", **kw); run(f"[F0] {nm}", costs=False, **kw)
    for nm, kw in base[:1] + base[4:5]:
        run(f"[S] {nm}", spot=True, **kw); run(f"[S0] {nm}", spot=True, costs=False, **kw)
    for fv in (0, 50, 200):
        run(f"[F] rank L/S ssfvol>={fv}", fv=fv)
    run("[F] rank L/S spot liq0", sl=0); run("[F] rank L/S spot liq100M", sl=1e8)
    for cap in (1e7, 3e7):
        run(f"[F] rank L/S {cap/1e6:.0f}M discrete 2000sh", cap=cap); run(f"[F] rank L/S {cap/1e6:.0f}M discrete +mini", cap=cap, mini=True)
    V_["TAIEX TR B&H"] = pd.DataFrame({"ret": rT.reindex(days).fillna(0)}).assign(long=lambda d: d.ret, short=0., hedge=0., cash=0., cost=0., borrow=0., turn=0.)
    rows = []
    for k, o in V_.items():
        for p, lo, hi in (("IS", IS0, IS1), ("OOS", IS1, SEND)):
            w_ = m.win(o, lo, hi); st = m49.stats(w_, rT); mx = (rT.reindex(w_.index).fillna(0) - m.RF / 252).mean() * 252
            rows.append(dict(name=k, period=p, CAGR=st["CAGR"], vol=st["vol"], sharpe=st["sharpe"], MDD=st["MDD"], beta=st["beta_reg"], alpha=st["alpha"], alpha_t=st["alpha_t"],
                             long_g=st["long_yr"], short_g=st["short_yr"], hedge=st["hedge_yr"], beta_mkt=st["beta_reg"] * mx, cost=-st["cost_yr"], turn=st["turn_yr"],
                             n_pos=INFO[k].mean() if k in INFO else np.nan))
    S = pd.DataFrame(rows); S.to_csv(OUT / "bab_ssf_summary.csv", index=False)
    for p in ("IS", "OOS"):
        print(f"\n== {p}"); print(S[S.period == p].drop(columns="period").round(3).to_string(index=False))
    o = V_["[S] rank L/S"]; w_ = m.win(o, IS1, None); print("\n[S] rank L/S spot proxy OOS to 2026-10-22:", {k: round(v, 3) for k, v in m49.stats(w_, rT).items() if k in ("CAGR", "sharpe", "MDD")})
    ky = ["[F] rank L/S", "[F0] rank L/S", "[S] rank L/S", "[S0] rank L/S", "[F] (b2) low-β SSF − TXF×βL", "[F] rank L/S FP + ex-post leg-β", "[F0] rank L/S FP + ex-post leg-β", "TAIEX TR B&H"]
    Y = pd.DataFrame({k: (1 + m.win(V_[k], IS0, SEND).ret).groupby(m.win(V_[k], IS0, SEND).index.year).prod() - 1 for k in V_}); Y.to_csv(OUT / "bab_ssf_yearly.csv")
    print("\n== yearly"); print(Y[ky].round(3).to_string())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for k in ky + ["[F] sector-neutral L/S", "[F0] sector-neutral L/S"]:
        e = (1 + m.win(V_[k], IS0, SEND).ret).cumprod(); st = dict(c="k", lw=2) if "TAIEX" in k else dict(lw=1.2, ls="--" if "0]" in k else "-")
        ax[0].plot(e.index, e, label=k, **st); ax[1].plot(e.index, e / e.cummax() - 1, **st)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7); ax[1].set_ylabel("drawdown"); ax[0].set_title("50 BAB on single-stock futures (F=SSF prices, S=spot proxy, 0=zero cost)")
    ax[2].plot(usz.index, usz.iloc[:, 0], label="SSF listed & active"); ax[2].plot(usz.index, usz.iloc[:, 1], label="main universe"); ax[2].legend(); ax[2].set_ylabel("# stocks")
    for a in ax:
        a.axvline(IS1, c="gray", lw=.8)
    plt.tight_layout(); plt.savefig(OUT / "bab_ssf.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
