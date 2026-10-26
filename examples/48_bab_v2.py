"""48 BAB v2：市場代理改用 TAIEX 報酬指數（data/index/taiex_daily.parquet tr_close，tools/fetch_taiex.py 抓取），
並加入電子業 BAB 與產業中性 BAB。β 估計、權重、成交、成本與 47 相同（引用 47 的 betas/sim/stats）。
- 全市場：(a)-(d) 全部變體，β 對 TAIEX TR；TXF 避險以 TAIEX β 計（TXF ≈ TAIEX 價格指數期貨）。首次 β 約 2011-01（個股資料自 2008）。
- 電子業：產業 = data/stocks/_industry_map.csv（現行上市櫃產業別；下市股以 FinMind TaiwanStockInfo 補，79 檔下市股無產業 → 排除）。
  電子 = 半導體/電腦及週邊/光電/通信網路/電子零組件/電子通路/資訊服務/其他電子（含 FinMind「電子工業」）。上市櫃皆含。
  β 對電子類指數（價格指數，無 TR）；穩健性：β 對 TAIEX TR。避險：放空 TXF × 低 β 腿的 TAIEX β（電子類指數不可交易）。
- 產業中性：β 在各產業內百分位排名後再做 rank 權重（多空在每個產業內大致平衡）。
期間：IS 2011-2019（另報 IS 2012-2019 以便與 47 比較），OOS 2020-01 ~ 2026-10-22。
"""
import importlib.util
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


def targets(D, B, t, mask=None, kind="rank", mode="ls", grp=None, BH=None, SF=None, liq=m.LIQ, pf=m.PFLOOR):
    b = B[t]; ok = D["traded"][t] & np.isfinite(b) & (D["t20"][t] >= liq) & (D["P"][t] >= pf)
    if mask is not None:
        ok &= mask
    i = np.where(ok)[0]
    if len(i) < 20:
        return None
    bb = b[i]
    if grp is not None:   # within-industry percentile ranks
        z = pd.Series(bb).groupby(grp[i]).rank(pct=True).values
    else:
        z = pd.Series(bb).rank().values
    zb = z.mean()
    if kind == "rank":
        k = 2 / np.abs(z - zb).sum(); wL = k * np.clip(zb - z, 0, None); wH = k * np.clip(z - zb, 0, None)
    else:
        q = 0.5 if kind == "ew" else 0.2
        lo, hi = bb <= np.quantile(bb, q), bb >= np.quantile(bb, 1 - q); wL, wH = lo / lo.sum(), hi / hi.sum()
    if SF is not None:
        s = SF[t, i] & (z > zb)
        wH = np.nan_to_num(np.where(s, pd.Series(np.where(s, bb, np.nan)).rank().values, 0)); wH = wH / wH.sum() if wH.sum() > 0 else wH
    bL, bH = wL @ bb, wH @ bb
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


def main():
    pd.set_option("display.width", 260)
    D = m.ss.prepare(m.sd); days = D["days"]; X = m.m37.load()
    rT, rE, r50 = idx("taiex_daily.parquet", "tr_close"), idx("sector_electronics_daily.parquet", "close"), m.bench()
    BT, BE, B50 = m.betas(D, rT), m.betas(D, rE), m.betas(D, r50)
    SF = m.sf_mask(D)
    im = pd.read_csv(ROOT / "data/stocks/_industry_map.csv", dtype={"code": str}).set_index("code").reindex(D["cols"])
    EL = im.elec.fillna(False).astype(bool).values; GRP = im.ind.fillna("NA").values; KNOWN = im.ind.notna().values
    reb = [t for t in np.where(np.r_[days[1:].month != days[:-1].month, False] & (days >= IS0 - pd.Timedelta(days=40)))[0] if t < len(days) - 1]

    def run(B, name, sim_kw=None, **kw):
        tg, info = {}, []
        for t in reb:
            x = targets(D, B, t, **kw)
            if x is not None:
                tg[t] = x[:2]; info.append((days[t], x[2], x[3], int((x[0] > 0).sum()), int((x[0] < 0).sum())))
        o = m.sim(D, X, tg, **(sim_kw or {})); o.attrs["info"] = pd.DataFrame(info, columns=["date", "bL", "bH", "nL", "nS"]).set_index("date")
        V[name] = o

    V = {}
    for tag, B in (("T", BT), ("50", B50)):
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
    for tag, B in (("E", BE), ("E|T", BT)):
        run(B, f"[{tag}] ELEC BAB rank", mask=EL)
        run(B, f"[{tag}] ELEC BAB EW quintile", mask=EL, kind="q5")
        run(B, f"[{tag}] ELEC (b2) low-β unlev − TXF×βL(TAIEX)", mask=EL, mode="lhedge_unlev", BH=BT)
        run(B, f"[{tag}] ELEC (d) long low-β unlev", mask=EL, mode="long")
        run(B, f"[{tag}] ELEC (d') long high-β unlev", mask=EL, mode="high")
    run(BE, "[E] ELEC EW all (liq-filtered)", mask=EL, mode="all")
    run(BE, "[E] ELEC sub-industry-neutral BAB", mask=EL, grp=GRP)
    for k, r in (("TAIEX TR B&H", rT), ("ELEC index B&H (price)", rE), ("0050 B&H", r50)):
        V[k] = pd.DataFrame({"ret": r.reindex(days).fillna(0)}).assign(long=lambda d: d.ret, short=0., hedge=0., cash=0., cost=0., borrow=0., turn=0.)
    rows = []
    for k, o in V.items():
        for p, lo, hi in PER:
            st = m.stats(m.win(o, lo, hi), rT)
            if k.startswith("[E") or "電子" in k:
                st["beta_elec"] = m.stats(m.win(o, lo, hi), rE)["beta3d"]
            rows.append(dict(name=k, period=p, **st))
    S = pd.DataFrame(rows); S.to_csv(OUT / "bab2_summary.csv", index=False)
    cols = ["name", "CAGR", "vol", "sharpe", "MDD", "beta3d", "beta_elec", "long_yr", "short_yr", "hedge_yr", "cost_yr", "turn_yr"]
    for p in ("IS", "IS12", "OOS"):
        print(f"\n== {p}"); print(S[S.period == p][cols].round(3).to_string(index=False))
    Y = pd.DataFrame({k: (1 + m.win(o, IS0, None).ret).groupby(m.win(o, IS0, None).index.year).prod() - 1 for k, o in V.items()})
    Y.to_csv(OUT / "bab2_yearly.csv")
    print(Y[[c for c in Y if "liq" not in c and "borrow" not in c]].round(3).T.to_string())
    for k in ("[T] (a) BAB rank (paper)", "[E] ELEC BAB rank"):
        i = V[k].attrs["info"]; print(k, "first reb", i.index[0].date(), i[["bL", "bH", "nL"]].mean().round(2).to_dict())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for fn, keys in (("bab2_market", ["[T] (a) BAB rank (paper)", "[50] (a) BAB rank (paper)", "[T] (b2) low-β unlev − TXF×βL", "[T] (c) long spot / short SF",
                                       "[T] (d) long low-β unlev", "[T] sector-neutral BAB rank", "TAIEX TR B&H", "0050 B&H"]),
                     ("bab2_elec", ["[E] ELEC BAB rank", "[E|T] ELEC BAB rank", "[E] ELEC sub-industry-neutral BAB", "[E] ELEC (b2) low-β unlev − TXF×βL(TAIEX)",
                                    "[E] ELEC (d) long low-β unlev", "[E] ELEC (d') long high-β unlev", "[E] ELEC EW all (liq-filtered)", "ELEC index B&H (price)", "TAIEX TR B&H"])):
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
