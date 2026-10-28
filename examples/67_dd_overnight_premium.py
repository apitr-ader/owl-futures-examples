"""67 深入：夜盤隔夜溢酬（60 H8）。做多 TXF 夜盤，1 分 K（2017-06~2026-10），1m bar 以起始時間標記。
進場時點：D1345＝前一日盤收盤（13:44 bar close）；N1500/N1505/N1515/N1530＝夜盤該 bar 開盤。
出場時點：X0500＝夜盤收盤（04:59 close）；X0845/X0900/X0930＝次日日盤該 bar 開盤。
成本：TX 50 元/邊（0.25 點）+ 2e-5 稅 + 每邊 1 點滑價；MTX 版：每點 50 元、50 元/邊（1 點）。
換月：夜盤與次日日盤永遠同一合約；只有 D1345 進場在結算日會用到舊合約收盤 → 改以新合約夜盤開盤進場（標記 roll），另提供「跳過換月夜」版本。
  組合策略（日盤做空/持有 24h）在換月日：舊合約日盤 13:30 結算出場、新合約夜盤開盤進場，兩腿成本都計。
外部資料（data/us/，不 commit）：S&P500 日收盤（Yahoo）、FOMC 決議日（federalreserve.gov）。
IS 2017-06~2021、OOS 2022+；WF 每年以前 3 年選進出場時點；隨機基準：同長度但隨機起點（日盤+夜盤任意時點）的做多窗口。
"""
import sys, re, json, importlib, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R
G63 = importlib.import_module("63_regime_filters")
US = Path("data/us"); TX_MARGIN, MTX_MARGIN = 701_000, 175_250   # 期交所 2026-10 公告原始保證金（維持 538,000 / 134,500）


def fomc_dates():
    mon = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])}
    ab = {k[:3]: v for k, v in mon.items()}; out = []
    h = (US / "fomc_cal.html").read_text()
    for blk in re.split(r'<h4><a id="\d+">', h)[1:]:
        y = int(blk[:4])
        for m, d in re.findall(r'fomc-meeting__month[^>]*><strong>([^<]*)</strong>.*?fomc-meeting__date[^>]*>([^<]*)<', blk, re.S):
            if "unscheduled" in d.lower() or "notation" in d.lower(): continue
            last = int(re.findall(r"\d+", d)[-1]); mm = m.split("/")[-1].strip()
            out.append(pd.Timestamp(y, mon.get(mm, ab.get(mm[:3], 0)), last))
    for y in range(2017, 2021):
        for m, d in re.findall(r'<h5[^>]*>(\w+(?:/\w+)?) ([\d\-]+) Meeting - %d</h5>' % y, (US / f"fomc_{y}.html").read_text()):
            mm = m.split("/")[-1]; out.append(pd.Timestamp(y, mon.get(mm, ab.get(mm[:3], 0)), int(d.split("-")[-1])))
    return pd.DatetimeIndex(sorted(set(out)))


def spx():
    j = json.loads((US / "gspc.json").read_text())["chart"]["result"][0]
    s = pd.Series(j["indicators"]["quote"][0]["close"], pd.to_datetime(j["timestamp"], unit="s").tz_localize("UTC").tz_convert("America/New_York").normalize().tz_localize(None))
    return s.dropna()


def build():
    d = R.load_1m(); hm = d.ts.dt.strftime("%H%M").values
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    days = pd.DatetimeIndex(sorted(d.tdate.unique())); rows = []
    gi = d.groupby(["tdate", "night"]).indices
    for i, t in enumerate(days):
        if i == 0 or (t, True) not in gi or (t, False) not in gi or (days[i - 1], False) not in gi: continue
        ni, di, pi = gi[(t, True)], gi[(t, False)], gi[(days[i - 1], False)]
        r = dict(tdate=t, start=d.ts.iat[ni[0]], cm_n=d.cm.iat[ni[0]], cm_p=d.cm.iat[pi[-1]], i_p=pi[-1], n0=ni[0], n1=ni[-1], d0=di[0], d1=di[-1])
        for k in ("1500", "1505", "1515", "1530"):
            j = ni[hm[ni] == k]; r["N" + k] = j[0] if len(j) else -1
        for k in ("0845", "0900", "0930"):
            j = di[hm[di] == k]; r["X" + k] = j[0] if len(j) else -1
        rows.append(r)
    return d, pd.DataFrame(rows).set_index("tdate"), o, h, l, c


def trade(S, o, h, l, c, en, ex, pv=200, comm_pts=0.25, skip_roll=False, direction=1):
    out = []
    for t, r in S.iterrows():
        roll = r.cm_p != r.cm_n
        if en == "D1345":
            if roll:
                if skip_roll: continue
                ie, ep = r.n0, o[r.n0]
            else: ie, ep = r.i_p, c[r.i_p]
        else:
            ie = r["N" + en[1:]]; ep = o[ie] if ie >= 0 else np.nan
            if roll and skip_roll: continue
        if ex == "X0500": ix, xp = r.n1, c[r.n1]
        else: ix = r[ex]; xp = o[ix] if ix >= 0 else np.nan; ix = ix - 1
        if not (ep == ep and xp == xp) or ie < 0: continue
        lo = l[ie + (en == "D1345" and not roll):ix + 1].min() if ix >= ie else ep; hi = h[ie:ix + 1].max() if ix >= ie else ep
        ep1, xp1 = ep + direction, xp - direction
        g = direction * (xp1 - ep1); cost = 2 * comm_pts + 2e-5 * (ep + xp)
        mae = (lo - ep1 if direction > 0 else ep1 - hi)
        out.append((t, g - cost, min(mae, 0) - 0, roll, ep))
    T = pd.DataFrame(out, columns=["tdate", "net", "mae", "roll", "ep"]).set_index("tdate"); T["twd"] = T.net * pv; T["mae_twd"] = T.mae * pv
    return T


def sstats(T, alld=None, pv=200):
    dp = T.twd if alld is None else T.twd.reindex(alld, fill_value=0); mae = T.mae_twd.reindex(dp.index).fillna(0)
    s = R.stats(dp, mae); return dict(Sharpe=s["Sharpe"], net=s["net_twd"], MDD_intra=s["MDD_intraday"], n=len(T), win=(T.net > 0).mean(), avg_pts=T.net.mean())


def main():
    d, S, o, h, l, c = build(); alld = S.index; IS = alld < "2022-01-01"
    print("nights", len(S), "roll nights", (S.cm_p != S.cm_n).sum())
    ENS = ["D1345", "N1500", "N1505", "N1515", "N1530"]; EXS = ["X0500", "X0845", "X0900", "X0930"]
    grid = {}; rows = []
    for en, ex in itertools.product(ENS, EXS):
        T = trade(S, o, h, l, c, en, ex); grid[(en, ex)] = T.twd.reindex(alld, fill_value=0)
        a, b = sstats(T[T.index < "2022-01-01"]), sstats(T[T.index >= "2022-01-01"])
        rows.append(dict(entry=en, exit=ex, IS=a["Sharpe"], OOS=b["Sharpe"], IS_avg=a["avg_pts"], OOS_avg=b["avg_pts"], OOS_MDD=b["MDD_intra"], n=a["n"] + b["n"]))
    V = pd.DataFrame(rows); print("\n== 進出場時點網格（TX 1 口）=="); print(V.round(3).to_string())
    print("Spearman IS/OOS %.3f" % V.IS.rank().corr(V.OOS.rank()))
    print("IS Sharpe pivot\n", V.pivot(index="entry", columns="exit", values="IS").round(2)); print("OOS Sharpe pivot\n", V.pivot(index="entry", columns="exit", values="OOS").round(2))
    wf = []; picks = []
    for Y in range(2020, 2027):
        sc = {k: R.sharpe(v[(v.index.year >= Y - 3) & (v.index.year < Y)]) for k, v in grid.items()}; b = max(sc, key=sc.get)
        wf.append(grid[b][grid[b].index.year == Y]); picks.append((Y, "+".join(b)))
    wf = pd.concat(wf); print("WF picks", picks, "WF Sharpe 2020+ %.3f, 2022+ %.3f" % (R.sharpe(wf), R.sharpe(wf[wf.index >= "2022-01-01"])))

    # 主規格：N1500→X0500（60 的版本）與 D1345→X0845（收→開）
    base = trade(S, o, h, l, c, "N1500", "X0500"); c2o = trade(S, o, h, l, c, "D1345", "X0845")
    skip = trade(S, o, h, l, c, "D1345", "X0845", skip_roll=True)
    print("\n== 換月處理（D1345→X0845）== include roll: Sharpe %.2f | skip roll: %.2f | roll nights n=%d avg %.1f pts" %
          (R.sharpe(c2o.twd.reindex(alld, fill_value=0)), R.sharpe(skip.twd.reindex(alld, fill_value=0)), c2o.roll.sum(), c2o[c2o.roll].net.mean()))
    # 隨機基準：同 14 小時長度、隨機起點的做多窗口
    rng = np.random.default_rng(1); L = 14 * 60; sims = []
    ts = d.ts.values; nb = len(d)
    for _ in range(200):
        st = rng.integers(0, nb - 2000, len(base)); en_ = np.searchsorted(ts, ts[st] + np.timedelta64(L, "m")); ok = (en_ < nb) & (d.cm.values[st] == d.cm.values[np.minimum(en_, nb - 1)])
        g = (o[en_[ok]] - 1) - (o[st[ok]] + 1) - 0.5 - 2e-5 * (o[st[ok]] + o[en_[ok]]); sims.append(g.mean() / g.std() * np.sqrt(252))
    print("random 14h long windows Sharpe-equivalent mean %.2f 5-95%% [%.2f, %.2f] vs base %.2f" % (np.mean(sims), np.percentile(sims, 5), np.percentile(sims, 95), base.net.mean() / base.net.std() * np.sqrt(252)))

    # 報酬分解：夜盤、夜→日盤開跳空、日盤、全日
    seg = pd.DataFrame(index=S.index)
    seg["night"] = (c[S.n1.values] - o[S.n0.values]); seg["gap"] = o[S.d0.values] - c[S.n1.values]; seg["day"] = c[S.d1.values] - o[S.d0.values]
    seg["eve"] = np.where(S.cm_p == S.cm_n, o[S.n0.values] - c[S.i_p.values], np.nan)   # 日盤收→夜盤開
    seg["full"] = seg[["eve", "night", "gap", "day"]].sum(axis=1, min_count=4)
    print("\n== 區段平均點數 / 年化 Sharpe（無成本）==")
    for k in seg: x = seg[k].dropna(); print(f"{k:6s} mean {x.mean():6.2f} pts  Sharpe {x.mean()/x.std()*np.sqrt(252):.2f}  share of full-mean {x.mean()/seg.full.mean():.2f}")
    sp = spx(); spr = np.log(sp).diff()
    us_d = S.start.dt.normalize()   # 夜盤 15:00 台北 ＝ 美東同一日 02/03:00 起，涵蓋美股該日常規盤
    seg["spx_same"] = spr.reindex(us_d.values).values; prevus = sp.index.searchsorted(us_d.values) - 1
    seg["spx_prev"] = spr.values[np.clip(prevus, 0, len(spr) - 1)]
    y = base.net.reindex(S.index)
    def ols(y, X, names):
        m = y.notna() & X.notna().all(axis=1); Y = y[m].values; A = np.c_[np.ones(m.sum()), X[m].values]; b, *_ = np.linalg.lstsq(A, Y, rcond=None)
        e = Y - A @ b; se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * e.var(ddof=A.shape[1])); r2 = 1 - e.var() / Y.var()
        return " ".join(f"{n}={v:.3f}(t={v/s:.1f})" for n, v, s in zip(["alpha_pts"] + names, b, se)) + f" R2={r2:.3f} n={m.sum()}"
    print("\n== Beta 分解（被解釋：N1500→X0500 淨點數/晚）==")
    print("on full-day TXF pts      :", ols(y, seg[["full"]], ["b_full"]))
    print("on day-session pts       :", ols(y, seg[["day"]], ["b_day"]))
    yr = y / S.N1500.map(lambda i: o[i] if i >= 0 else np.nan).reindex(S.index).values   # 報酬率
    print("on SPX same US day (ret) :", ols(yr, seg[["spx_same"]], ["b_spx"]))
    print("on SPX same + prev       :", ols(yr, seg[["spx_same", "spx_prev"]], ["b_spx", "b_spx_prev"]))
    # 以 B&H 日報酬（全日）模擬「同 beta 的長期做多」對照：alpha 年化
    full_r = (seg.full / o[S.d0.values]).reindex(S.index)
    m = yr.notna() & full_r.notna(); b = np.polyfit(full_r[m], yr[m], 1)
    print(f"beta to full-day ret {b[0]:.3f}; residual alpha annualized {(yr[m]-b[0]*full_r[m]).mean()*252:.2%} vs raw {yr[m].mean()*252:.2%}")

    # 濾網
    fom = fomc_dates(); us_dates = us_d
    f = {}
    f["skip_FOMC"] = ~us_dates.isin(fom).values
    f["skip_FOMC±1"] = ~(us_dates.isin(fom) | us_dates.isin(fom - pd.Timedelta(days=1)) | us_dates.isin(fom + pd.Timedelta(days=1))).values
    f["SPXprev_up"] = (seg.spx_prev > 0).values; f["SPXprev_down"] = (seg.spx_prev < 0).values
    f["not_Fri_night"] = (S.start.dt.weekday != 4).values; f["Fri_night_only"] = (S.start.dt.weekday == 4).values
    f["not_roll_night"] = (S.cm_p == S.cm_n).values
    Dd = pd.read_parquet(R.DATAD); Dd.index = Dd.date.dt.tz_localize(None).dt.normalize(); r = np.log(Dd.close).diff().fillna(0) * 100
    gv = pd.Series(np.nan, Dd.index)
    for Y in range(2012, 2027):
        w, a, bb = G63.garch_fit(r[r.index.year < Y].values[1:]); v = G63.garch_path(r.values, w, a, bb); mm = Dd.index.year == Y; gv[mm] = np.sqrt(v[mm])
    gp = G63.pct(gv); prevday = pd.Series(S.index, S.index).map(lambda t: Dd.index[Dd.index < t][-1])
    gpn = gp.reindex(prevday.values).values
    f["GARCH10-90"] = (gpn >= .1) & (gpn <= .9); f["GARCH<80"] = gpn < .8; f["GARCH>20"] = gpn > .2
    seg["prev_day"] = seg.day.shift(1); f["prevTXday_up"] = (seg.prev_day > 0).values; f["prevTXday_down"] = (seg.prev_day < 0).values
    yb = base.twd.reindex(alld, fill_value=0); fr = [dict(filter="none", keep=1, IS=R.sharpe(yb[IS]), OOS=R.sharpe(yb[~IS]), OOS_net=yb[~IS].sum(), OOS_avg_pts=base[base.index >= "2022"].net.mean())]
    for k, m_ in f.items():
        z = yb * m_; fr.append(dict(filter=k, keep=m_.mean(), IS=R.sharpe(z[IS]), OOS=R.sharpe(z[~IS]), OOS_net=z[~IS].sum(), OOS_avg_pts=base.net.reindex(alld)[m_ & ~IS].mean()))
    F = pd.DataFrame(fr); print("\n== 濾網（N1500→X0500）=="); print(F.round(3).to_string())
    print("FOMC dates parsed:", len(fom), fom.min().date(), "~", fom.max().date())

    # 組合：夜盤做多 + 日盤 空手/做空/做多(=24h 持有)
    def dayseg(direction):
        g = direction * ((c[S.d1.values] - direction) - (o[S.d0.values] + direction)) - 0.5 - 2e-5 * (o[S.d0.values] + c[S.d1.values]); return pd.Series(g * 200, S.index)
    nightT = base.twd.reindex(alld, fill_value=0)
    combos = {"night_long+day_flat": nightT, "night_long+day_short": nightT + dayseg(-1), "night_long+day_long(分段進出)": nightT + dayseg(1)}
    # 24h 持有（不分段，只在換月換倉）：日K 收→收 + 換月兩腿
    hold = (seg.full.fillna(seg.night + seg.gap + seg.day) * 200).copy(); rollm = (S.cm_p != S.cm_n).values
    hold[rollm] = ((seg.night + seg.gap + seg.day)[rollm] - 2 * (0.5 + 2 + 4e-5 * o[S.d0.values][rollm])) * 200 / 2 * 2
    combos["hold_24h(B&H,換月兩腿成本)"] = hold.fillna(0)
    combos["day_short_only"] = dayseg(-1)
    print("\n== 組合 ==")
    for k, v in combos.items(): print(f"{k:30s} IS {R.sharpe(v[IS]):.2f} OOS {R.sharpe(v[~IS]):.2f} OOS net/yr {v[~IS].sum()/(~IS).sum()*252/1e4:.0f}萬 MDD {R.stats(v)['MDD_close']/1e4:.0f}萬")

    # 年度表
    Y = pd.DataFrame({"night_TX_net萬": base.twd.groupby(base.index.year).sum() / 1e4, "n": base.groupby(base.index.year).size(),
                      "win": base.groupby(base.index.year).net.apply(lambda x: (x > 0).mean()), "avg_pts": base.groupby(base.index.year).net.mean(),
                      "Sharpe": nightT.groupby(nightT.index.year).apply(R.sharpe), "day_long_萬": dayseg(1).groupby(S.index.year).sum() / 1e4,
                      "c2o_萬": c2o.twd.groupby(c2o.index.year).sum() / 1e4})
    print("\n== 年度（N1500→X0500, TX 1 口）=="); print(Y.round(2).to_string())

    # 壓力／保證金
    print("\n== 壓力測試（TX 1 口，N1500→X0500）==")
    w = base.sort_values("net").head(10); w["pct"] = w.net / w.ep; print("最差 10 晚（淨點數、跌幅%）:\n", w[["net", "mae", "pct"]].round(4).to_string())
    worst_mae = base.mae.min(); s_all = sstats(base, alld); px = c[-1]
    worst_pct = (base.mae / base.ep).min(); stress_pts = worst_pct * px
    print(f"最差盤中 MAE {worst_mae:.0f} 點（{worst_pct:.2%}）→ 以目前指數 {px:.0f} 換算 {stress_pts:.0f} 點 = {stress_pts*200:,.0f} 元/口")
    eq = nightT.cumsum(); dd_pct = None
    print(f"全期盤中 MDD {s_all['MDD_intra']:,.0f} 元/口；以目前價位比例放大 ×{px/base.ep.mean():.2f} → {s_all['MDD_intra']*px/base.ep.mean():,.0f}")
    lose = (base.net < 0).astype(int).values; run = mx = 0
    for v in lose: run = run + 1 if v else 0; mx = max(mx, run)
    print("最長連敗晚數", mx)
    mdd_now = -s_all["MDD_intra"] * px / base.ep.mean()
    for mult, lab in ((1.0, "現行"), (1.3, "保證金+30%"), (1.6, "保證金+60%（危機時曾見）")):
        for kk, mg in (("TX", TX_MARGIN), ("MTX", MTX_MARGIN)):
            scale = 1 if kk == "TX" else .25
            need = mg * mult + max(-stress_pts * 200 * scale * 1.5, mdd_now * scale)
            print(f"{lab:22s} {kk}: 原始保證金 {mg*mult:,.0f} + 緩衝 max(1.5×最差夜, 歷史MDD) = 建議資金 {need:,.0f} 元/口")
    # MTX 成本
    mt = trade(S, o, h, l, c, "N1500", "X0500", pv=50, comm_pts=1.0); mtd = mt.twd.reindex(alld, fill_value=0)
    print(f"\n== MTX（50 元/點，手續費 50 元/邊=1 點）== avg net {mt.net.mean():.2f} pts vs TX {base.net.mean():.2f}; Sharpe IS {R.sharpe(mtd[IS]):.2f} OOS {R.sharpe(mtd[~IS]):.2f}; "
          f"4 口 MTX 年淨 {mtd.sum()*4/len(alld)*252/1e4:.1f}萬 vs 1 口 TX {nightT.sum()/len(alld)*252/1e4:.1f}萬; 成本佔毛利 TX {(2.5+2e-5*2*base.ep.mean())/(base.net.mean()+2.5+2e-5*2*base.ep.mean()):.0%} MTX {(4+2e-5*2*base.ep.mean())/(mt.net.mean()+4+2e-5*2*base.ep.mean()):.0%}")
    for s_ in (0.5, 1, 2, 3): 
        tt = base.net + 2 - 2 * s_; print(f"slippage {s_} pt/side: avg {tt.mean():.2f} pts Sharpe {tt.mean()/tt.std()*np.sqrt(252):.2f}")

    V.to_csv(R.OUT / "dd67_variants.csv", index=False); F.to_csv(R.OUT / "dd67_filters.csv", index=False); Y.to_csv(R.OUT / "dd67_yearly.csv")
    plt = R.style(); fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for k in ((("N1500", "X0500")), ("D1345", "X0845"), ("N1530", "X0500"), ("N1500", "X0900")): ax[0, 0].plot(grid[k].cumsum() / 1e4, label="→".join(k))
    ax[0, 0].plot(dayseg(1).cumsum() / 1e4, label="日盤做多", c=R.GRAY); ax[0, 0].axvline(pd.Timestamp("2022-01-01"), c=R.GRAY, ls="--"); ax[0, 0].set_title("進出場時點（TX 1 口，萬元）"); ax[0, 0].legend(fontsize=8)
    for k in ("night_long+day_flat", "night_long+day_short", "hold_24h(B&H,換月兩腿成本)"): ax[0, 1].plot(combos[k].cumsum() / 1e4, label=k)
    ax[0, 1].set_title("組合（萬元）"); ax[0, 1].legend(fontsize=8)
    eqv = nightT.cumsum(); ax[1, 0].fill_between(eqv.index, (eqv - eqv.cummax()) / 1e4, color=R.COLORS[2]); ax[1, 0].set_title("回撤（萬元/口）")
    ax[1, 1].hist(base.net, bins=120, color=R.GOLD); ax[1, 1].set_title("每晚淨點數分布"); ax[1, 1].set_yscale("log")
    for a in ax.flat: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(R.OUT / "dd_67_overnight.png", dpi=100); print("chart results/dd_67_overnight.png")


if __name__ == "__main__":
    main()
