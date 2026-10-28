"""71 Reddit #5/#6：TXO 逐履約價未平倉量（OI）推估 dealer gamma（GEX 代理）。

資料：TAIFEX 選擇權每日交易行情下載（optDataDown，含「未沖銷契約數」與「結算價」），存 data/txo_oi/YYYYMM.csv（repo 外）。
  既有 RPT 逐筆檔只有成交、沒有 OI，所以另行下載。用「一般」時段列（OI 只在此列）。
GEX_t = Σ_series,K  sgn × OI × Γ(Black-76, 結算價反推 IV, F=TXF 收盤, r=0) × F² × 1% × 50  （元 / 指數 1% 變動）
符號假設（無法得知真實 dealer 部位，兩種都測）：
  A「dealer 買 call、賣 put」（SpotGamma 慣例，客戶賣 call/買 put）：call +、put −
  B「dealer 全部賣方」（客戶淨買方）：全部 −，即 −總 gamma 量
測試（t 日收盤資訊 → t+1）：
  1. 次日 TXF 實現波動（日內 range/open、|close-close|）：GEX 百分位 vs 次日波動 Spearman；控制今日波動後的偏相關；負/正 gamma 分組。
  2. 趨勢 vs 均值回歸：次日 open→close 報酬 與 今日 close-close 報酬 之符號關係，分 regime。
     策略：GEX 低於門檻（負 gamma）→ 跟隨今日方向；高於門檻（正 gamma）→ 反向；t+1 開盤進、收盤出 1 口 TXF。
     成本 50 元/邊 + 2e-5 稅 + 1 點滑價；MDD 盤中最差（以當日不利極值）。
  3. 結算日 pinning：結算日前一日收盤 該檔到期序列 C+P OI 最大履約價 K*；結算日 TXF 收盤 vs 開盤距 K* 是否縮小，
     對照安慰劑履約價 K*±2 檔、K*±5 檔；策略：結算日開盤朝 K* 方向下 1 口 TXF、收盤出（|open−K*|>門檻）。
IS 2017–2021、OOS 2022+；隨機基準、逐年 WF（前 3 年）。
"""
import glob, math, itertools
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from _rqo import *

OI_DIR = ROOT / "data/txo_oi"; END = pd.Timestamp("2026-10-21")
rng = np.random.default_rng(5)


def load_oi():
    fs = sorted(glob.glob(str(OI_DIR / "20*.csv"))); L = []
    for f in fs:
        d = pd.read_csv(f, index_col=False, dtype=str)
        d.columns = [c.strip() for c in d.columns]
        d = d[d["交易時段"].str.strip() == "一般"]
        L.append(pd.DataFrame({"date": pd.to_datetime(d["交易日期"]), "expiry": d["到期月份(週別)"].str.strip(),
                               "K": pd.to_numeric(d["履約價"], errors="coerce"), "cp": np.where(d["買賣權"].str.strip() == "買權", "C", "P"),
                               "settle": pd.to_numeric(d["結算價"], errors="coerce"), "oi": pd.to_numeric(d["未沖銷契約數"], errors="coerce")}))
    o = pd.concat(L); o = o[(o.oi > 0) & o.settle.notna() & (o.settle > 0)]
    return o[o.expiry.str.match(r"^\d{6}([WF]\d)?$")]


def b76_iv_gamma(p, F, K, T, cp):
    lo, hi = np.full(len(p), 1e-3), np.full(len(p), 3.0); sq = np.sqrt(T)
    for _ in range(50):
        m = (lo + hi) / 2; d1 = (np.log(F / K) + .5 * m * m * T) / (m * sq); d2 = d1 - m * sq
        v = np.where(cp == "C", F * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - F * norm.cdf(-d1))
        lo = np.where(v < p, m, lo); hi = np.where(v >= p, m, hi)
    s = (lo + hi) / 2; d1 = (np.log(F / K) + .5 * s * s * T) / (s * sq)
    return s, norm.pdf(d1) / (F * s * sq)


def gex(o, F):
    from tools.optdaily import expiries
    E = expiries().set_index("expiry").expiry_date
    o = o[o.expiry.isin(E.index)].copy(); o["xd"] = o.expiry.map(E)
    o["F"] = o.date.map(F.close); o = o.dropna(subset=["F"])
    o["T"] = ((o.xd - o.date).dt.days + .5) / 365
    o = o[(o["T"] > 0)]
    intr = np.where(o.cp == "C", np.maximum(o.F - o.K, 0), np.maximum(o.K - o.F, 0))
    o["p"] = np.maximum(o.settle.values, intr + 0.05)
    iv, g = b76_iv_gamma(o.p.values, o.F.values, o.K.values, o["T"].values, o.cp.values)
    o["g"] = np.where(iv > 2.9, 0, g) * o.oi * o.F ** 2 * .01 * MULT
    gc = o[o.cp == "C"].groupby("date").g.sum(); gp = o[o.cp == "P"].groupby("date").g.sum()
    G = pd.DataFrame({"A": gc - gp, "B": -(gc + gp)}).fillna(0)
    return G, o


def pct(s, n=252):
    return s.rolling(n, min_periods=120).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)


def tx_stats(pnl, worst, a, b):
    m = (pnl.index >= a) & (pnl.index < b); p = pnl[m]; w = worst[m]
    eq = p.cumsum(); prev = eq.shift(fill_value=0)
    mddw = float((prev + w - prev.cummax().clip(lower=0)).min()) if len(p) else 0
    return dict(n=int((p != 0).sum()), net=p.sum(), Sharpe=sharpe(p), MDDw=mddw, win=(p[p != 0] > 0).mean())


def main():
    F = load_fut("2016-01-01"); F = F[F.index <= END]
    o = load_oi(); print("OI rows", len(o), o.date.min().date(), o.date.max().date())
    G, oo = gex(o, F); G = G[G.index.isin(F.index)]
    nxt = F.shift(-1); r_cc = np.log(F.close / F.close.shift()); r_cc[F.roll] = np.nan
    D = pd.DataFrame({"A": G.A, "B": G.B}).join(F)
    D["rng1"] = ((nxt.high - nxt.low) / nxt.open).reindex(D.index); D["rng0"] = (F.high - F.low) / F.open
    D["abs1"] = r_cc.shift(-1).abs().reindex(D.index); D["vol20"] = r_cc.rolling(20).std().reindex(D.index)
    D["r0"] = r_cc.reindex(D.index); D["oc1"] = (nxt.close - nxt.open).reindex(D.index)
    D["roll1"] = F.roll.shift(-1).reindex(D.index).fillna(False).astype(bool)
    for s in "AB":
        D[s + "n"] = D[s] / (D.close ** 2 * .01 * MULT * 1e3)   # 正規化：以指數平方縮放
        D[s + "p"] = pct(D[s + "n"])
    D = D.dropna(subset=["rng1", "Ap"])
    print("\n== 1. GEX vs 次日波動 ==")
    out1 = []
    for s in "AB":
        for nm, a, b in (("IS", "2017", "2022"), ("OOS", "2022", "2027")):
            x = D[(D.index >= a) & (D.index < b)]
            rho = spearmanr(x[s + "p"], x.rng1)[0]
            X = np.c_[np.ones(len(x)), x.rng0, x.vol20.fillna(x.vol20.mean())]; res = x.rng1 - X @ np.linalg.lstsq(X, x.rng1, rcond=None)[0]  # 控制今日 range 與 20 日波動
            rho_p = spearmanr(x[s + "p"], res)[0]
            neg = x[x[s] < 0].rng1.mean() if (x[s] < 0).any() else np.nan; pos = x[x[s] >= 0].rng1.mean()
            lo = x[x[s + "p"] < .2].rng1.mean(); hi = x[x[s + "p"] > .8].rng1.mean()
            out1.append(dict(sign=s, per=nm, n=len(x), rho_range=rho, rho_resid=rho_p, frac_neg=(x[s] < 0).mean(), rng_neg=neg, rng_pos=pos, rng_p20=lo, rng_p80=hi))
    T1 = pd.DataFrame(out1); print(T1.round(4).to_string(index=False))
    print("\n== 2. 趨勢/回歸 ==")
    # 次日 open→close 與今日方向
    D["dir"] = np.sign(D.r0)
    rows = []; keep = {}
    cost = 2 * (FCOMM / FMULT + FSLIP) + 0  # 點
    for s, thr in itertools.product("AB", (.2, .3, .5)):
        lo_, hi_ = thr, 1 - thr
        reg = np.where(D[s + "p"] < lo_, 1, np.where(D[s + "p"] > hi_, -1, 0))   # +1 跟隨、−1 反向
        for mode in ("both", "trend_only", "revert_only"):
            pos = D.dir * reg
            if mode == "trend_only": pos = pos.where(reg == 1, 0)
            if mode == "revert_only": pos = pos.where(reg == -1, 0)
            pos = pos.where(~D.roll1, 0).fillna(0)
            nx_o = nxt.open.reindex(D.index)
            c_pts = np.abs(pos) * (cost + FTAX * 2 * nx_o)
            pnl = ((pos * D.oc1).fillna(0) - c_pts) * FMULT
            nh, nl = nxt.high.reindex(D.index), nxt.low.reindex(D.index)
            worst = (np.where(pos > 0, nl - nx_o, np.where(pos < 0, nx_o - nh, 0)) - c_pts) * FMULT
            worst = pd.Series(worst, D.index).fillna(0)
            pnl = pd.Series(pnl.values, D.index); key = (s, thr, mode); keep[key] = (pnl, worst)
            r = dict(sign=s, thr=thr, mode=mode)
            for nm, a, b in (("IS", pd.Timestamp("2017-01-01"), IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1))):
                r.update({f"{nm}_{k}": v for k, v in tx_stats(pnl, worst, a, b).items()})
            rows.append(r)
    R2 = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R2.round(3).to_string(index=False))
    print("Spearman IS/OOS:", round(R2.IS_Sharpe.rank().corr(R2.OOS_Sharpe.rank()), 3))
    b2 = R2.loc[R2.IS_Sharpe.idxmax()]; kb = (b2["sign"], b2["thr"], b2["mode"]); pnl, worst = keep[kb]
    oos = pnl[pnl.index >= IS_END]; act = oos[oos != 0]
    base = D.oc1[D.index >= IS_END].fillna(0)
    rnd = []
    for _ in range(500):
        sgn = rng.choice([-1, 1], len(base)); m = rng.random(len(base)) < (act.size / len(base))
        rnd.append(((sgn * base.values - (cost + FTAX * 2 * nxt.open.reindex(D.index)[D.index >= IS_END].values)) * m).sum() * FMULT)
    print(f"best {kb} OOS net {oos.sum():.0f}; 隨機方向同頻率 百分位 {(np.array(rnd) < oos.sum()).mean():.2f} 中位數 {np.median(rnd):.0f}")
    # 條件自相關
    for s in "AB":
        for nm, a, b in (("IS", "2017", "2022"), ("OOS", "2022", "2027")):
            x = D[(D.index >= a) & (D.index < b)]
            for lab, m in (("負/低 gamma p<.3", x[s + "p"] < .3), ("正/高 gamma p>.7", x[s + "p"] > .7)):
                print(f"  {s} {nm} {lab}: Spearman(r_t, oc_t+1) {x[m][["r0","oc1"]].dropna().corr("spearman").iloc[0,1]:+.3f} n={m.sum()}")
    wf = []; picks = []
    for Y in range(2020, 2027):
        a, b, c = pd.Timestamp(f"{Y-3}-01-01"), pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y+1}-01-01")
        sc = {k: sharpe(v[0][(v[0].index >= a) & (v[0].index < b)]) for k, v in keep.items()}
        k = max(sc, key=sc.get); picks.append((Y, k, round(sc[k], 2))); wf.append(keep[k][0][(keep[k][0].index >= b) & (keep[k][0].index < c)])
    wf = pd.concat(wf); wo = wf[wf.index >= IS_END]; print("WF picks", picks); print(f"WF 2022+ Sharpe {sharpe(wo):.2f} net {wo.sum():.0f}")
    print("\n== 3. 結算日 pinning ==")
    from tools.optdaily import expiries
    E = expiries(); E = E[(E.expiry_date >= "2017-01-01") & (E.expiry_date <= END)]
    pr = []
    days = F.index
    for _, e in E.iterrows():
        xd = e.expiry_date
        if xd not in days: continue
        p = days.searchsorted(xd) - 1; t0 = days[p]
        x = o[(o.date == t0) & (o.expiry == e.expiry)]
        if x.empty: continue
        oi = x.groupby("K").oi.sum(); Ks = oi.idxmax(); ks = np.array(sorted(oi.index)); step = np.median(np.diff(ks)) if len(ks) > 2 else 50
        op, cl, hi, lo = F.loc[xd, ["open", "high", "low", "close"]]
        row = dict(xd=xd, kind=e.kind, K=Ks, open=op, close=cl, high=hi, low=lo, step=step, F0=F.close[t0],
                   d_open=abs(op - Ks), d_close=abs(cl - Ks))
        for j in (2, 5):
            for sg in (-1, 1): row[f"pl{sg*j}"] = abs(cl - (Ks + sg * j * step)) - abs(op - (Ks + sg * j * step))
        pr.append(row)
    P = pd.DataFrame(pr); P["shrink"] = P.d_close - P.d_open
    P["plc"] = P[[c for c in P if c.startswith("pl")]].mean(axis=1)
    for nm, a, b in (("IS", "2017", "2022"), ("OOS", "2022", "2027")):
        x = P[(P.xd >= a) & (P.xd < b)]
        print(f"{nm} n={len(x)} 距 K* 縮小比例 {(x.shrink<0).mean():.3f} 平均變化 {x.shrink.mean():+.1f} 點 | 安慰劑(±2,±5 檔) 平均 {x.plc.mean():+.1f} 縮小比例 {(x.plc<0).mean():.3f} | 收盤距 K* 中位 {x.d_close.median():.0f} 開盤 {x.d_open.median():.0f}")
    prow = []; pkeep = {}
    for thr in (0, 50, 100, 200):
        sgn = np.where(P.open - P.K > thr, -1, np.where(P.K - P.open > thr, 1, 0))
        c_pts = np.abs(sgn) * (cost + FTAX * 2 * P.open)
        pnl = pd.Series(((sgn * (P.close - P.open)) - c_pts).values * FMULT, P.xd.values)
        worst = pd.Series((np.where(sgn > 0, P.low - P.open, np.where(sgn < 0, P.open - P.high, 0)) - c_pts).values * FMULT, P.xd.values)
        pkeep[thr] = pnl; r = dict(thr=thr)
        for nm, a, b in (("IS", pd.Timestamp("2017-01-01"), IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1))):
            st = tx_stats(pnl, worst, a, b); st["Sharpe"] = (pnl[(pnl.index >= a) & (pnl.index < b)].pipe(lambda s: s.mean() / s.std() * math.sqrt(len(s) / ((b - a).days / 365.25))))
            r.update({f"{nm}_{k}": v for k, v in st.items()})
        prow.append(r)
    R3 = pd.DataFrame(prow); print(R3.round(3).to_string(index=False))
    # 對照：結算日一律做多、安慰劑履約價 K*±2 檔（門檻 100）——檢查是否只是多頭漂移／K* 常在價外 call 端
    for lab, ref in (("K*", P.K), ("K*+2檔", P.K + 2 * P.step), ("K*-2檔", P.K - 2 * P.step), ("前日收盤", P.F0), ("一律做多", None)):
        sgn = np.ones(len(P)) if ref is None else np.where(P.open - ref > 100, -1, np.where(ref - P.open > 100, 1, 0))
        pp = pd.Series((sgn * (P.close - P.open) - np.abs(sgn) * (cost + FTAX * 2 * P.open)).values * FMULT, P.xd.values)
        print(f"  對照 {lab}: IS net {pp[pp.index < IS_END].sum():.0f} OOS net {pp[pp.index >= IS_END].sum():.0f} 做多比例 {(sgn > 0).mean():.2f}")
    print("  K* 相對前日收盤 中位數（點）:", (P.K - P.F0).median(), " K*>前收 比例", round((P.K > P.F0).mean(), 3))
    T1.to_csv(RES / "rqo_71_vol.csv", index=False); R2.to_csv(RES / "rqo_71_trend.csv", index=False); R3.to_csv(RES / "rqo_71_pin.csv", index=False); P.to_csv(RES / "rqo_71_pin_events.csv", index=False)
    plt = plt_style(); fig, axs = plt.subplots(3, 1, figsize=(12, 12), facecolor=BG)
    ax = axs[0]; sty(ax); ax.plot(D.index, D.An, color=GOLD, lw=.7, label="假設 A（call+ / put−）"); ax.plot(D.index, D.Bn, color=BLUE, lw=.7, label="假設 B（全部 −）")
    ax.axhline(0, color=GR); ax.set_title("TXO OI 推估 dealer GEX（以指數² 正規化）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=9)
    ax = axs[1]; sty(ax)
    for s, col in (("A", GOLD), ("B", BLUE)):
        q = pd.qcut(D[s + "p"], 5, labels=False); m = D.groupby(q).rng1.mean() * 100
        ax.bar(np.arange(5) + (0 if s == "A" else .4), m.values, width=.4, color=col, label=f"假設 {s}")
    ax.set_xticks(np.arange(5) + .2); ax.set_xticklabels(["GEX 最低 20%", "2", "3", "4", "最高 20%"]); ax.set_ylabel("次日 TXF 日內振幅 %")
    ax.set_title("GEX 五分位 vs 次日振幅（全期 2017–2026）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR)
    ax = axs[2]; sty(ax); ax.plot(pnl.cumsum() / 1e4, color=GOLD, lw=2, label=f"IS 最佳 {kb}"); ax.plot(wf.cumsum() / 1e4, color=BLUE, label="walk-forward")
    ax.plot(pkeep[100].cumsum() / 1e4, color=GR, label="結算日朝 K* pinning（門檻 100 點）"); ax.axvline(IS_END, color="white", ls="--")
    ax.set_title(f"gamma regime 跟隨/反向 TXF｜IS Sharpe {b2.IS_Sharpe:.2f} → OOS {b2.OOS_Sharpe:.2f}"); ax.set_ylabel("累積淨損益（萬元，1 口）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=9)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "rqo_71_dealer_gamma.png", dpi=110, facecolor=BG)


if __name__ == "__main__":
    main()
