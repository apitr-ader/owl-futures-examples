"""56 Initial Balance（IB）突破 台指期 1 分 K 回測 + 樣本內最佳化。

參考：TradingView「Initial Balance Breakout Strategy」(MyjxUMGt)；research/tradingview_strategies.md 第二輪 H。
框架、資料、成本、成交慣例同 55（import 55 的 load_bars / manage）。不 commit 任何資料。
IB：日盤 08:45 起 N 分（N=30/60）；夜盤 15:00 起 N 分，或美股開盤後 1 小時（夏令 21:30、冬令 22:30，'nus'）。
進場：(stop) 停損單 IBH+buf / IBL−buf，buf = 1 tick + k×ATR14(1 分 K)，k∈{0,2}；觸發後成交價 = max(觸發價, 開盤) + 1 點滑價。
     同一根同時穿越上下緣：開盤在外側者成立，否則跳過該根。
     (retest) 收盤站上 IBH（跌破 IBL）後，於 IB 邊緣掛限價，需穿價 ≥1 tick 才成交（保守），無滑價。
停損：opp = IB 另一側；mid = IB 中點；q25 = 進場價 ∓ 0.25×IB 寬（固定 R）。停損出場 1 點滑價、跳空以開盤價。
停利：進場 ± m×IB 寬（m=1/1.5/2/3，限價、無滑價）或 0 = 抱到收盤（收盤平倉 1 點滑價）。同根碰停損+停利 → 停損。
成交當根碰停損 → 停損；成交當根不算停利。
不開新倉時間：日盤 13:00、夜盤 04:00；收盤強平（日盤 13:45、夜盤 05:00）。
每盤：每方向最多 1 筆（'dir'）或全部 1 筆（'one'）；同時只持 1 部位。
濾網（皆為無單位、可跨指數水位）：
  寬度：IB 寬在同類 IB 過去 L=20/60 盤中的百分位，'lt50' 只做窄 IB、'gt50' 只做寬 IB。
  bias：'close' = IB 結束價 vs 參考收盤（日盤：前一日盤收；夜盤：當日日盤收）順勢；
        'prev' = 前一盤（日盤看前一夜盤、夜盤看當日日盤）開→收方向順勢。
成本每邊 50 元 + 2e-5 期交稅 + 市價/停損 1 點滑價；1 口 TXF 200 元/點；年化以固定 100 萬元/口（不複利）。
IS 2017-06~2021，OOS 2022+；目標 IS 日損益 Sharpe（至少 150 筆）。
"""
import sys, math, itertools, importlib.util
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("sb55", HERE / "55_txf_ict_silver_bullet.py"); sb = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(sb)
load_bars, manage = sb.load_bars, sb.manage
OUT = Path("results"); CAP = 1_000_000; PV = 200.0; IS_END = pd.Timestamp("2022-01-01"); MIN_IS = 150
TYPES = ("d30", "d60", "n30", "n60", "nus")
COMBOS = {"d30+n30": ("d30", "n30"), "d60+n60": ("d60", "n60"), "d60+nus": ("d60", "nus")}
ENTRY = (("stop", 0.0), ("stop", 2.0), ("retest", 0.0)); WFIL = ("none", "lt50_20", "gt50_20", "lt50_60", "gt50_60")
BIAS = ("none", "close", "prev"); STOPM = ("opp", "mid", "q25"); TGT = (1.0, 1.5, 2.0, 3.0, 0.0); MAXT = ("dir", "one")


def build(d):
    ts = d.ts.values.astype("datetime64[ns]").astype(np.int64)
    sess = d.groupby(["skey", "night"]).agg(i0=("ts", lambda x: x.index[0]), i1=("ts", lambda x: x.index[-1]),
                                            op=("open", "first"), cl=("close", "last")).reset_index()
    day = sess[~sess.night].set_index("skey"); ngt = sess[sess.night].set_index("skey")
    days = day.index.values; ny = ZoneInfo("America/New_York"); tp = ZoneInfo("Asia/Taipei")
    H, L, C = d.high.values, d.low.values, d.close.values
    idx = lambda t: np.searchsorted(ts, np.datetime64(t, "ns").astype(np.int64))
    rows = []
    def add(typ, tdate, i0, i1, ibs, ibe, cut, ref, prev_mv):
        s = max(idx(ibs), i0); e = min(idx(ibe), i1 + 1); ce = min(idx(cut), i1 + 1)
        if e - s < 5 or ce <= e: return
        hi, lo = H[s:e].max(), L[s:e].min()
        rows.append(dict(typ=typ, tdate=tdate, e=e, ce=ce, fe=i1 + 1, ibh=hi, ibl=lo, w=hi - lo,
                         b_close=np.sign(C[e - 1] - ref) if ref == ref else 0.0, b_prev=np.sign(prev_mv) if prev_mv == prev_mv else 0.0))
    for k, D in enumerate(days):
        r = day.loc[D]; D = pd.Timestamp(D)
        ref = day.loc[days[k - 1]].cl if k else np.nan
        pn = ngt.index[ngt.index < D]
        pm = (ngt.loc[pn[-1]].cl - ngt.loc[pn[-1]].op) if len(pn) and (D - pd.Timestamp(pn[-1])).days <= 5 else np.nan
        for N in (30, 60):
            add(f"d{N}", D, r.i0, r.i1, D + pd.Timedelta("8h45min"), D + pd.Timedelta("8h45min") + pd.Timedelta(minutes=N), D + pd.Timedelta("13h"), ref, pm)
    for N0 in ngt.index:
        r = ngt.loc[N0]; N = pd.Timestamp(N0)
        nd = days[days > N0]; td = pd.Timestamp(nd[0]) if len(nd) else N + pd.Timedelta(days=1)
        ref, pm = (day.loc[N0].cl, day.loc[N0].cl - day.loc[N0].op) if N0 in day.index else (np.nan, np.nan)
        cut = N + pd.Timedelta("1D4h")
        for N_ in (30, 60):
            add(f"n{N_}", td, r.i0, r.i1, N + pd.Timedelta("15h"), N + pd.Timedelta("15h") + pd.Timedelta(minutes=N_), cut, ref, pm)
        ws = pd.Timestamp(N.year, N.month, N.day, 9, 30, tzinfo=ny).astimezone(tp).replace(tzinfo=None)
        add("nus", td, r.i0, r.i1, ws, ws + pd.Timedelta(hours=1), cut, ref, pm)
    I = pd.DataFrame(rows).sort_values(["typ", "tdate"]).reset_index(drop=True)
    for L_ in (20, 60):  # 過去 L 盤（不含當盤）IB 寬百分位
        I[f"pct{L_}"] = I.groupby("typ").w.transform(lambda x: x.rolling(L_ + 1).apply(lambda v: (v[:-1] < v[-1]).mean(), raw=True))
    return I


@njit(cache=True)
def run(o, h, l, c, atr, E, CE, FE, IBH, IBL, ALLOW, BD, mode, kbuf, stopm, tm, one):
    n = len(E); out = np.full((n, 2, 7), np.nan)  # dir, ep, stop, xp, kind, amb, risk
    for k in range(n):
        if not ALLOW[k]: continue
        hi = IBH[k]; lo = IBL[k]; W = hi - lo
        if W <= 0: continue
        usedL = False; usedS = False; armL = False; armS = False
        j = E[k]
        while j < CE[k]:
            if one and (usedL or usedS): break
            canL = (not usedL) and BD[k] >= 0; canS = (not usedS) and BD[k] <= 0
            if not canL and not canS: break
            dr = 0; ep = 0.0
            if mode == 0:
                buf = 1.0 + kbuf * atr[j - 1]; tL = hi + buf; tS = lo - buf
                hl = canL and h[j] >= tL; hs = canS and l[j] <= tS
                if hl and hs:
                    if o[j] >= tL: hs = False
                    elif o[j] <= tS: hl = False
                    else: hl = False; hs = False
                if hl: dr = 1; ep = max(o[j], tL) + 1.0
                elif hs: dr = -1; ep = min(o[j], tS) - 1.0
            else:
                if armL and canL and l[j] <= hi - 1.0: dr = 1; ep = min(o[j], hi)
                elif armS and canS and h[j] >= lo + 1.0: dr = -1; ep = max(o[j], lo)
                if c[j] > hi: armL = True
                if c[j] < lo: armS = True
            if dr == 0:
                j += 1; continue
            if stopm == 0: stp = lo if dr > 0 else hi
            elif stopm == 1: stp = (hi + lo) / 2
            else: stp = ep - dr * 0.25 * W
            risk = dr * (ep - stp)
            if risk <= 0:
                stp = ep - dr * 0.25 * W; risk = 0.25 * W
            tg = ep + dr * tm * W if tm > 0 else dr * 1e9
            slot = 0 if dr > 0 else 1
            if (dr > 0 and l[j] <= stp) or (dr < 0 and h[j] >= stp):
                xp = stp - dr * 1.0; kind = 1; amb = 0; xj = j
            else:
                xp, kind, amb, xj = manage(o, h, l, c, atr, j, FE[k], dr, ep, stp, tg, False, risk)
            out[k, slot, 0] = dr; out[k, slot, 1] = ep; out[k, slot, 2] = stp; out[k, slot, 3] = xp
            out[k, slot, 4] = kind; out[k, slot, 5] = amb; out[k, slot, 6] = risk
            if dr > 0: usedL = True
            else: usedS = True
            armL = False; armS = False
            j = xj + 1
    return out


@njit(cache=True)
def run_random(o, h, l, c, atr, RB, FE, DR, RK, TD):
    n = len(RB); out = np.full((n, 7), np.nan)
    for k in range(n):
        j = RB[k]
        if j < 0: continue
        dr = DR[k]; ep = o[j] + dr; rk = RK[k]; stp = ep - dr * rk; tg = ep + dr * TD[k] if TD[k] > 0 else dr * 1e9
        if (dr > 0 and l[j] <= stp) or (dr < 0 and h[j] >= stp):
            xp = stp - dr; kind = 1; amb = 0
        else:
            xp, kind, amb, xj = manage(o, h, l, c, atr, j, FE[k], dr, ep, stp, tg, False, rk)
        out[k, 0] = dr; out[k, 1] = ep; out[k, 3] = xp; out[k, 4] = kind; out[k, 5] = amb; out[k, 6] = rk
    return out


def to_trades(out, J):
    if out.ndim == 3:
        rep = np.repeat(np.arange(len(J)), 2); out = out.reshape(-1, 7)
    else: rep = np.arange(len(J))
    t = pd.DataFrame(out, columns=["dir", "ep", "stop", "xp", "kind", "amb", "risk"]); t["tdate"] = J.tdate.values[rep]; t["typ"] = J.typ.values[rep]
    t = t.dropna(subset=["dir"]).copy()
    t["net"] = t.dir * (t.xp - t.ep) - (0.5 + 2e-5 * (t.ep + t.xp)); t["R"] = t.net / t.risk; t["W"] = np.nan
    return t


def metrics(t, alld, lo=None, hi=None):
    if lo is not None: t = t[(t.tdate >= lo) & (t.tdate < hi)]; alld = alld[(alld >= lo) & (alld < hi)]
    n = len(t)
    if n == 0: return dict(trades=0, Sharpe=np.nan)
    dp = t.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV; eq = dp.cumsum()
    w = t.net[t.net > 0].sum(); ls = -t.net[t.net < 0].sum()
    return dict(trades=n, win=(t.net > 0).mean(), avgR=t.R.mean(), PF=w / ls if ls > 0 else np.nan, net_pts=t.net.sum(), net_twd=t.net.sum() * PV,
                ann_ret=dp.sum() / (len(alld) / 252) / CAP, Sharpe=dp.mean() / dp.std() * math.sqrt(252) if dp.std() > 0 else np.nan,
                MDD_twd=(eq - eq.cummax()).min(), amb=int(t.amb.sum()))


def fast_daily(t, dix, nd):
    return np.bincount(dix[t.index.values] if False else t._di.values, weights=t.net.values * PV, minlength=nd)


def sharpe(x):
    s = x.std(); return x.mean() / s * math.sqrt(252) if s > 0 else np.nan


def param_iter():
    return itertools.product(ENTRY, WFIL, BIAS, STOPM, TGT, MAXT)


def sim_type(I, arr, typ, p, touch=None):
    (em, kb), wf, bi, sm, tm, mx = p
    J = I[I.typ == typ]
    allow = np.ones(len(J), bool)
    if wf != "none":
        side, L_ = wf.split("_"); pc = J[f"pct{L_}"].values
        allow = (pc < .5) if side == "lt50" else (pc > .5)
        allow &= ~np.isnan(pc)
    bd = np.zeros(len(J)) if bi == "none" else J["b_" + bi].values.astype(float)
    if bi != "none": allow &= bd != 0
    o, h, l, c, a = arr
    out = run(o, h, l, c, a, J.e.values, J.ce.values, J.fe.values, J.ibh.values, J.ibl.values, allow, bd,
              0 if em == "stop" else 1, kb, STOPM.index(sm), tm, mx == "one")
    return to_trades(out, J)


def main():
    OUT.mkdir(exist_ok=True)
    d = load_bars(); I = build(d); arr = tuple(d[k].values for k in ("open", "high", "low", "close", "atr"))
    alld = pd.DatetimeIndex(sorted(I.tdate.unique())); nd = len(alld); dpos = pd.Series(np.arange(nd), index=alld)
    isd = alld < IS_END
    print("instances", I.groupby("typ").size().to_dict(), alld[0].date(), alld[-1].date())
    params = list(param_iter()); keys = []; daily = []; cnt = []; T = {}
    for typ in TYPES:
        for p in params:
            t = sim_type(I, arr, typ, p); t["_di"] = dpos.reindex(t.tdate).values
            T[(typ, p)] = t
    print("sims done")
    rows = []
    def rec(win, p, t):
        dv = np.bincount(t._di.values, weights=t.net.values * PV, minlength=nd)
        isn = int((t.tdate < IS_END).sum()); on = len(t) - isn
        def m(mask_t, mask_d):
            tt = t[mask_t]; w = tt.net[tt.net > 0].sum(); ls = -tt.net[tt.net < 0].sum()
            dd = dv[mask_d]; eq = dd.cumsum()
            return dict(trades=len(tt), win=(tt.net > 0).mean() if len(tt) else np.nan, avgR=tt.R.mean() if len(tt) else np.nan,
                        PF=w / ls if ls > 0 else np.nan, net_twd=tt.net.sum() * PV, ann_ret=dd.sum() / (len(dd) / 252) / CAP,
                        Sharpe=sharpe(dd), MDD_twd=(eq - np.maximum.accumulate(eq)).min())
        r = dict(win=win, entry=f"{p[0][0]}{'' if p[0][0]=='retest' else '_k'+str(p[0][1])}", wfil=p[1], bias=p[2], stop=p[3], tgt=p[4], maxt=p[5])
        isT = (t.tdate < IS_END).values
        r.update({f"IS_{k}": v for k, v in m(isT, isd).items()}); r.update({f"OOS_{k}": v for k, v in m(~isT, ~isd).items()})
        o25 = (~isd) & (alld.year != 2025); r["OOSx25_Sharpe"] = sharpe(dv[o25]); r["y2025_twd"] = dv[alld.year == 2025].sum()
        rows.append(r); daily.append(dv.astype(np.float32)); keys.append((win, p))
    for typ in TYPES:
        for p in params: rec(typ, p, T[(typ, p)])
    for cn, (a, b) in COMBOS.items():
        for p in params: rec(cn, p, pd.concat([T[(a, p)], T[(b, p)]]))
    G = pd.DataFrame(rows); G["id"] = range(len(G)); DP = np.vstack(daily)
    G.drop(columns=[]).to_csv(OUT / "ib_grid.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    ok = G[G.IS_trades >= MIN_IS].copy(); ok["OOS_rank"] = ok.OOS_Sharpe.rank(ascending=False)
    top = ok.sort_values("IS_Sharpe", ascending=False)
    cols = ["win", "entry", "wfil", "bias", "stop", "tgt", "maxt", "IS_trades", "IS_win", "IS_avgR", "IS_PF", "IS_ann_ret", "IS_Sharpe",
            "OOS_trades", "OOS_win", "OOS_avgR", "OOS_PF", "OOS_net_twd", "OOS_ann_ret", "OOS_Sharpe", "OOS_MDD_twd", "OOSx25_Sharpe", "y2025_twd", "OOS_rank"]
    print(f"configs {len(G)}, IS trades>={MIN_IS}: {len(ok)}")
    print("\n== IS top10 =="); print(top.head(10)[cols].round(3).to_string(index=False))
    sp = lambda s: s.IS_Sharpe.rank().corr(s.OOS_Sharpe.rank())
    print("\nSpearman IS/OOS %.3f | IS>0 %.3f OOS>0 %.3f OOS-ex2025>0 %.3f | both>0 %.3f" % (sp(ok), (ok.IS_Sharpe > 0).mean(), (ok.OOS_Sharpe > 0).mean(), (ok.OOSx25_Sharpe > 0).mean(), ((ok.IS_Sharpe > 0) & (ok.OOS_Sharpe > 0)).mean()))
    for w, s in ok.groupby("win"):
        print(f"  {w:8s} N={len(s):4d} sp {sp(s):.3f} IS>0 {(s.IS_Sharpe>0).mean():.2f} OOS>0 {(s.OOS_Sharpe>0).mean():.2f} OOSx25>0 {(s.OOSx25_Sharpe>0).mean():.2f} meanIS {s.IS_Sharpe.mean():.3f} meanOOS {s.OOS_Sharpe.mean():.3f} bestIS {s.IS_Sharpe.max():.2f}")
    for k in ("entry", "wfil", "bias", "stop", "tgt", "maxt"):
        print(ok.groupby(k)[["IS_Sharpe", "OOS_Sharpe", "OOSx25_Sharpe", "IS_avgR", "OOS_avgR"]].mean().round(3).to_string()); print()
    best = top.iloc[0]; bw, bp = keys[int(best.id)]
    print("BEST", bw, bp)
    btypes = COMBOS.get(bw, (bw,))
    def simb(I_, arr_, p=bp): return pd.concat([sim_type(I_, arr_, ty, p) for ty in btypes])
    bt = simb(I, arr)
    per = [("IS", alld[0], IS_END), ("OOS", IS_END, alld[-1] + pd.Timedelta(days=1)), ("FULL", alld[0], alld[-1] + pd.Timedelta(days=1))]
    def show(t, lab):
        print(lab); print(pd.DataFrame({pn: metrics(t, alld, lo, hi) for pn, lo, hi in per}).T.round(3).to_string())
    show(bt, "best")
    o25 = bt[bt.tdate.dt.year != 2025]; print("OOS excl 2025:", {k: round(v, 3) for k, v in metrics(o25, alld[alld.year != 2025], IS_END, per[1][2]).items()})
    for ty in btypes:
        if len(btypes) > 1: show(bt[bt.typ == ty], f"best, {ty} only")
    bt["year"] = bt.tdate.dt.year
    yr = bt.groupby("year").agg(trades=("net", "size"), win=("net", lambda x: (x > 0).mean()), avgR=("R", "mean"), net_pts=("net", "sum"))
    yr["net_twd"] = yr.net_pts * PV; yr["ret_on_1M"] = yr.net_twd / CAP; print("\nper year (best):"); print(yr.round(3).to_string())
    print(f"ambiguous stop+target bars: {int(bt.amb.sum())} of {len(bt)}")
    # 隨機進場基準：同盤別、IB 結束後到截止前隨機一根市價進場、隨機方向；停損距離與停利距離取自最佳參數實際交易
    rng = np.random.default_rng(7); J = I[I.typ.isin(btypes)].reset_index(drop=True)
    rk_pool = bt.risk.values; td_pool = (np.abs(bt.ep - bt.stop).values * 0 + 1)  # placeholder
    W_all = J.w.values; tm = bp[4]
    o, h, l, c, a = arr; stats = []
    for rep in range(200):
        pick = np.zeros(len(J), bool); pick[rng.choice(len(J), size=min(len(bt), len(J)), replace=False)] = True
        RB = np.where(pick, [rng.integers(e, ce) for e, ce in zip(J.e.values, J.ce.values)], -1)
        DR = rng.choice([-1, 1], size=len(J)); ii = rng.integers(0, len(bt), size=len(J))
        RK = rk_pool[ii]; TD = (W_all * tm) if tm > 0 else np.zeros(len(J))
        tr = to_trades(run_random(o, h, l, c, a, RB, J.fe.values, DR.astype(float), RK, TD.astype(float)), J)
        stats.append({pn: metrics(tr, alld, lo, hi) for pn, lo, hi in per})
    for pn, lo, hi in per:
        sh = np.array([s[pn]["Sharpe"] for s in stats]); real = metrics(bt, alld, lo, hi)["Sharpe"]
        print(f"random {pn}: Sharpe mean {sh.mean():.3f} [5%,95%] [{np.percentile(sh,5):.3f},{np.percentile(sh,95):.3f}] avgR {np.mean([s[pn]['avgR'] for s in stats]):.3f} | real {real:.3f} pctile {(sh < real).mean():.2f}")
    for rule in ("3min", "5min"):
        dd = load_bars(rule); II = build(dd); show(simb(II, tuple(dd[k].values for k in ("open", "high", "low", "close", "atr"))), f"best on {rule}")
    # walk-forward
    cntY = {}
    picks = []; wf = []
    for Y in range(2020, alld.year.max() + 1):
        trm = (alld.year >= Y - 3) & (alld.year < Y); tsm = alld.year == Y
        X = DP[:, trm]; sh = X.mean(1) / X.std(1) * math.sqrt(252)
        ntr = np.array([0] * len(G))
        # 交易數門檻：訓練期至少 90 筆（以 IS 平均頻率估計）
        freq = (G.IS_trades / isd.sum()).values; sh[freq * trm.sum() < 90] = np.nan
        b = int(np.nanargmax(sh)); y = DP[b, tsm]; wf.append(pd.Series(y, index=alld[tsm]))
        picks.append(dict(year=Y, **G.loc[b, ["win", "entry", "wfil", "bias", "stop", "tgt", "maxt"]].to_dict(), train_sharpe=round(sh[b], 3), test_sharpe=round(sharpe(y), 3), test_twd=round(y.sum())))
    wfr = pd.concat(wf); P = pd.DataFrame(picks); print("\n== walk-forward (trailing 3y) =="); print(P.to_string(index=False))
    ex = wfr[wfr.index.year != 2025]
    print("WF stitched: net %.0f ann %.2f%% Sharpe %.3f MDD %.0f | excl 2025 Sharpe %.3f net %.0f" % (wfr.sum(), wfr.sum() / (len(wfr) / 252) / CAP * 100, sharpe(wfr.values), (wfr.cumsum() - wfr.cumsum().cummax()).min(), sharpe(ex.values), ex.sum()))
    bt.to_csv(OUT / "ib_best_trades.csv", index=False); P.to_csv(OUT / "ib_wf_picks.csv", index=False)
    chart(ok, bt, alld, wfr)


def chart(ok, bt, alld, wfr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK TC"; plt.rcParams["axes.unicode_minus"] = False
    wins = list(TYPES) + list(COMBOS)
    for name, (ix, cx) in {"ib_heatmap_stop_tgt.png": ("stop", "tgt"), "ib_heatmap_entry_filter.png": ("entry", "wfil")}.items():
        fig, ax = plt.subplots(2, len(wins), figsize=(3.2 * len(wins), 7.5))
        for j, w in enumerate(wins):
            s = ok[ok.win == w]
            for i, k in enumerate(("IS_Sharpe", "OOS_Sharpe")):
                a = ax[i, j]; P = s.pivot_table(index=ix, columns=cx, values=k, aggfunc="mean")
                a.imshow(P.values, cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
                for (y, x), v in np.ndenumerate(P.values): a.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=7)
                a.set_xticks(range(len(P.columns))); a.set_xticklabels(["收盤" if c == 0 else c for c in P.columns], fontsize=7, rotation=45)
                a.set_yticks(range(len(P.index))); a.set_yticklabels(P.index, fontsize=7); a.set_title(f"{w} {k[:-7]} Sharpe 平均", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / name, dpi=100); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6)); a = ax[0]
    a.plot(alld, (bt.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV).cumsum() / 1e4, label="IS 最佳參數")
    a.plot(wfr.index, wfr.cumsum().values / 1e4, label="walk-forward（3 年滾動）")
    a.axvline(IS_END, c="gray", ls="--"); a.set_ylabel("累積淨損益（萬元，1 口 TXF）"); a.legend(); a.grid(alpha=.3); a.set_title("IB 突破 台指期 1 分 K")
    a = ax[1]
    for w, s in ok.groupby("win"): a.scatter(s.IS_Sharpe, s.OOS_Sharpe, s=4, label=w)
    a.axhline(0, c="gray"); a.axvline(0, c="gray"); a.legend(markerscale=3, fontsize=8); a.set_xlabel("IS Sharpe 2017-21"); a.set_ylabel("OOS Sharpe 2022+"); a.grid(alpha=.3)
    a.set_title("各參數 IS vs OOS Sharpe")
    fig.tight_layout(); fig.savefig(OUT / "ib_equity.png", dpi=100)
    print("charts results/ib_heatmap_stop_tgt.png results/ib_heatmap_entry_filter.png results/ib_equity.png")


if __name__ == "__main__":
    main()
