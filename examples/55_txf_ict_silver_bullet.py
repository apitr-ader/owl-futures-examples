"""55 ICT Silver Bullet（FVG 回補）台指期 1 分 K 回測 + 樣本內最佳化。

參考：MarkLeRoy「ICT Silver Bullet Backtest」、JadeCap Silver Bullet（掃流動性）；research/tradingview_strategies.md 第二輪 B/C。
資料：data/TXF_1c_1m_both_*.parquet（近月連續、結算日夜盤換月；K 棒時間 = 該分鐘開始）。不 commit 任何資料。
窗口（台北時間）：open 08:45–09:45、pm 12:30–13:30、night = 紐約 09:30 起 1 小時（夏令 21:30、冬令 22:30）。
FVG：三根 K，多 low[j] > high[j-2]、空 high[j] < low[j-2]，三根都在窗口內，缺口 ≥ minFVG 點。
進場：FVG 完成後掛限價（50% 或邊緣；多單向下取整、空單向上取整），窗口內有更新的合格 FVG 則改掛；窗口結束未成交取消。
  成交：保守 = 價格穿越 ≥1 tick（多 low ≤ 限價−1）；touch = 觸價即成交。成交 K 若碰停損 → 停損；成交 K 不算停利（保守）。
停損：FVG 第一根 K 外 1 tick；停利：固定 R（1/1.5/2/3）或 trail（1.5R 後 2×ATR14 移動停損，無固定停利）。
同一根 K 同時碰停損與停利 → 算停損（可用 tick 解析）。跳空越過停損 → 以開盤價出場。
每窗口一筆；強平：窗口結束、窗口結束 +30 分、或盤別結束（不超過該盤收盤）。
濾網：sweep = 進場前（盤別開始至 FVG 完成）須先掃過參考高/低（日盤窗口：前一夜盤高低；夜盤窗口：當日日盤高低），
  只做反向 FVG（掃低 → 做多；掃高 → 做空）。bias = 多單需 FVG 完成 K 收盤 > 參考收盤（日盤：前一日盤收盤；夜盤：當日日盤收盤）。
成本（每邊）：50 元、期交稅 2e-5×價格×200、市價/停損出場滑價 1 點；限價進場與停利無滑價。1 口 TXF = 200 元/點（MTX = 1/4）。
報酬：以固定資本 100 萬元/口計（約 2~3 倍原始保證金），年化 = 年均淨損益 / 100 萬（不複利）。
IS 2017-06~2021；OOS 2022+；目標：IS 日損益 Sharpe（至少 100 筆）。
"""
import sys, math, itertools
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
DATA = "data/TXF_1c_1m_both_2017-06-01_2026-10-22.parquet"
OUT = Path("results"); CAP = 1_000_000; PV = 200.0
IS_END = pd.Timestamp("2022-01-01")
WINS = ("open", "pm", "night"); MINF = (0, 5, 10, 20); ENTRY = ("mid", "edge"); TGT = (1.0, 1.5, 2.0, 3.0, 0.0)  # 0 = trail
FLAT = ("w0", "w30", "sess")


def load_bars(rule=None):
    d = pd.read_parquet(DATA, columns=["date", "session", "open", "high", "low", "close"])
    d["ts"] = d.date.dt.tz_localize(None); d = d.drop(columns="date").sort_values("ts").reset_index(drop=True)
    reg = d.session.astype(str) == "regular"
    day = d.ts.dt.normalize()
    d["skey"] = np.where(reg, day, np.where(d.ts.dt.hour >= 15, day, day - pd.Timedelta(days=1)))
    d["night"] = ~reg
    if rule:
        g = d.groupby(["skey", "night", d.ts.dt.floor(rule)])
        d = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).reset_index()
        d = d.rename(columns={"ts": "ts"}).sort_values("ts").reset_index(drop=True)
    for k in ("open", "high", "low", "close"): d[k] = d[k].astype(float)
    tr = np.maximum(d.high - d.low, np.maximum((d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()))
    d["atr"] = tr.rolling(14, min_periods=1).mean()
    return d


def build_instances(d, rule_min=1):
    ts = d.ts.values.astype("datetime64[ns]").astype(np.int64)
    sess = d.groupby(["skey", "night"]).agg(i0=("ts", lambda x: x.index[0]), i1=("ts", lambda x: x.index[-1]),
                                            hi=("high", "max"), lo=("low", "min"), cl=("close", "last")).reset_index()
    day = sess[~sess.night].set_index("skey"); ngt = sess[sess.night].set_index("skey")
    days = day.index.values; nights = ngt.index.values
    ny = ZoneInfo("America/New_York"); tp = ZoneInfo("Asia/Taipei")
    rows = []
    def add(win, tdate, i0, i1, ws, we, ref_hi, ref_lo, bias):
        s = np.searchsorted(ts, np.datetime64(ws, "ns").astype(np.int64)); e = np.searchsorted(ts, np.datetime64(we, "ns").astype(np.int64))
        s = max(s, i0); e = min(e, i1 + 1)
        if e - s < 3: return
        pre = slice(i0, s)
        ph = d.high.values[pre].max() if s > i0 else np.nan; pl = d.low.values[pre].min() if s > i0 else np.nan
        fl = {}
        for f in FLAT:
            if f == "sess": fl[f] = i1 + 1
            else:
                add_m = 0 if f == "w0" else 30
                fl[f] = min(np.searchsorted(ts, np.datetime64(we + pd.Timedelta(minutes=add_m), "ns").astype(np.int64)), i1 + 1)
                fl[f] = max(fl[f], e)
        rows.append(dict(win=win, tdate=tdate, s=s, e=e, fw0=fl["w0"], fw30=fl["w30"], fsess=fl["sess"],
                         pre_hi=ph, pre_lo=pl, ref_hi=ref_hi, ref_lo=ref_lo, bias=bias))
    for k, D in enumerate(days):
        r = day.loc[D]; D = pd.Timestamp(D)
        pn = nights[nights < D]; pdays = days[:k]
        rh, rl = (ngt.loc[pn[-1]].hi, ngt.loc[pn[-1]].lo) if len(pn) and (D - pd.Timestamp(pn[-1])).days <= 5 else (np.nan, np.nan)
        bias = day.loc[pdays[-1]].cl if len(pdays) else np.nan
        add("open", D, r.i0, r.i1, D + pd.Timedelta("8h45min"), D + pd.Timedelta("9h45min"), rh, rl, bias)
        add("pm", D, r.i0, r.i1, D + pd.Timedelta("12h30min"), D + pd.Timedelta("13h30min"), rh, rl, bias)
    for N in nights:
        r = ngt.loc[N]; N = pd.Timestamp(N)
        ws = pd.Timestamp(N.year, N.month, N.day, 9, 30, tzinfo=ny).astimezone(tp).replace(tzinfo=None)
        nd = days[days > N]; td = pd.Timestamp(nd[0]) if len(nd) else N + pd.Timedelta(days=1)
        rh, rl, bias = (day.loc[N].hi, day.loc[N].lo, day.loc[N].cl) if N in day.index else (np.nan, np.nan, np.nan)
        add("night", td, r.i0, r.i1, ws, ws + pd.Timedelta(hours=1), rh, rl, bias)
    return pd.DataFrame(rows)


@njit(cache=True)
def manage(o, h, l, c, atr, j0, fe, dr, ep, stop, tgt, trail, risk):
    """從 j0 下一根開始管理部位（j0 為成交 K，已檢查過停損）。回傳 exit_px, kind(1 stop 2 tgt 3 flat), amb, exit_j。"""
    hh = h[j0] if dr > 0 else l[j0]
    armed = False
    for j in range(j0 + 1, fe):
        if dr > 0:
            hs = l[j] <= stop; ht = (not trail) and h[j] >= tgt
            if hs:
                px = min(o[j], stop) - 1.0
                return px, 1, 1 if ht else 0, j
            if ht:
                return max(o[j], tgt), 2, 0, j
            hh = max(hh, h[j])
            if trail:
                if hh - ep >= 1.5 * risk: armed = True
                if armed: stop = max(stop, hh - 2 * atr[j])
        else:
            hs = h[j] >= stop; ht = (not trail) and l[j] <= tgt
            if hs:
                px = max(o[j], stop) + 1.0
                return px, 1, 1 if ht else 0, j
            if ht:
                return min(o[j], tgt), 2, 0, j
            hh = min(hh, l[j])
            if trail:
                if ep - hh >= 1.5 * risk: armed = True
                if armed: stop = min(stop, hh + 2 * atr[j])
    return c[fe - 1] - dr * 1.0, 3, 0, fe - 1


@njit(cache=True)
def run(o, h, l, c, atr, S, E, FE, PH, PL, RH, RL, BI, minf, edge, R, sweep, bias, touch):
    n = len(S)
    out = np.full((n, 8), np.nan)  # dir, ep, stop, exit, kind, amb, risk, entry_j
    trail = R == 0.0
    for k in range(n):
        s = S[k]; we = E[k]; fe = FE[k]
        rh = RH[k]; rl = RL[k]
        sw_hi = (PH[k] > rh) if not (np.isnan(PH[k]) or np.isnan(rh)) else False
        sw_lo = (PL[k] < rl) if not (np.isnan(PL[k]) or np.isnan(rl)) else False
        pend = False; dr = 0; ep = 0.0; stp = 0.0; tg = 0.0; rk = 0.0
        for j in range(s, we):
            if pend:
                f = (l[j] <= ep - 1.0) if dr > 0 else (h[j] >= ep + 1.0)
                if touch: f = (l[j] <= ep) if dr > 0 else (h[j] >= ep)
                if f:
                    epx = min(ep, o[j]) if dr > 0 else max(ep, o[j])  # 開盤跳過限價則以開盤價（較佳）成交
                    if (dr > 0 and l[j] <= stp) or (dr < 0 and h[j] >= stp):
                        xp = (stp - 1.0) if dr > 0 else (stp + 1.0)
                        out[k, 0] = dr; out[k, 1] = epx; out[k, 2] = stp; out[k, 3] = xp; out[k, 4] = 1; out[k, 5] = 0; out[k, 6] = rk; out[k, 7] = j
                    else:
                        xp, kind, amb, xj = manage(o, h, l, c, atr, j, fe, dr, epx, stp, tg, trail, rk)
                        out[k, 0] = dr; out[k, 1] = epx; out[k, 2] = stp; out[k, 3] = xp; out[k, 4] = kind; out[k, 5] = amb; out[k, 6] = rk; out[k, 7] = j
                    break
            if not np.isnan(rh) and h[j] > rh: sw_hi = True
            if not np.isnan(rl) and l[j] < rl: sw_lo = True
            if j >= s + 2:
                bull = l[j] > h[j - 2] and (l[j] - h[j - 2]) >= minf
                bear = h[j] < l[j - 2] and (l[j - 2] - h[j]) >= minf
                if bull and sweep and not sw_lo: bull = False
                if bear and sweep and not sw_hi: bear = False
                if bias and not np.isnan(BI[k]):
                    if bull and not c[j] > BI[k]: bull = False
                    if bear and not c[j] < BI[k]: bear = False
                if bias and np.isnan(BI[k]): bull = False; bear = False
                if bull:
                    dr = 1; ep = l[j] if edge else math.floor((l[j] + h[j - 2]) / 2); stp = l[j - 2] - 1.0
                    rk = ep - stp; tg = ep + R * rk; pend = rk > 0
                elif bear:
                    dr = -1; ep = h[j] if edge else math.ceil((h[j] + l[j - 2]) / 2); stp = h[j - 2] + 1.0
                    rk = stp - ep; tg = ep - R * rk; pend = rk > 0
    return out


@njit(cache=True)
def run_random(o, h, l, c, atr, S, E, FE, RB, DR, RK, R):
    n = len(S); out = np.full((n, 8), np.nan); trail = R == 0.0
    for k in range(n):
        j = RB[k]
        if j < 0: continue
        dr = DR[k]; ep = o[j] + dr * 1.0; rk = RK[k]
        stp = ep - dr * rk; tg = ep + dr * R * rk
        if (dr > 0 and l[j] <= stp) or (dr < 0 and h[j] >= stp):
            out[k, 0] = dr; out[k, 1] = ep; out[k, 3] = stp - dr; out[k, 4] = 1; out[k, 6] = rk; continue
        xp, kind, amb, xj = manage(o, h, l, c, atr, j, FE[k], dr, ep, stp, tg, trail, rk)
        out[k, 0] = dr; out[k, 1] = ep; out[k, 3] = xp; out[k, 4] = kind; out[k, 5] = amb; out[k, 6] = rk
    return out


def trades_df(out, I):
    t = pd.DataFrame(out, columns=["dir", "ep", "stop", "xp", "kind", "amb", "risk", "ej"]); t["tdate"] = I.tdate.values; t["win"] = I.win.values
    t = t.dropna(subset=["dir"]).copy()
    t["gross"] = t.dir * (t.xp - t.ep); t["cost"] = 0.5 + 2e-5 * (t.ep + t.xp)
    t["net"] = t.gross - t.cost; t["R"] = t.net / t.risk
    return t


def metrics(t, alld, lo=None, hi=None):
    if lo is not None:
        t = t[(t.tdate >= lo) & (t.tdate < hi)]; alld = alld[(alld >= lo) & (alld < hi)]
    dp = t.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV
    n = len(t); yrs = len(alld) / 252
    if n == 0: return dict(trades=0, Sharpe=np.nan)
    w = t.net[t.net > 0].sum(); ls = -t.net[t.net < 0].sum()
    eq = dp.cumsum()
    return dict(trades=n, win=(t.net > 0).mean(), avgR=t.R.mean(), PF=w / ls if ls > 0 else np.nan, net_pts=t.net.sum(),
                net_twd=t.net.sum() * PV, ann_ret=dp.sum() / yrs / CAP,
                Sharpe=dp.mean() / dp.std() * math.sqrt(252) if dp.std() > 0 else np.nan, MDD_twd=(eq - eq.cummax()).min(),
                amb=int(t.amb.sum()))


def main():
    OUT.mkdir(exist_ok=True)
    d = load_bars(); I = build_instances(d)
    o, h, l, c, atr = (d[k].values for k in ("open", "high", "low", "close", "atr"))
    alld = np.array(sorted(I.tdate.unique())); alld = pd.DatetimeIndex(alld)
    print("bars", len(d), "instances", I.groupby("win").size().to_dict(), alld[0].date(), alld[-1].date())
    def sim(cfg, I_=None, arr=None, touch=False):
        I_ = I if I_ is None else I_; o_, h_, l_, c_, a_ = arr if arr is not None else (o, h, l, c, atr)
        J = I_ if cfg["win"] == "all" else I_[I_.win == cfg["win"]]
        out = run(o_, h_, l_, c_, a_, J.s.values, J.e.values, J["f" + cfg["flat"]].values, J.pre_hi.values, J.pre_lo.values,
                  J.ref_hi.values, J.ref_lo.values, J.bias.values, float(cfg["minf"]), cfg["entry"] == "edge", float(cfg["R"]),
                  bool(cfg["sweep"]), bool(cfg["bias"]), touch)
        return trades_df(out, J)
    rows = []; T = {}
    keys = list(itertools.product(WINS + ("all",), MINF, ENTRY, TGT, (0, 1), (0, 1), FLAT))
    for i, (w, mf, en, R, sw, bi, fl) in enumerate(keys):
        cfg = dict(win=w, minf=mf, entry=en, R=R, sweep=sw, bias=bi, flat=fl)
        t = sim(cfg); T[i] = t
        r = dict(id=i, **cfg)
        for pn, lo, hi in (("IS", alld[0], IS_END), ("OOS", IS_END, alld[-1] + pd.Timedelta(days=1))):
            r.update({f"{pn}_{k}": v for k, v in metrics(t, alld, lo, hi).items()})
        rows.append(r)
    G = pd.DataFrame(rows); G.to_csv(OUT / "sb_grid.csv", index=False)
    print("configs", len(G))
    ok = G[G.IS_trades >= 100].copy()
    ok["IS_rank"] = ok.IS_Sharpe.rank(ascending=False); ok["OOS_rank"] = ok.OOS_Sharpe.rank(ascending=False)
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    cols = ["win", "minf", "entry", "R", "sweep", "bias", "flat", "IS_trades", "IS_win", "IS_avgR", "IS_PF", "IS_ann_ret", "IS_Sharpe",
            "OOS_trades", "OOS_win", "OOS_avgR", "OOS_PF", "OOS_net_twd", "OOS_ann_ret", "OOS_Sharpe", "OOS_MDD_twd", "OOS_rank"]
    top = ok.sort_values("IS_Sharpe", ascending=False)
    print("\n== IS top10 =="); print(top.head(10)[cols].round(3).to_string(index=False))
    print("\nSpearman IS/OOS Sharpe (IS trades>=100, N=%d): %.3f" % (len(ok), ok.IS_Sharpe.rank().corr(ok.OOS_Sharpe.rank())))
    for w in WINS + ("all",):
        s = ok[ok.win == w]; print(f"  {w}: N={len(s)} spearman {s.IS_Sharpe.rank().corr(s.OOS_Sharpe.rank()):.3f}  IS>0 {(s.IS_Sharpe>0).mean():.2f}  OOS>0 {(s.OOS_Sharpe>0).mean():.2f}  mean IS {s.IS_Sharpe.mean():.3f} OOS {s.OOS_Sharpe.mean():.3f}")
    print("\nfraction with Sharpe>0: IS %.3f OOS %.3f" % ((ok.IS_Sharpe > 0).mean(), (ok.OOS_Sharpe > 0).mean()))
    for k in ("win", "minf", "entry", "R", "sweep", "bias", "flat"):
        print(ok.groupby(k)[["IS_Sharpe", "OOS_Sharpe", "IS_avgR", "OOS_avgR"]].mean().round(3).to_string()); print()

    best = top.iloc[0]; bcfg = {k: best[k] for k in ("win", "minf", "entry", "R", "sweep", "bias", "flat")}
    bt = T[int(best.id)]
    print("BEST", bcfg)
    full = (alld[0], alld[-1] + pd.Timedelta(days=1))
    per = [("IS", alld[0], IS_END), ("OOS", IS_END, full[1]), ("FULL", *full)]
    def show(t, lab):
        print(lab, {pn: {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics(t, alld, lo, hi).items()} for pn, lo, hi in per})
    show(bt, "best conservative")
    tt = sim(bcfg, touch=True); show(tt, "best touch-fill")
    # 每窗口 / 每年
    bt["year"] = bt.tdate.dt.year
    print("\nper window (best params, each window alone):")
    for w in WINS:
        c2 = dict(bcfg, win=w); tw = sim(c2); print(" ", w, {pn: {k: round(v, 3) for k, v in metrics(tw, alld, lo, hi).items() if k in ("trades", "win", "avgR", "PF", "net_twd", "Sharpe")} for pn, lo, hi in per})
    yr = bt.groupby("year").agg(trades=("net", "size"), win=("net", lambda x: (x > 0).mean()), avgR=("R", "mean"), net_pts=("net", "sum"))
    yr["net_twd"] = yr.net_pts * PV; yr["ret_on_1M"] = yr.net_twd / CAP; print("\nper year (best):"); print(yr.round(3).to_string())
    # tick 解析同 K 停損/停利
    amb = bt[bt.amb == 1]
    print(f"\nambiguous stop+target bars: {len(amb)} of {len(bt)} trades")
    tcfg, tb = bcfg, bt
    if len(amb) == 0:  # 最佳參數無同 K 衝突 → 改以同參數 R=1 抽樣驗證「同 K 算停損」的保守程度
        tcfg = dict(bcfg, R=1.0); tb = sim(tcfg); amb = tb[tb.amb == 1]
        print(f"  sample check on R=1 variant: {len(amb)} ambiguous of {len(tb)}")
    res_tick = tick_resolve(d, amb.sample(min(150, len(amb)), random_state=1), tcfg) if len(amb) else None
    if res_tick is not None and len(res_tick):
        print("tick-resolved sample:", res_tick.result.value_counts().to_dict())
        adj = tb.copy()
        for ix, r in res_tick[res_tick.result == "target_first"].iterrows():
            row = adj.loc[ix]; xp = row.ep + row.dir * tcfg["R"] * row.risk
            adj.loc[ix, "net"] = row.dir * (xp - row.ep) - (0.5 + 2e-5 * (row.ep + xp)); adj.loc[ix, "R"] = adj.loc[ix, "net"] / row.risk
        show(tb, "tick-check variant (bar rule)"); show(adj, "tick-check variant with tick-resolved sample")
    # 隨機進場基準
    rng = np.random.default_rng(7); J = I if bcfg["win"] == "all" else I[I.win == bcfg["win"]]
    Jt = J.reset_index(drop=True); bt_idx = bt.copy(); bt_idx["k"] = np.searchsorted(Jt.s.values, J.loc[bt.index, "s"].values) if False else None
    trade_risk = bt.risk.values
    rnd_stats = []
    for rep in range(200):
        RB = np.array([rng.integers(s + 3, e) if e - s > 3 else -1 for s, e in zip(Jt.s.values, Jt.e.values)])
        traded = np.zeros(len(Jt), bool); traded[rng.choice(len(Jt), size=min(len(bt), len(Jt)), replace=False)] = True
        RB = np.where(traded, RB, -1)
        DR = rng.choice([-1, 1], size=len(Jt)); RK = rng.choice(trade_risk, size=len(Jt))
        out = run_random(o, h, l, c, atr, Jt.s.values, Jt.e.values, Jt["f" + bcfg["flat"]].values, RB, DR, RK.astype(float), float(bcfg["R"]))
        tr = trades_df(out, Jt)
        rnd_stats.append({pn: metrics(tr, alld, lo, hi) for pn, lo, hi in per})
    for pn in ("IS", "OOS", "FULL"):
        sh = np.array([r[pn]["Sharpe"] for r in rnd_stats]); ar = np.array([r[pn]["avgR"] for r in rnd_stats])
        real = metrics(bt, alld, *[x for x in per if x[0] == pn][0][1:])
        print(f"random {pn}: Sharpe mean {sh.mean():.3f} [5%,95%] [{np.percentile(sh,5):.3f},{np.percentile(sh,95):.3f}] | avgR mean {ar.mean():.3f} | real Sharpe {real['Sharpe']:.3f} pctile {(sh < real['Sharpe']).mean():.2f}")
    # 3/5 分 K 穩健度
    for rule in ("3min", "5min"):
        dd = load_bars(rule); II = build_instances(dd)
        arr = tuple(dd[k].values for k in ("open", "high", "low", "close", "atr"))
        tr = sim(bcfg, II, arr); show(tr, f"best on {rule}")
    # walk-forward
    years = sorted(set(alld.year)); wf = []; picks = []
    daily = {i: T[i].groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV for i in G.id}
    DP = pd.DataFrame(daily)
    for Y in range(2020, years[-1] + 1):
        tr = DP.loc[f"{Y-3}-01-01":f"{Y-1}-12-31"]
        cnt = pd.Series({i: ((T[i].tdate >= f"{Y-3}-01-01") & (T[i].tdate < f"{Y}-01-01")).sum() for i in G.id})
        sh = tr.mean() / tr.std() * math.sqrt(252); sh[cnt < 60] = np.nan; b = sh.idxmax()
        wf.append(DP.loc[f"{Y}-01-01":f"{Y}-12-31", b]); picks.append(dict(year=Y, **G.loc[b, ["win", "minf", "entry", "R", "sweep", "bias", "flat"]].to_dict(), train_sharpe=round(sh.max(), 3),
                  test_sharpe=round(DP.loc[f"{Y}-01-01":f"{Y}-12-31", b].mean() / DP.loc[f"{Y}-01-01":f"{Y}-12-31", b].std() * math.sqrt(252), 3),
                  test_twd=DP.loc[f"{Y}-01-01":f"{Y}-12-31", b].sum()))
    wfr = pd.concat(wf); P = pd.DataFrame(picks); print("\n== walk-forward (trailing 3y) =="); print(P.to_string(index=False))
    print("WF stitched: net TWD %.0f  ann %.2f%%  Sharpe %.3f  MDD %.0f" % (wfr.sum(), wfr.sum() / (len(wfr) / 252) / CAP * 100, wfr.mean() / wfr.std() * math.sqrt(252), (wfr.cumsum() - wfr.cumsum().cummax()).min()))
    bt.to_csv(OUT / "sb_best_trades.csv", index=False); P.to_csv(OUT / "sb_wf_picks.csv", index=False)
    chart(G, ok, bt, alld, wfr, DP, int(best.id), tt)


def tick_resolve(d, amb, cfg):
    try:
        from tools.tickdata import load_ticks
    except Exception as ex:
        print("tick load failed", ex); return None
    res = []
    for ix, r in amb.iterrows():
        # 找出成交後第一根同時碰停損與停利的 K
        j = int(r.ej); risk = r.risk; dr = r.dir; tgt = r.ep + dr * cfg["R"] * risk; stp = r.stop
        jj = j + 1
        while jj < len(d) and not ((d.low.values[jj] <= stp and d.high.values[jj] >= tgt) if dr > 0 else (d.high.values[jj] >= stp and d.low.values[jj] <= tgt)):
            jj += 1
        t0 = d.ts.values[jj]; t0 = pd.Timestamp(t0)
        try:
            tk = load_ticks("TX", r.tdate.strftime("%Y-%m-%d"), r.tdate.strftime("%Y-%m-%d"), contract="near")
        except Exception as ex:
            res.append(dict(idx=ix, result="no_tick")); continue
        tk = tk[(tk.ts.dt.tz_localize(None) >= t0) & (tk.ts.dt.tz_localize(None) < t0 + pd.Timedelta(minutes=1))] if tk.ts.dt.tz is not None else tk[(tk.ts >= t0) & (tk.ts < t0 + pd.Timedelta(minutes=1))]
        p = tk.price.values
        hs = np.where((p <= stp) if dr > 0 else (p >= stp))[0]; ht = np.where((p >= tgt) if dr > 0 else (p <= tgt))[0]
        if len(p) == 0: out = "no_tick"
        elif len(ht) and (not len(hs) or ht[0] < hs[0]): out = "target_first"
        elif len(hs): out = "stop_first"
        else: out = "neither"
        res.append(dict(idx=ix, result=out))
    return pd.DataFrame(res, columns=["idx", "result"]).set_index("idx")


def chart(G, ok, bt, alld, wfr, DP, bid, tt):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK TC"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 4, figsize=(22, 9))
    for j, w in enumerate(WINS + ("all",)):
        s = ok[ok.win == w]
        for i, k in enumerate(("IS_Sharpe", "OOS_Sharpe")):
            a = ax[i, j]; P = s.pivot_table(index="minf", columns="R", values=k, aggfunc="mean")
            a.imshow(P.values, cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
            for (y, x), v in np.ndenumerate(P.values): a.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=9)
            a.set_xticks(range(len(P.columns))); a.set_xticklabels(["trail" if c == 0 else c for c in P.columns]); a.set_yticks(range(len(P.index))); a.set_yticklabels(P.index)
            a.set_title(f"{w} {k}（平均）"); a.set_xlabel("R 目標"); a.set_ylabel("最小 FVG 點")
    fig.tight_layout(); fig.savefig(OUT / "sb_heatmap.png", dpi=100)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    a = ax[0]
    a.plot(alld, (bt.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV).cumsum() / 1e4, label="IS 最佳（保守成交）")
    a.plot(alld, (tt.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV).cumsum() / 1e4, label="同參數 touch 成交", ls="--")
    w0 = wfr.cumsum() / 1e4; a.plot(w0.index, w0.values, label="walk-forward（3 年滾動）")
    a.axvline(IS_END, c="gray", ls="--"); a.set_ylabel("累積淨損益（萬元，1 口 TXF）"); a.legend(); a.grid(alpha=.3); a.set_title("ICT Silver Bullet 台指期 1 分 K")
    a = ax[1]; a.scatter(ok.IS_Sharpe, ok.OOS_Sharpe, s=6, c=ok.win.map({"open": 0, "pm": 1, "night": 2, "all": 3}), cmap="tab10")
    a.axhline(0, c="gray"); a.axvline(0, c="gray"); a.set_xlabel("IS Sharpe 2017-21"); a.set_ylabel("OOS Sharpe 2022+"); a.grid(alpha=.3); a.set_title("各參數 IS vs OOS Sharpe")
    fig.tight_layout(); fig.savefig(OUT / "sb_equity.png", dpi=100)
    print("charts results/sb_heatmap.png results/sb_equity.png")


if __name__ == "__main__":
    main()
