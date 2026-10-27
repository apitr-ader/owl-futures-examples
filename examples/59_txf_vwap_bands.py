"""59 Session-anchored VWAP ±kσ 通道（均值回歸 / 突破）台指期 1 分 K 回測 + 樣本內最佳化。

參考：TradingView「VWAP Band Mean Reversion」(CNC6abWu, ortizbruno115)；research/tradingview_strategies.md 第二輪 G。
該頁面只有一句話描述：價格伸出 session VWAP 的標準差通道後收回，往 VWAP 方向進場；ADX 濾網擋掉強趨勢日；收盤前平倉；建議 5 分 K。
沒有公開參數與績效。本檔把它擴成「回歸」與「突破」兩族（ADX 濾網未實作，以停損 / 時間出場取代）。框架、資料、成本、成交慣例同 55/58。
VWAP：以 session 起點錨定（日盤 08:45 起；夜盤 15:00 起，兩者各自獨立測試），典型價 tp=(H+L+C)/3 以 1 分 K 成交量加權累積；
  σ = 成交量加權標準差 sqrt(Σv·tp²/Σv − VWAP²)；上下軌 = VWAP ± k×σ（k = 1.5 / 2 / 2.5）。每根 K 收盤後（含該根）更新。
進場（訊號在 K 收盤，下一根開盤成交 + 1 點滑價；session 前 30 分不進場；日盤 13:00 / 夜盤 04:00 以後不開新倉；同時 1 部位；每 session 最多 3 筆）：
  mr_touch ：K 棒觸及上（下）軌但收盤回到軌內 → 做空（多）。
  mr_in    ：前一根收盤在軌外、本根收盤回到軌內 → 做空（多）。
  brk_close：首次收盤突破上（下）軌（前一根收盤在軌內）→ 順勢做多（空）。
  brk_2    ：連續兩根收盤在軌外（第 2 根確認，前一根之前在軌內）→ 順勢。
停損：進場價 ∓ s×D，D = 前 20 個日盤平均高低差，s ∈ {0.1, 0.2, 0.4}（跨指數水位可比），至少 5 點。
停利：vwap = 訊號當下的 VWAP（靜態限價，僅回歸族）；opp = 訊號當下對側軌（僅回歸族）；1R / 2R = 停損距離倍數；none = 不設。
時間出場：進場後 60 / 180 分，或 session 結束（日盤收盤 / 夜盤 05:00）；一律 session 內強平（市價 1 點滑價），不跨 session，故無換月問題。
同根 K 同碰停損與停利 → 停損；成交當根碰停損 → 停損、成交當根不算停利；跳空越停損 → 開盤價 − 滑價；停利限價觸價成交（無滑價）。
成本：每邊 50 元 + 2e-5×價格 期交稅 + 市價 / 停損 1 點滑價；1 點 = 200 元；100 萬元/口固定資本（不複利）。MDD 盤中最壞 = 逐筆用 MAE 估。
IS 2017-06~2021；OOS 2022~；目標 IS 日損益 Sharpe（至少 150 筆）；walk-forward = 每年用前 3 年重選。日盤與夜盤各自獨立最佳化與報告。
不 commit 任何資料；從 workspace 根目錄執行：  .venv-pq/bin/python repo/examples/59_txf_vwap_bands.py
"""
import sys, math, itertools, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
def _imp(name, f):
    sp = importlib.util.spec_from_file_location(name, HERE / f); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m58 = _imp("ovp58", "58_txf_overnight_volume_profile.py")
load_bars, DATA, manage2, sharpe, dd_stats, metrics = m58.load_bars, m58.DATA, m58.manage2, m58.sharpe, m58.dd_stats, m58.metrics
OUT = Path("results"); CAP = 1_000_000; PV = 200.0; IS_END = pd.Timestamp("2022-01-01"); MIN_IS = 150
KS = (1.5, 2.0, 2.5); ETYPES = ("mr_touch", "mr_in", "brk_close", "brk_2"); STOPS = (0.1, 0.2, 0.4); TIMES = (60, 180, 0)
TGTS = {"mr": ("vwap", "opp", "1R", "2R", "none"), "brk": ("1R", "2R", "none")}
WARM = 30
COLS = ["dir", "ep", "stop", "xp", "kind", "amb", "risk", "mae", "ej", "xj", "tgd"]


@njit(cache=True)
def vwap_arrays(h, l, c, v, sid):
    n = len(h); VW = np.zeros(n); SD = np.zeros(n); sv = 0.0; sp = 0.0; sp2 = 0.0
    for i in range(n):
        if i == 0 or sid[i] != sid[i - 1]: sv = 0.0; sp = 0.0; sp2 = 0.0
        tp = (h[i] + l[i] + c[i]) / 3.0
        sv += v[i]; sp += v[i] * tp; sp2 += v[i] * tp * tp
        VW[i] = sp / sv; SD[i] = math.sqrt(max(sp2 / sv - VW[i] * VW[i], 0.0))
    return VW, SD


@njit(cache=True)
def run(o, h, l, c, VW, SD, S0, CE, FE, STP, kk, et, tgt, tm, T):
    """et: 0 mr_touch 1 mr_in 2 brk_close 3 brk_2；tgt: 0 vwap 1 opp 2 R 3 none；tm = R 倍數。"""
    n = len(S0); out = np.full((n, 3, 11), np.nan)
    for q in range(n):
        j = S0[q] + WARM; ns = 0; S = STP[q]
        while j < CE[q] - 1 and ns < 3:
            if SD[j] <= 0 or SD[j - 1] <= 0 or SD[j - 2] <= 0: j += 1; continue
            up = VW[j] + kk * SD[j]; lo = VW[j] - kk * SD[j]
            upp = VW[j - 1] + kk * SD[j - 1]; lop = VW[j - 1] - kk * SD[j - 1]
            up2 = VW[j - 2] + kk * SD[j - 2]; lo2 = VW[j - 2] - kk * SD[j - 2]
            sig = 0
            if et == 0:
                if h[j] >= up and c[j] < up: sig = -1
                elif l[j] <= lo and c[j] > lo: sig = 1
            elif et == 1:
                if c[j - 1] > upp and c[j] <= up: sig = -1
                elif c[j - 1] < lop and c[j] >= lo: sig = 1
            elif et == 2:
                if c[j] > up and c[j - 1] <= upp: sig = 1
                elif c[j] < lo and c[j - 1] >= lop: sig = -1
            else:
                if c[j] > up and c[j - 1] > upp and c[j - 2] <= up2: sig = 1
                elif c[j] < lo and c[j - 1] < lop and c[j - 2] >= lo2: sig = -1
            if sig == 0: j += 1; continue
            ej = j + 1; dr = float(sig); ep = o[ej] + dr; stp = ep - dr * S
            if tgt == 0: tg = VW[j]
            elif tgt == 1: tg = lo if sig < 0 else up
            elif tgt == 2: tg = ep + dr * tm * S
            else: tg = dr * 1e9
            if tgt < 3 and dr * (tg - ep) <= 0: j += 1; continue  # 目標已在進場價後方：不做
            fe = min(ej + T, FE[q]) if T > 0 else FE[q]
            if fe <= ej + 1: fe = ej + 1
            if (dr > 0 and l[ej] <= stp) or (dr < 0 and h[ej] >= stp):
                xp = stp - dr; kind = 1; amb = 0; xj = ej; mae = (ep - l[ej]) if dr > 0 else (h[ej] - ep)
            else:
                xp, kind, amb, xj, mae = manage2(o, h, l, c, ej, fe, dr, ep, stp, tg)
            mae = max(mae, dr * (ep - xp))
            out[q, ns, 0] = dr; out[q, ns, 1] = ep; out[q, ns, 2] = stp; out[q, ns, 3] = xp; out[q, ns, 4] = kind; out[q, ns, 5] = amb
            out[q, ns, 6] = S; out[q, ns, 7] = mae; out[q, ns, 8] = ej; out[q, ns, 9] = xj; out[q, ns, 10] = abs(tg - ep) if tgt < 3 else 0.0
            ns += 1; j = xj + 1
    return out


@njit(cache=True)
def run_random(o, h, l, c, RB, FE, DR, STP, TGD, T):
    n = len(RB); out = np.full((n, 11), np.nan)
    for q in range(n):
        ej = RB[q]
        if ej < 0: continue
        dr = DR[q]; S = STP[q]; ep = o[ej] + dr; stp = ep - dr * S; tg = ep + dr * TGD[q] if TGD[q] > 0 else dr * 1e9
        fe = min(ej + T, FE[q]) if T > 0 else FE[q]
        if fe <= ej + 1: fe = ej + 1
        if (dr > 0 and l[ej] <= stp) or (dr < 0 and h[ej] >= stp):
            xp = stp - dr; kind = 1; amb = 0; xj = ej; mae = (ep - l[ej]) if dr > 0 else (h[ej] - ep)
        else:
            xp, kind, amb, xj, mae = manage2(o, h, l, c, ej, fe, dr, ep, stp, tg)
        mae = max(mae, dr * (ep - xp))
        out[q, 0] = dr; out[q, 1] = ep; out[q, 2] = stp; out[q, 3] = xp; out[q, 4] = kind; out[q, 5] = amb
        out[q, 6] = S; out[q, 7] = mae; out[q, 8] = ej; out[q, 9] = xj; out[q, 10] = TGD[q]
    return out


def to_trades(out, I):
    k = out.shape[0]
    if out.ndim == 3: rep = np.repeat(np.arange(k), out.shape[1]); out = out.reshape(-1, 11)
    else: rep = np.arange(k)
    t = pd.DataFrame(out, columns=COLS); t["di"] = rep; t["tdate"] = I.tdate.values[rep]
    t = t.dropna(subset=["dir"]).copy()
    t["cost"] = 0.5 + 2e-5 * (t.ep + t.xp); t["net"] = t.dir * (t.xp - t.ep) - t.cost; t["R"] = t.net / t.risk
    return t.sort_values(["di", "ej"]).reset_index(drop=True)


def build(d, sess_type):
    ts = d.ts.values
    s = d.groupby(["skey", "night"]).agg(i0=("ts", lambda x: x.index[0]), i1=("ts", lambda x: x.index[-1]), n=("ts", "size"),
                                         hi=("high", "max"), lo=("low", "min")).reset_index()
    day = s[~s.night].set_index("skey"); ngt = s[s.night].set_index("skey"); days = day.index.values
    D20 = (day.hi - day.lo).shift(1).rolling(20, min_periods=5).mean(); rows = []
    if sess_type == "day":
        for Dd in days:
            r = day.loc[Dd]; t0 = pd.Timestamp(Dd) + pd.Timedelta("8h45min")
            if r.n < 200 or d.ts.iloc[r.i0] > t0 + pd.Timedelta(minutes=15) or np.isnan(D20.loc[Dd]): continue
            cut = np.searchsorted(ts[r.i0:r.i1 + 1], np.datetime64(pd.Timestamp(Dd) + pd.Timedelta("13h")))
            if cut < 100: continue
            rows.append(dict(tdate=pd.Timestamp(Dd), s=r.i0, ce=r.i0 + cut, fe=r.i1 + 1, D=D20.loc[Dd]))
    else:
        for N0 in ngt.index:
            r = ngt.loc[N0]; nd = days[days > N0]
            if not len(nd) or r.n < 300 or (pd.Timestamp(nd[0]) - pd.Timestamp(N0)).days > 5 or np.isnan(D20.loc[nd[0]]): continue
            if d.ts.iloc[r.i0] > pd.Timestamp(N0) + pd.Timedelta("15h15min"): continue
            cut = np.searchsorted(ts[r.i0:r.i1 + 1], np.datetime64(pd.Timestamp(N0) + pd.Timedelta("1D4h")))
            if cut < 100: continue
            rows.append(dict(tdate=pd.Timestamp(nd[0]), s=r.i0, ce=r.i0 + cut, fe=r.i1 + 1, D=D20.loc[nd[0]]))
    return pd.DataFrame(rows).sort_values("tdate").reset_index(drop=True)


def param_list():
    P = []
    for k in KS:
        for e in ETYPES:
            fam = "mr" if e.startswith("mr") else "brk"
            for tg in TGTS[fam]:
                for st in STOPS:
                    for T in TIMES: P.append((k, e, tg, st, T))
    return P


def label(p): return f"k{p[0]}|{p[1]}|tgt {p[2]}|stop{p[3]}D|time {'close' if p[4]==0 else str(p[4])+'m'}"


def cfg_run(arr, VW, SD, I, p):
    k, e, tg, st, T = p; o, h, l, c = arr
    tgi = {"vwap": 0, "opp": 1, "1R": 2, "2R": 2, "none": 3}[tg]; tm = {"1R": 1.0, "2R": 2.0}.get(tg, 0.0)
    return run(o, h, l, c, VW, SD, I.s.values, I.ce.values, I.fe.values, (I.D.values * st).clip(min=5.0), k, ETYPES.index(e), tgi, tm, T)


def analyze(d, arr, VW, SD, sess):
    I = build(d, sess); alld = pd.DatetimeIndex(I.tdate); nd = len(alld); isd = alld < IS_END
    tag = "d" if sess == "day" else "n"; nm_cn = "日盤" if sess == "day" else "夜盤"
    print(f"\n################ {nm_cn} VWAP ################\nsessions {nd} {alld[0].date()}..{alld[-1].date()} | IS {isd.sum()} OOS {(~isd).sum()}")
    params = param_list(); rows = []; DP = np.zeros((len(params), nd), np.float32); CN = np.zeros((len(params), nd), np.int16); T = {}
    for i, p in enumerate(params):
        t = to_trades(cfg_run(arr, VW, SD, I, p), I); T[i] = t
        DP[i] = np.bincount(t.di.values, weights=t.net.values * PV, minlength=nd); CN[i] = np.bincount(t.di.values, minlength=nd)
        r = dict(id=i, k=p[0], etype=p[1], tgt=p[2], stop=p[3], time=p[4], fam="mr" if p[1].startswith("mr") else "brk")
        for nm, mk in (("IS", isd), ("OOS", ~isd)):
            tt = t[(t.tdate < IS_END) if nm == "IS" else (t.tdate >= IS_END)]; dv = DP[i][mk].astype(float); eq = dv.cumsum()
            w = tt.net[tt.net > 0].sum(); ls = -tt.net[tt.net < 0].sum()
            r.update({f"{nm}_trades": len(tt), f"{nm}_win": (tt.net > 0).mean() if len(tt) else np.nan, f"{nm}_avgR": tt.R.mean() if len(tt) else np.nan,
                      f"{nm}_PF": w / ls if ls > 0 else np.nan, f"{nm}_net_twd": tt.net.sum() * PV, f"{nm}_ann": dv.sum() / (len(dv) / 252) / CAP,
                      f"{nm}_Sharpe": sharpe(dv), f"{nm}_MDD": (eq - np.maximum.accumulate(eq)).min()})
        rows.append(r)
    G = pd.DataFrame(rows); G.to_csv(OUT / f"vwap_{tag}_grid.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    ok = G[G.IS_trades >= MIN_IS].copy(); ok["OOS_rank"] = ok.OOS_Sharpe.rank(ascending=False); top = ok.sort_values("IS_Sharpe", ascending=False)
    cols = ["k", "etype", "tgt", "stop", "time", "IS_trades", "IS_win", "IS_avgR", "IS_PF", "IS_ann", "IS_Sharpe", "OOS_trades", "OOS_win", "OOS_avgR", "OOS_PF", "OOS_ann", "OOS_Sharpe", "OOS_MDD", "OOS_rank"]
    print(f"configs {len(G)}, IS trades>={MIN_IS}: {len(ok)}"); print("== IS top10 =="); print(top.head(10)[cols].round(3).to_string(index=False))
    rho = ok.IS_Sharpe.rank().corr(ok.OOS_Sharpe.rank())
    print("\nSpearman IS/OOS %.3f | IS>0 %.3f OOS>0 %.3f both>0 %.3f | mean IS %.3f mean OOS %.3f" % (rho, (ok.IS_Sharpe > 0).mean(), (ok.OOS_Sharpe > 0).mean(), ((ok.IS_Sharpe > 0) & (ok.OOS_Sharpe > 0)).mean(), ok.IS_Sharpe.mean(), ok.OOS_Sharpe.mean()))
    for k in ("fam", "etype", "k", "tgt", "stop", "time"):
        g = ok.groupby(k).agg(N=("id", "size"), IS_Sh=("IS_Sharpe", "mean"), OOS_Sh=("OOS_Sharpe", "mean"), IS_avgR=("IS_avgR", "mean"), OOS_avgR=("OOS_avgR", "mean"), IS_pos=("IS_Sharpe", lambda x: (x > 0).mean()), OOS_pos=("OOS_Sharpe", lambda x: (x > 0).mean()))
        print(g.round(3).to_string()); print()
    b = top.iloc[0]; bi = int(b.id); bp = params[bi]; bt = T[bi]; print("BEST (IS Sharpe)", label(bp))
    per = {"IS": (alld[0], IS_END), "OOS": (IS_END, alld[-1] + pd.Timedelta(days=1)), "FULL": (alld[0], alld[-1] + pd.Timedelta(days=1))}
    res = pd.DataFrame({k: metrics(bt, alld, lo, hi, full=True) for k, (lo, hi) in per.items()}).T
    print(res.round(3).to_string()); res.to_csv(OUT / f"vwap_{tag}_summary.csv")
    bt["year"] = bt.tdate.dt.year
    yr = bt.groupby("year").agg(trades=("net", "size"), win=("net", lambda x: (x > 0).mean()), avgR=("R", "mean"), net_pts=("net", "sum")); yr["net_twd"] = yr.net_pts * PV
    print("\nper year (best):"); print(yr.round(3).to_string()); yr.to_csv(OUT / f"vwap_{tag}_best_yearly.csv")
    o_ = bt[bt.tdate >= IS_END]
    print("OOS ex top10 net_twd %.0f | OOS ex 2026 net_twd %.0f (n=%d) | exit kinds %s | long/short %s" % (
        (o_.net.sum() - o_.net.nlargest(10).sum()) * PV, o_[o_.tdate.dt.year != 2026].net.sum() * PV, len(o_[o_.tdate.dt.year != 2026]), bt.kind.value_counts().to_dict(), bt.dir.value_counts().to_dict()))
    # 鄰居
    dims = dict(k=KS, etype=ETYPES, tgt=("vwap", "opp", "1R", "2R", "none"), stop=STOPS, time=TIMES); nb = []
    for k, vals in dims.items():
        for v in vals:
            m = (G[k] == v)
            for x in dims:
                if x != k: m &= (G[x] == b[x])
            if m.sum(): q = G[m].iloc[0]; nb.append((k, v, int(q.IS_trades), round(q.IS_Sharpe, 3), round(q.OOS_Sharpe, 3), round(q.OOS_ann, 4)))
    print("\n== neighbours of best =="); print(pd.DataFrame(nb, columns=["dim", "value", "IS_trades", "IS_Sharpe", "OOS_Sharpe", "OOS_ann"]).to_string(index=False))
    # 隨機進場基準
    rs = np.random.default_rng(7); o, h, l, c = arr; STPb = (I.D.values * bp[3]).clip(min=5.0); nb_ = len(bt); stats = []; eqr = []; tpool = bt.tgd.values
    for rep in range(300):
        pick = rs.choice(nd, size=min(nb_, nd), replace=False); RB = np.full(nd, -1)
        RB[pick] = [rs.integers(I.s.values[q] + WARM, I.ce.values[q] - 1) for q in pick]
        tr = to_trades(run_random(o, h, l, c, RB, I.fe.values, rs.choice([-1.0, 1.0], size=nd), STPb, tpool[rs.integers(0, len(tpool), size=nd)], bp[4]), I)
        stats.append({pn: metrics(tr, alld, lo, hi) for pn, (lo, hi) in per.items()}); eqr.append(np.bincount(tr.di.values, weights=tr.net.values * PV, minlength=nd).cumsum())
    eqr = np.array(eqr); rb = []; print("\n== random-entry baseline (300 reps) ==")
    for pn in per:
        sh = np.array([s[pn]["Sharpe"] for s in stats]); real = res.loc[pn, "Sharpe"]; ar = np.mean([s[pn]["avgR"] for s in stats])
        rb.append(dict(period=pn, rand_Sharpe_mean=sh.mean(), p5=np.percentile(sh, 5), p95=np.percentile(sh, 95), rand_avgR=ar, real_Sharpe=real, real_avgR=res.loc[pn, "avgR"], real_pctile=(sh < real).mean()))
        print(f"{pn}: random Sharpe mean {sh.mean():.3f} [5%,95%] [{np.percentile(sh,5):.3f},{np.percentile(sh,95):.3f}] avgR {ar:.3f} | real {real:.3f} avgR {res.loc[pn,'avgR']:.3f} pctile {(sh < real).mean():.2f}")
    pd.DataFrame(rb).to_csv(OUT / f"vwap_{tag}_random_baseline.csv", index=False)
    # walk-forward
    picks = []; wf = []
    for Y in range(2020, alld.year.max() + 1):
        trm = (alld.year >= Y - 3) & (alld.year < Y); tsm = alld.year == Y
        X = DP[:, trm].astype(float); sd_ = X.std(1); sh = X.mean(1) / np.where(sd_ > 0, sd_, np.nan) * math.sqrt(252); sh[CN[:, trm].sum(1) < 90] = np.nan
        bb = int(np.nanargmax(sh)); y = DP[bb, tsm].astype(float); wf.append(pd.Series(y, index=alld[tsm]))
        picks.append(dict(year=Y, cfg=label(params[bb]), train_Sharpe=round(sh[bb], 3), test_Sharpe=round(sharpe(y), 3), test_twd=round(y.sum()), trades=int(CN[bb, tsm].sum())))
    wfr = pd.concat(wf); P = pd.DataFrame(picks); print("\n== walk-forward (trailing 3y) =="); print(P.to_string(index=False)); P.to_csv(OUT / f"vwap_{tag}_wf_picks.csv", index=False)
    ws = {}
    for nm, s in (("WF 2020+", wfr), ("WF 2022+", wfr[wfr.index >= IS_END])):
        eq = s.cumsum(); ws[nm] = dict(net_twd=s.sum(), ann=s.sum() / (len(s) / 252) / CAP, Sharpe=sharpe(s.values), MDD_close=(eq - eq.cummax().clip(lower=0)).min()); print(nm, {k: round(v, 3) for k, v in ws[nm].items()})
    pd.DataFrame(ws).T.to_csv(OUT / f"vwap_{tag}_wf_summary.csv"); bt.drop(columns=["year"]).to_csv(OUT / f"vwap_{tag}_best_trades.csv", index=False)
    return dict(tag=tag, nm=nm_cn, ok=ok, bt=bt, alld=alld, wfr=wfr, eqr=eqr, bp=bp, res=res, yr=yr)


def charts(R):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK TC"; plt.rcParams["axes.unicode_minus"] = False
    gold = "#C9A227"; bg = "#111111"; gr = "#8a8a8a"; wh = "#e6e6e6"
    def sty(ax):
        ax.set_facecolor(bg); ax.tick_params(colors=wh); ax.grid(alpha=.15, color=gr)
        for s in ax.spines.values(): s.set_color(gr)
        ax.xaxis.label.set_color(wh); ax.yaxis.label.set_color(wh); ax.title.set_color(wh)
    for r in R:
        alld = r["alld"]; bt = r["bt"]; eq = (bt.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV).cumsum() / 1e4
        fig, ax = plt.subplots(figsize=(12, 6.2), facecolor=bg); sty(ax); q5, q50, q95 = np.percentile(r["eqr"], [5, 50, 95], axis=0) / 1e4
        ax.fill_between(alld, q5, q95, color=gr, alpha=.35, label="隨機進場 5–95%（300 次）"); ax.plot(alld, q50, color=gr, lw=1, label="隨機進場 中位數")
        ax.plot(alld, eq, color=gold, lw=2, label="IS 最佳參數（2017-21 選定）"); ax.plot(r["wfr"].index, r["wfr"].cumsum().values / 1e4, color="#5fb0ff", lw=1.5, label="walk-forward（前 3 年重選，2020 起）")
        ax.axvline(IS_END, color="white", ls="--", lw=1); ax.text(IS_END, ax.get_ylim()[1], " OOS →", color="white", va="top")
        ax.set_ylabel("累積淨損益（萬元，1 口 TXF）"); ax.legend(facecolor=bg, labelcolor=wh, edgecolor=gr, loc="lower left", fontsize=9)
        i_, o_ = r["res"].loc["IS"], r["res"].loc["OOS"]
        ax.set_title(f"{r['nm']} VWAP ±kσ｜{label(r['bp'])}\nIS Sharpe {i_.Sharpe:.2f} → OOS Sharpe {o_.Sharpe:.2f}（OOS {int(o_.trades)} 筆，勝率 {o_.win*100:.0f}%，淨 {o_.net_twd/1e4:.0f} 萬）", fontsize=11, color=gold)
        fig.text(0.98, 0.01, "夜盤貓頭鷹｜含成本與滑價｜回測不代表未來績效", color=gold, ha="right", fontsize=9); fig.tight_layout(rect=[0, .03, 1, 1])
        fig.savefig(OUT / f"vwap_equity_{r['tag']}.png", dpi=110, facecolor=bg); plt.close(fig)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), facecolor=bg); cl = dict(zip(ETYPES, (gold, "#ff9f43", "#5fb0ff", "#7bd88f")))
    for j, r in enumerate(R):
        a = ax[0, j]; sty(a); ok = r["ok"]
        for e, g in ok.groupby("etype"): a.scatter(g.IS_Sharpe, g.OOS_Sharpe, s=6, color=cl[e], label=e, alpha=.7)
        a.axhline(0, c=gr); a.axvline(0, c=gr); a.set_xlabel("IS Sharpe 2017-21"); a.set_ylabel("OOS Sharpe 2022+")
        a.set_title(f"{r['nm']} 各參數 IS vs OOS（Spearman {ok.IS_Sharpe.rank().corr(ok.OOS_Sharpe.rank()):.2f}，N={len(ok)}）"); a.legend(facecolor=bg, labelcolor=wh, edgecolor=gr, fontsize=8)
        a = ax[1, j]; sty(a); yr = r["yr"]; a.bar(yr.index.astype(str), yr.net_twd / 1e4, color=[gold if v > 0 else "#d9534f" for v in yr.net_twd]); a.axhline(0, c=gr)
        a.set_title(f"{r['nm']} IS 最佳參數 各年度淨損益（萬元）"); a.set_ylabel("萬元")
    fig.tight_layout(); fig.savefig(OUT / "vwap_is_vs_oos.png", dpi=110, facecolor=bg); plt.close(fig)
    print("charts results/vwap_equity_d.png results/vwap_equity_n.png results/vwap_is_vs_oos.png")


def main():
    OUT.mkdir(exist_ok=True)
    d = load_bars(); vol = pd.read_parquet(DATA, columns=["date", "volume"]); vol["ts"] = vol.date.dt.tz_localize(None)
    d = d.merge(vol[["ts", "volume"]].drop_duplicates("ts"), on="ts", how="left"); d["volume"] = d.volume.fillna(1).astype(float)
    d = d.sort_values("ts").reset_index(drop=True)
    sid = (d.skey.astype("int64").values * 2 + d.night.astype(int).values)
    VW, SD = vwap_arrays(d.high.values, d.low.values, d.close.values, d.volume.values, sid)
    print("sanity: median σ/VWAP %.4f%%; close outside ±2σ (excl. first %d bars): %.3f" % (np.median(SD / VW) * 100, WARM, np.mean(np.abs(d.close.values - VW) > 2 * SD)))
    arr = tuple(d[k].values for k in ("open", "high", "low", "close"))
    R = [analyze(d, arr, VW, SD, s) for s in ("day", "night")]
    charts(R)


if __name__ == "__main__":
    main()
