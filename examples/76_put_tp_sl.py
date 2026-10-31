"""76 月選 OTM put 賣方：停利 × 兩倍點數停損（承 74/75）。

定義：入場收取權利金 C0（點；價差 = 賣腿 − 買腿，不含成本）。
  停損：部位虧損達 2×C0 點 ⇔ 部位價值（賣腿 − 買腿）≥ 3×C0。盤中判定。
  停利：已實現 p×C0（p∈{50%,65%,80%}）⇔ 價值 ≤ (1−p)×C0；或 none（持有至結算）。
  停損 on/off × 停利 4 種 = 8 變體；結構：裸賣 x3%/x4%/δ0.20，價差 (x3%,x4%,δ0.20)×寬度(2%,4%)，共 9 結構 72 變體。
時序：前一月選結算日 s 收盤選履約價（同 74 E+1）→ s+1 日開盤成交；出場後空手至下一月循環（預設）。
盤中判定：先用日 K（賣腿 day_high − 買腿 day_low 為最差、賣腿 low − 買腿 high 為最佳）篩出「可能觸發日」；
  該日再用 TAIFEX RPT 逐筆 → 1 分鐘最後成交價（前向填補，起點 = 前日收盤/入場價）求第一個觸發分鐘，
  於下一分鐘價格出場 + 每腿 1 tick 滑價 + 30 元 + 0.1% 稅（opt_cost_pts）。開盤跳空即以開盤後第一分鐘價成交。
  缺逐筆資料時退回日 K：停損以 max(門檻, 開盤) 成交、停利以 min(門檻, 開盤)；同日兩者皆可能 → 視為停損（保守）。
  只用日盤（08:45–13:45）；夜盤跳空在次日開盤才被看到。結算日不判定（同 74），結算價 ≈ TXF 收盤。
成本：每腿 30 元/邊、0.1%×權利金、1 tick；結算價內每腿 30 元 + 0.1%×內含值；1 點 = 50 元。
驗證：IS 2017–2021 選參、OOS 2022+；Spearman IS→OOS；逐年 WF（擴張窗自 2017）；隨機基準（隨機 config、隨機出場日）。
資金：100 萬起複利，口數 = floor(u×權益/單位保證金)，u∈{10%,25%}（保證金近似同 75）。
注意：價差在買腿入場日無報價時跳過該月（含 202003），價差樣本 96–103 筆 vs 裸賣 114 筆。
輸出：results/dd_put_tpsl_{grid,equity,yearly,wf,2020}.csv、dd_put_tpsl.png；快取 data/cache/（不入 repo）。
"""
import sys, pickle, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from _rqo import *
import importlib.util
spec = importlib.util.spec_from_file_location("m74", Path(__file__).parent / "74_txo_put_deepdive_grid.py"); m74 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m74)

END = m74.END; IS0 = pd.Timestamp("2017-01-01"); CAP0 = 1_000_000.0
CACHE = ROOT / "data/cache/dd_put_tpsl_trades.pkl"; MIN = ROOT / "data/cache/txo_1m_put"; MIN.mkdir(parents=True, exist_ok=True)
ZIP = str(ROOT / "data/options_daily/zip")
STRUCTS = [("x", .03, 0), ("x", .04, 0), ("d", .20, 0), ("x", .03, .02), ("x", .03, .04), ("x", .04, .02), ("x", .04, .04), ("d", .20, .02), ("d", .20, .04)]
TPS = [.5, .65, .8, None]; SLS = [2.0, None]
rng = np.random.default_rng(76)


def sname(s): return f"{'裸賣' if s[2] == 0 else '價差'} {s[0]}{s[1]:.2f}" + (f" w{s[2]:.0%}" if s[2] else "")


def build():
    F = load_fut("2016-06-01"); days = F.index[F.index <= END]
    E = expiries(); allm = E[E.kind == "monthly"].sort_values("expiry_date").reset_index(drop=True)
    M = allm[(allm.expiry_date <= END) & (allm.expiry_date >= IS0)]
    O0 = load_opt("2016-11-01", END, cp="P"); O0 = O0[O0.expiry.isin(M.expiry)]
    T = []
    for X, xd in zip(M.expiry, M.expiry_date):
        if xd not in days: continue
        g = O0[O0.expiry == X]
        pv = {f: g.pivot_table(index="date", columns="strike", values=f) for f in ("day_open", "day_high", "day_low", "day_close")}
        s = allm[allm.expiry_date < xd].expiry_date.iloc[-1]
        if s not in days or s not in pv["day_close"].index: continue
        si = days.get_loc(s); d1 = days[si + 1]; ix = days.get_loc(xd); hold = days[si + 1: ix + 1]
        if d1 not in pv["day_close"].index: continue
        Fs = F.close[s]; cl = pv["day_close"].loc[s].dropna(); cl = cl[(cl.index < Fs) & (cl > 0.5)]
        if len(cl) < 3: continue
        Tt = max((xd - s).days, 1) / 365
        iv, dl = m74.b76_put(cl.values, np.full(len(cl), Fs), cl.index.values.astype(float), np.full(len(cl), Tt)); ok = iv < 2.9
        Fc, Fl, Fh = F.close[hold].values, F.low[hold].values, F.high[hold].values
        px = lambda K: (pv["day_open"].at[d1, K] if not np.isnan(pv["day_open"].at[d1, K]) else pv["day_close"].at[d1, K]) if K in pv["day_open"].columns else np.nan
        for st in STRUCTS:
            if st[0] == "x": K = cl.index[np.abs(cl.index - Fs * (1 - st[1])).argmin()]
            else:
                if ok.sum() == 0: continue
                K = cl.index[ok][np.abs(np.abs(dl[ok]) - st[1]).argmin()]
            ps = px(K)
            if np.isnan(ps) or ps < 1: continue
            S = m74.leg_path(pv, K, hold, Fc, Fl, Fh, True); Kl, pl, L = None, 0.0, [np.zeros(len(hold))] * 4
            if st[2]:
                cand = pv["day_close"].columns[pv["day_close"].columns < K]
                if len(cand) == 0: continue
                Kl = cand[np.abs(cand - (K - st[2] * Fs)).argmin()]; pl = px(Kl)
                if np.isnan(pl): continue
                L = m74.leg_path(pv, Kl, hold, Fc, Fl, Fh, False)
            T.append(dict(st=st, X=X, s=s, d1=d1, hold=hold, Fs=Fs, K=K, Kl=Kl, ps=ps, pl=pl, S=S, L=L, Fc=Fc, Fl=Fl, w=st[2], C0=ps - pl))
    pickle.dump(T, open(CACHE, "wb")); return T


# ---------- 1 分鐘（逐筆 RPT → 分鐘最後成交；只存需要的 expiry/strike；不入 repo）----------
def _min_one(args):
    d, refs, want = args
    from tools.opt_convert import read_rpt, parse
    o = MIN / f"{d}.parquet"
    try:
        b, _, _ = read_rpt(refs); df, *_ = parse(b, d); del b
        if df is None: return d, "none"
        df = df[(~df.night) & (df.cp == "P")]; df["expiry"] = df.expiry.str.strip()
        df = df[(df.sec >= 8 * 3600 + 45 * 60) & (df.sec < 13 * 3600 + 45 * 60)]
        keep = np.zeros(len(df), bool)
        for X, ks in want.items(): keep |= ((df.expiry == X) & df.strike.isin(ks)).values
        x = df[keep]; x = x.assign(m=((x.sec - (8 * 3600 + 45 * 60)) // 60).astype(int))
        r = x.groupby(["expiry", "strike", "m"]).price.last().reset_index()
        mark = pd.DataFrame([(X, k, -1, np.nan) for X, ks in want.items() for k in ks], columns=["expiry", "strike", "m", "price"])
        r = pd.concat([r, mark]); r["strike"] = r.strike.astype(float)
        old = pd.read_parquet(o) if o.exists() else None
        r = pd.concat([old, r]).drop_duplicates(["expiry", "strike", "m"]) if old is not None else r
        r.to_parquet(o); return d, f"ok {len(r)}"
    except Exception as e:
        return d, f"err {e}"


_MC = {}
def minutes(d, X, K):
    """回傳長度 300 的分鐘最後成交（NaN=該分鐘無成交）或 None（無資料）。"""
    key = str(d.date())
    if key not in _MC:
        p = MIN / f"{key}.parquet"; _MC[key] = pd.read_parquet(p) if p.exists() else None
    g = _MC[key]
    if g is None: return None
    h = g[(g.expiry == X) & (g.strike == float(K))]
    if not (h.m == -1).any(): return None
    h = h[h.m >= 0]; a = np.full(300, np.nan); a[h.m.values.clip(0, 299)] = h.price.values; return a


def resolve(t, i, lvl_sl, lvl_tp, need):
    """第 i 日盤中判定。回傳 (kind, 出場價值, (賣腿價, 買腿價)) 或 None。need: 缺資料清單。"""
    d = t["hold"][i]; S, L = t["S"], t["L"]; hedged = t["w"] > 0
    ms = minutes(d, t["X"], t["K"]); ml = minutes(d, t["X"], t["Kl"]) if hedged else np.zeros(300)
    have = ms is not None and ml is not None
    if not have:
        need.setdefault(str(d.date()), {}).setdefault(t["X"], set()).update([t["K"]] + ([t["Kl"]] if hedged else []))
        vo = S[0][i] - L[0][i]; vw = min(S[1][i] - L[2][i], t["K"] - t["Kl"]) if hedged else S[1][i]; vb = max(S[2][i] - L[1][i], 0)
        if lvl_sl is not None and vw >= lvl_sl: v = max(lvl_sl, vo); return "stop", v, (S[0][i] if vo >= lvl_sl else S[0][i] + (v - vo), L[0][i])
        if lvl_tp is not None and vb <= lvl_tp: v = min(lvl_tp, vo); return "tp", v, (max(S[0][i] - (vo - v), .1), L[0][i])
        return None
    p0s = t["ps"] if i == 0 else S[3][i - 1]; p0l = t["pl"] if i == 0 else L[3][i - 1]
    fs = pd.Series(ms).ffill().fillna(p0s).values; fl = pd.Series(ml).ffill().fillna(p0l).values if hedged else np.zeros(300)
    if i == 0:  # 入場在開盤：從第一筆成交後才開始判定
        first = np.flatnonzero(~np.isnan(ms)); st0 = first[0] if len(first) else 300
    else: st0 = 0
    v = fs - fl
    if hedged: v = np.minimum(v, t["K"] - t["Kl"])
    hit_sl = np.flatnonzero(v[st0:] >= lvl_sl) + st0 if lvl_sl is not None else np.array([], int)
    hit_tp = np.flatnonzero(v[st0:] <= lvl_tp) + st0 if lvl_tp is not None else np.array([], int)
    cands = [(hit_sl[0], "stop")] if len(hit_sl) else []
    if len(hit_tp): cands.append((hit_tp[0], "tp"))
    if not cands: return None
    m, kind = min(cands); m2 = min(m + 1, 299)
    # 下一分鐘「可得」價格：賣腿取 m 之後第一筆成交（無則用 m 時的填補值）
    nxt = np.flatnonzero(~np.isnan(ms[m + 1:])); ps_ = ms[m + 1 + nxt[0]] if len(nxt) else fs[m]
    if hedged:
        nl = np.flatnonzero(~np.isnan(ml[m + 1:])); pl_ = ml[m + 1 + nl[0]] if len(nl) else fl[m]
    else: pl_ = 0.0
    return kind, ps_ - pl_, (ps_, pl_)


def sim(t, tp, sl, need, rand_exit=None):
    """回傳 dict：net(點)、日損益陣列、日盤中最差陣列(相對當日結算損益)、出場原因、持有天數。"""
    S, L = t["S"], t["L"]; C0 = t["C0"]; hedged = t["w"] > 0; n = len(t["hold"])
    vc = S[3] - L[3]; vw = np.maximum(np.minimum(S[1] - L[2], t["K"] - t["Kl"]) if hedged else S[1], vc)
    lsl = (1 + sl) * C0 if sl else None; ltp = (1 - tp) * C0 if tp else None
    cin = opt_cost_pts(t["ps"]) + (opt_cost_pts(t["pl"]) if hedged else 0)
    ex, why, vexit, legs = n - 1, "settle", None, None
    for i in range(n - 1):
        if rand_exit is not None:
            if i == rand_exit: ex, why, vexit, legs = i, "rand", vc[i], (S[3][i], L[3][i]); break
            continue
        if (lsl is not None and vw[i] >= lsl) or (ltp is not None and max(S[2][i] - L[1][i], 0) <= ltp):
            r = resolve(t, i, lsl, ltp, need)
            if r: ex, why, vexit, legs = i, r[0], r[1], r[2]; break
    pnl = np.empty(ex + 1); prev = C0
    for i in range(ex + 1): pnl[i] = prev - vc[i]; prev = vc[i]
    worst = vc[:ex + 1] - vw[:ex + 1]; pnl[0] -= cin
    if why != "settle":
        pv_ = C0 if ex == 0 else vc[ex - 1]
        pnl[ex] = pv_ - vexit - opt_cost_pts(max(legs[0], .1)) - (opt_cost_pts(max(legs[1], .1)) if hedged else 0)
        worst[ex] = 0  # 出場日：損益已含成交價（含跳空），不另計盤中
    else:
        iS = max(t["K"] - t["Fc"][-1], 0); iL = max(t["Kl"] - t["Fc"][-1], 0) if hedged else 0
        pnl[ex] -= (OCOMM / MULT + OTAX * iS if iS > 0 else 0) + (OCOMM / MULT + OTAX * iL if iL > 0 else 0)
    return dict(net=pnl.sum(), pnl=pnl, worst=worst, why=why, L=ex + 1)


def margin_unit(t):
    if t["w"] > 0: return (t["K"] - t["Kl"]) * MULT
    A = .075 * t["Fs"] * MULT; return t["ps"] * MULT + max(A - max(t["Fs"] - t["K"], 0) * MULT, A / 2)


def run_variant(TT, tp, sl, days, need, u=None, rand=False):
    pc = pd.Series(0.0, days); wc = pd.Series(0.0, days); rec = []; eq = CAP0
    for t in TT:
        re_ = int(rng.integers(0, len(t["hold"]) - 1)) if rand and rng.random() < rand else None
        r = sim(t, tp, sl, need, rand_exit=re_) if rand else sim(t, tp, sl, need)
        k = 1 if u is None else int(u * eq / margin_unit(t))
        if k <= 0: break
        idx = t["hold"][:r["L"]]; d = r["pnl"] * MULT * k
        pc[idx] += d; wc[idx] = np.minimum(wc[idx].values, r["worst"] * MULT * k)
        rec.append(dict(s=t["s"], X=t["X"], k=k, net=d.sum(), net1=r["net"] * MULT, why=r["why"], C0=t["C0"], eq0=eq))
        eq += d.sum()
        if eq <= 0: break
    return pc, wc, pd.DataFrame(rec)


def stats(pc, wc, rec, a, b, eq0=0.0):
    m = (pc.index >= a) & (pc.index < b); p = pc[m]; w = wc[m]
    base = eq0 + pc[pc.index < a].sum(); eq = base + p.cumsum(); prev = eq.shift(fill_value=base)
    r = rec[(rec.s >= a) & (rec.s < b)]; win = r[r.net1 > 0]; los = r[r.net1 <= 0]
    peak = np.maximum.accumulate(np.r_[base, eq.values])[:-1]
    out = dict(n=len(r), win=len(win) / max(len(r), 1), avg_win=win.net1.mean(), avg_loss=los.net1.mean(), net=p.sum(),
               Sharpe=sharpe(p if eq0 == 0 else p / prev), MDDw=float((prev.values + w.values - peak).min()), worst=r.net1.min(),
               stop_rate=(r.why == "stop").mean(), tp_rate=(r.why == "tp").mean())
    if eq0: out.update(MDDw_pct=float(((prev.values + w.values) / peak - 1).min()), CAGR=(eq.iloc[-1] / base) ** (252 / len(p)) - 1 if eq.iloc[-1] > 0 else -1)
    return out


def ensure_minutes(T, days):
    need = {}
    for it in range(10):
        need = {}
        for t in T:
            for tp in TPS:
                for sl in SLS: sim(t, tp, sl, need)
        need = {d: {X: {k for k in ks if (d, X, k) not in _tried} for X, ks in v.items()} for d, v in need.items()}
        need = {d: {X: ks for X, ks in v.items() if ks} for d, v in need.items()}; need = {d: v for d, v in need.items() if v}
        print(f"round {it}: 需補 1 分鐘資料 {len(need)} 日", flush=True)
        if not need: break
        from tools.opt_convert import index_sources
        src = index_sources(ZIP); jobs = [(d, src[d], {X: sorted(ks) for X, ks in v.items()}) for d, v in sorted(need.items()) if d in src]
        for d, v in need.items(): _tried.update((d, X, k) for X, ks in v.items() for k in ks)
        with ProcessPoolExecutor(3) as ex:
            for i, (d, s) in enumerate(ex.map(_min_one, jobs)):
                if i % 50 == 0 or s.startswith("err"): print(" ", i, d, s, flush=True)
        _MC.clear()
_tried = set()


def main():
    T = pickle.load(open(CACHE, "rb")) if CACHE.exists() else build()
    T = [t for t in T if t["s"] >= IS0]
    F = load_fut("2016-06-01"); days = F.index[(F.index <= END) & (F.index >= IS0)]
    ensure_minutes(T, days)
    G = {}
    for t in T: G.setdefault(t["st"], []).append(t)
    rows, ser, eqrows, yrows, fb = [], {}, [], [], []
    PER = (("IS", IS0, IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1)))
    for st in STRUCTS:
        for tp, sl in itertools.product(TPS, SLS):
            need = {}; pc, wc, rec = run_variant(G[st], tp, sl, days, need); fb.append(len(need))
            key = (sname(st), f"TP{int(tp*100)}%" if tp else "TP無", "SL2x" if sl else "SL無"); ser[key] = pc
            r = dict(struct=key[0], tp=key[1], sl=key[2])
            for nm, a, b in PER:
                for k, v in stats(pc, wc, rec, a, b).items(): r[f"{nm}_{k}"] = v
            m20 = rec[rec.X.isin(["202003", "202004"])]
            w20 = wc[(wc.index >= "2020-02-15") & (wc.index <= "2020-04-20")]
            r.update(m2020_net=m20.net1.sum(), m2020_why="/".join(m20.why), m2020_intraday_worst=w20.min())
            rows.append(r)
            for Y, g in rec.groupby(rec.s.dt.year): yrows.append(dict(struct=key[0], tp=key[1], sl=key[2], year=Y, net=g.net1.sum(), stops=(g.why == "stop").sum()))
            for u in (.10, .25):
                pe, we, re = run_variant(G[st], tp, sl, days, {}, u=u)
                e = dict(struct=key[0], tp=key[1], sl=key[2], u=u)
                for nm, a, b in PER + (("ALL", IS0, END + pd.Timedelta(days=1)),):
                    s_ = stats(pe, we, re, a, b, eq0=CAP0); e.update({f"{nm}_{k}": s_[k] for k in ("CAGR", "Sharpe", "MDDw_pct")})
                eqrows.append(e)
    print("日 K 退回（缺 1 分鐘）次數合計", sum(fb))
    R = pd.DataFrame(rows); R.to_csv(RES / "dd_put_tpsl_grid.csv", index=False)
    EQ = pd.DataFrame(eqrows); EQ.to_csv(RES / "dd_put_tpsl_equity.csv", index=False)
    Yd = pd.DataFrame(yrows); Yd.to_csv(RES / "dd_put_tpsl_yearly.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 50)
    sp = R.IS_Sharpe.rank().corr(R.OOS_Sharpe.rank()); print(f"Spearman IS→OOS Sharpe（72 變體）{sp:.3f}")
    for s_ in R.struct.unique():
        q = R[R.struct == s_]; print(f"  {s_}: 結構內 Spearman {q.IS_Sharpe.rank().corr(q.OOS_Sharpe.rank()):.2f}")
    cols = ["struct", "tp", "sl", "IS_n", "IS_win", "IS_net", "IS_Sharpe", "OOS_n", "OOS_win", "OOS_avg_win", "OOS_avg_loss", "OOS_net", "OOS_Sharpe", "OOS_MDDw", "OOS_worst", "OOS_stop_rate", "OOS_tp_rate", "m2020_net", "m2020_why"]
    print(R[cols].round(2).to_string(index=False))
    print(EQ.round(3).to_string(index=False))
    # 停利/停損邊際
    print("\n邊際平均 Sharpe"); print(R.groupby("tp")[["IS_Sharpe", "OOS_Sharpe"]].mean().round(2)); print(R.groupby("sl")[["IS_Sharpe", "OOS_Sharpe", "OOS_MDDw"]].mean().round(2))
    best = R.loc[R.IS_Sharpe.idxmax()]; bk = (best.struct, best.tp, best.sl)
    print(f"IS 最佳 {bk} OOS Sharpe {best.OOS_Sharpe:.2f}；72 變體 OOS 百分位 {(R.OOS_Sharpe < best.OOS_Sharpe).mean():.2f} 中位 {R.OOS_Sharpe.median():.2f}")
    # 隨機出場基準：同結構、同出場比例（= 停損+停利率）隨機日平倉，200 次
    rb = []
    for st in (STRUCTS[2], STRUCTS[7]):
        for tp, sl in ((.5, 2.0), (None, 2.0)):
            k = (sname(st), f"TP{int(tp*100)}%" if tp else "TP無", "SL2x" if sl else "SL無"); row = R[(R.struct == k[0]) & (R.tp == k[1]) & (R.sl == k[2])].iloc[0]
            rate = row.OOS_stop_rate + row.OOS_tp_rate
            sh = [stats(*run_variant(G[st], None, None, days, {}, rand=rate), IS_END, END + pd.Timedelta(days=1))["Sharpe"] for _ in range(200)]
            rb.append(dict(variant=str(k), real_OOS=row.OOS_Sharpe, rand_med=np.median(sh), pct=(np.array(sh) < row.OOS_Sharpe).mean()))
    RB = pd.DataFrame(rb); print("\n隨機出場基準"); print(RB.round(2).to_string(index=False)); RB.to_csv(RES / "dd_put_tpsl_random.csv", index=False)
    # WF：每年用 2017..Y-1 擴張窗選 Sharpe 最佳
    wf, picks = [], []
    for Y in range(2019, 2027):
        a, b, c = IS0, pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y+1}-01-01")
        sc = {k: sharpe(s[(s.index >= a) & (s.index < b)]) for k, s in ser.items()}; kb = max(sc, key=sc.get)
        seg = ser[kb][(ser[kb].index >= b) & (ser[kb].index < c)]; wf.append(seg); picks.append(dict(year=Y, pick=str(kb), fit_sharpe=sc[kb], year_net=seg.sum()))
    wf = pd.concat(wf); W = pd.DataFrame(picks); W.to_csv(RES / "dd_put_tpsl_wf.csv", index=False); print(W.round(2).to_string(index=False))
    print(f"WF 2022+ Sharpe {sharpe(wf[wf.index >= IS_END]):.2f} net {wf[wf.index >= IS_END].sum():.0f}")
    chart(ser, R, wf)


def chart(ser, R, wf):
    plt = plt_style(); fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
    ax = axs[0]; sty(ax); cols = [GOLD, GR, BLUE, "#e07b39", WH, "#9ad36a"]
    pick = [("裸賣 d0.20", "TP無", "SL無"), ("裸賣 d0.20", "TP50%", "SL2x"), ("裸賣 d0.20", "TP無", "SL2x"), ("價差 d0.20 w2%", "TP無", "SL無"), ("價差 d0.20 w2%", "TP50%", "SL2x"), ("價差 d0.20 w2%", "TP無", "SL2x")]
    for k, c in zip(pick, cols):
        if k in ser: ax.plot(ser[k].cumsum() / 1e4, color=c, lw=1.3, label=" ".join(k))
    ax.plot(wf.cumsum() / 1e4, color="#d36ac4", lw=1, ls="--", label="walk-forward")
    ax.axvline(IS_END, color="white", ls="--", lw=.8); ax.set_title("累積淨損益（萬元，1 組）"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=8)
    ax = axs[1]; sty(ax); ax.scatter(R.IS_Sharpe, R.OOS_Sharpe, c=np.where(R.sl == "SL2x", 1, 0), cmap="cividis", s=20)
    ax.set_xlabel("IS Sharpe"); ax.set_ylabel("OOS Sharpe"); ax.set_title("72 變體 IS vs OOS（亮=兩倍停損）")
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "dd_put_tpsl.png", dpi=110, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
