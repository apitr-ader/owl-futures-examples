"""81 月選賣方：裸賣 put vs 勒式（sell put + sell call）vs 鐵兀鷹（put 價差 + call 價差），1 口（承 76/77/80 引擎）。

時序同 76：前一月選結算日 s 收盤選履約價 → s+1 日開盤成交（無開盤價用收盤），持有至結算（結算價 ≈ TXF 收盤）。
履約價：put K≈F(1−x) 取 K<F；call K≈F(1+x) 取 K>F；價差買腿 = 賣腿再往外 w=4%×F 最近履約價。
停損（2x）：組合價值（空腿 − 多腿）≥ 3×C0（C0 = 入場總權利金點數），盤中判定：日 K 最差值篩選 → 1 分鐘最後成交
  （逐筆 RPT，前向填補）找第一個觸發分鐘，各腿於之後第一筆成交出場；缺分鐘資料退回 max(門檻, 開盤價值)。
  鐵兀鷹另有「各邊獨立 2x 停損」變體（put/call 價差各自以自身權利金判定）。
成本：每腿每邊 30 元 + 0.1%×權利金 + 1 tick；結算價內每腿 30 元 + 0.1%×內含值；1 點 = 50 元。
MDD：逐日 mark-to-market + 盤中最差（空腿取日高、多腿取日低，價差各邊以寬度為上限）。
Sharpe：月損益 ×√12。IS 2017–2021、OOS 2022+。任一腿入場日無報價 → 該變體略過該月（列出 n）。
輸出：results/dd_strangle_{summary,monthly,yearly_legs,trades}.csv、dd_strangle_equity.png；快取 data/cache/（不入 repo）。
"""
import importlib.util, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from _rqo import *
spec = importlib.util.spec_from_file_location("m74", Path(__file__).parent / "74_txo_put_deepdive_grid.py"); m74 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m74)

END = m74.END; IS0 = pd.Timestamp("2017-01-01")
CACHE = ROOT / "data/cache/dd_strangle_months.pkl"; MIN = ROOT / "data/cache/txo_1m_cp"; MIN.mkdir(parents=True, exist_ok=True)
ZIP = str(ROOT / "data/options_daily/zip")
# 變體：(名稱, [(cp, x, w)], 停損模式)；w=0 裸賣；停損 None / "tot" / "side"
V = [("裸賣 P2%", [("P", .02, 0)], None),
     ("勒式 P2%+C2%", [("P", .02, 0), ("C", .02, 0)], None),
     ("勒式 P2%+C3%", [("P", .02, 0), ("C", .03, 0)], None),
     ("勒式 P3%+C3%", [("P", .03, 0), ("C", .03, 0)], None),
     ("put價差 2%/w4%", [("P", .02, .04)], None),
     ("put價差 2%/w4% SL2x", [("P", .02, .04)], "tot"),
     ("鐵兀鷹 2%/w4%", [("P", .02, .04), ("C", .02, .04)], None),
     ("鐵兀鷹 2%/w4% SL2x合計", [("P", .02, .04), ("C", .02, .04)], "tot"),
     ("鐵兀鷹 2%/w4% SL2x各邊", [("P", .02, .04), ("C", .02, .04)], "side")]
SIDES = sorted({s for _, ss, _ in V for s in ss})


def leg_path(pv, K, hold, Fc, Fl, Fh, cp):
    if cp == "P": return m74.leg_path(pv, K, hold, Fc, Fl, Fh, True)
    get = lambda f: pv[f][K].reindex(hold).values.astype(float) if K in pv[f].columns else np.full(len(hold), np.nan)
    o, h, l, c = get("day_open"), get("day_high"), get("day_low"), get("day_close")
    ic, ih, il = np.maximum(Fc - K, 0), np.maximum(Fh - K, 0), np.maximum(Fl - K, 0)
    last = np.nan
    for i in range(len(hold)):
        if np.isnan(c[i]): c[i] = max(last, ic[i]) if not np.isnan(last) else ic[i]
        last = c[i]
    c[-1] = ic[-1]; o = np.where(np.isnan(o), c, o)
    h = np.where(np.isnan(h), np.maximum(c, ih), h); h = np.maximum(h, c)
    l = np.where(np.isnan(l), np.minimum(c, il), l); l = np.minimum(l, c)
    return o, h, l, c


def build():
    F = load_fut("2016-06-01"); days = F.index[F.index <= END]
    E = expiries(); allm = E[E.kind == "monthly"].sort_values("expiry_date").reset_index(drop=True)
    M = allm[(allm.expiry_date <= END) & (allm.expiry_date >= IS0)]
    O0 = load_opt("2016-11-01", END); O0 = O0[O0.expiry.isin(M.expiry)]
    T = []
    for X, xd in zip(M.expiry, M.expiry_date):
        if xd not in days: continue
        s = allm[allm.expiry_date < xd].expiry_date.iloc[-1]
        if s not in days: continue
        si = days.get_loc(s); d1 = days[si + 1]; ix = days.get_loc(xd); hold = days[si + 1: ix + 1]
        Fs = F.close[s]; Fc, Fl, Fh = F.close[hold].values, F.low[hold].values, F.high[hold].values
        sides = {}
        for cp in ("P", "C"):
            g = O0[(O0.expiry == X) & (O0.cp == cp)]
            pv = {f: g.pivot_table(index="date", columns="strike", values=f) for f in ("day_open", "day_high", "day_low", "day_close")}
            if s not in pv["day_close"].index or d1 not in pv["day_close"].index: continue
            cl = pv["day_close"].loc[s].dropna(); cl = cl[((cl.index < Fs) if cp == "P" else (cl.index > Fs)) & (cl > 0.5)]
            if len(cl) < 3: continue
            px = lambda K: (pv["day_open"].at[d1, K] if not np.isnan(pv["day_open"].at[d1, K]) else pv["day_close"].at[d1, K]) if K in pv["day_open"].columns else np.nan
            sg = -1 if cp == "P" else 1
            for c_, x, w in SIDES:
                if c_ != cp: continue
                K = cl.index[np.abs(cl.index - Fs * (1 + sg * x)).argmin()]; ps = px(K)
                if np.isnan(ps) or ps < 1: continue
                legs = [dict(cp=cp, K=K, q=1, p0=ps, path=leg_path(pv, K, hold, Fc, Fl, Fh, cp))]
                if w:
                    cols = pv["day_close"].columns; cand = cols[cols < K] if cp == "P" else cols[cols > K]
                    if len(cand) == 0: continue
                    Kl = cand[np.abs(cand - (K + sg * w * Fs)).argmin()]; pl = px(Kl)
                    if np.isnan(pl): continue
                    legs.append(dict(cp=cp, K=Kl, q=-1, p0=pl, path=leg_path(pv, Kl, hold, Fc, Fl, Fh, cp)))
                sides[(cp, x, w)] = dict(legs=legs, width=abs(K - legs[1]["K"]) if w else None, C0=ps - (legs[1]["p0"] if w else 0))
        T.append(dict(X=X, s=s, d1=d1, hold=hold, Fs=Fs, Fc=Fc, sides=sides))
    pickle.dump(T, open(CACHE, "wb")); return T


# ---------- 1 分鐘（逐筆 RPT → 分鐘最後成交，P/C 皆存；不入 repo）----------
def _min_one(args):
    d, refs, want = args
    from tools.opt_convert import read_rpt, parse
    o = MIN / f"{d}.parquet"
    try:
        b, _, _ = read_rpt(refs); df, *_ = parse(b, d); del b
        if df is None: return d, "none"
        df = df[~df.night]; df["expiry"] = df.expiry.str.strip()
        df = df[(df.sec >= 8 * 3600 + 45 * 60) & (df.sec < 13 * 3600 + 45 * 60)]
        keep = np.zeros(len(df), bool)
        for (X, cp), ks in want.items(): keep |= ((df.expiry == X) & (df.cp == cp) & df.strike.isin(ks)).values
        x = df[keep]; x = x.assign(m=((x.sec - (8 * 3600 + 45 * 60)) // 60).astype(int))
        r = x.groupby(["expiry", "cp", "strike", "m"]).price.last().reset_index()
        mark = pd.DataFrame([(X, cp, k, -1, np.nan) for (X, cp), ks in want.items() for k in ks], columns=["expiry", "cp", "strike", "m", "price"])
        r = pd.concat([r, mark]); r["strike"] = r.strike.astype(float)
        old = pd.read_parquet(o) if o.exists() else None
        r = pd.concat([old, r]).drop_duplicates(["expiry", "cp", "strike", "m"]) if old is not None else r
        r.to_parquet(o); return d, f"ok {len(r)}"
    except Exception as e:
        return d, f"err {e}"


_MC = {}
def minutes(d, X, cp, K):
    key = str(d.date())
    if key not in _MC:
        p = MIN / f"{key}.parquet"; _MC[key] = pd.read_parquet(p) if p.exists() else None
    g = _MC[key]
    if g is None: return None
    h = g[(g.expiry == X) & (g.cp == cp) & (g.strike == float(K))]
    if not (h.m == -1).any(): return None
    h = h[h.m >= 0]; a = np.full(300, np.nan); a[h.m.values.clip(0, 299)] = h.price.values; return a


def grp_vals(G, i, kind):
    """一組價差/裸腿第 i 日價值：kind o/c/w(最差)/b(最佳)。"""
    L = G["legs"]
    if kind in "oc":
        v = sum(l["q"] * l["path"]["oc".index(kind) * 3] [i] for l in L)
    elif kind == "w": v = sum(l["path"][1][i] if l["q"] > 0 else -l["path"][2][i] for l in L)
    if G["width"] is not None: v = min(v, G["width"])
    return v


def sim(t, groups, stop, need):
    """groups: [[side,...] 停損群組]。回傳每腿每日損益（點，含成本）與每日盤中最差。"""
    n = len(t["hold"]); allsides = [s for g in groups for s in g]
    legs_pnl = {k: np.zeros(n) for k in allsides}; worst = np.zeros(n); why = []
    for grp in groups:
        S = [t["sides"][k] for k in grp]; C0 = sum(s["C0"] for s in S); lvl = 3 * C0 if stop else None
        vc = np.array([[grp_vals(s, i, "c") for i in range(n)] for s in S])  # sides × days
        vw = np.array([[max(grp_vals(s, i, "w"), vc[j, i]) for i in range(n)] for j, s in enumerate(S)])
        ex, exv = n - 1, None
        if stop:
            for i in range(n - 1):
                if vw[:, i].sum() < lvl: continue
                r = resolve(t, i, S, lvl, need)
                if r is not None: ex, exv = i, r; break
        for j, (k, s) in enumerate(zip(grp, S)):
            p = np.zeros(n); prev = s["C0"]
            for i in range(ex + 1): p[i] = prev - vc[j, i]; prev = vc[j, i]
            p[0] -= sum(opt_cost_pts(l["p0"]) for l in s["legs"])
            if exv is not None:
                lp = exv[j]; pv_ = s["C0"] if ex == 0 else vc[j, ex - 1]
                p[ex] = pv_ - sum(l["q"] * x for l, x in zip(s["legs"], lp)) - sum(opt_cost_pts(max(x, .1)) for x in lp)
            else:
                for l in s["legs"]:
                    iv = max((l["K"] - t["Fc"][-1]) if l["cp"] == "P" else (t["Fc"][-1] - l["K"]), 0)
                    if iv > 0: p[ex] -= OCOMM / MULT + OTAX * iv
            legs_pnl[k] = p
        w = (vc[:, :ex + 1] - vw[:, :ex + 1]).sum(0)
        if exv is not None: w[ex] = 0
        worst[:ex + 1] += w; why.append("stop" if exv is not None else "settle")
    return legs_pnl, worst, "/".join(why)


def resolve(t, i, S, lvl, need):
    d = t["hold"][i]; X = t["X"]; allL = [l for s in S for l in s["legs"]]
    ms = [minutes(d, X, l["cp"], l["K"]) for l in allL]
    if any(m_ is None for m_ in ms):
        for l in allL: need.setdefault(str(d.date()), {}).setdefault((X, l["cp"]), set()).add(l["K"])
        vo = sum(grp_vals(s, i, "o") for s in S); vw = sum(grp_vals(s, i, "w") for s in S)
        if vw < lvl: return None
        add = max(lvl - vo, 0)  # 以 max(門檻, 開盤) 成交：差額加到第一個空腿
        out = []
        for s in S:
            lp = [l["path"][0][i] for l in s["legs"]]; out.append(lp)
        out[int(np.argmax([s_["legs"][0]["path"][0][i] - (s_["legs"][0]["p0"]) for s_ in S]))][0] += add
        return out
    fill = []
    for l, m_ in zip(allL, ms):
        p0 = l["p0"] if i == 0 else l["path"][3][i - 1]
        fill.append(pd.Series(m_).ffill().fillna(p0).values)
    st0 = 0
    if i == 0:
        f0 = np.flatnonzero(~np.isnan(ms[0])); st0 = f0[0] if len(f0) else 300
    k = 0; v = np.zeros(300)
    for s in S:
        vs = np.zeros(300)
        for l in s["legs"]: vs += l["q"] * fill[k]; k += 1
        if s["width"] is not None: vs = np.minimum(vs, s["width"])
        v += vs
    hit = np.flatnonzero(v[st0:] >= lvl) + st0
    if not len(hit): return None
    m0 = hit[0]; out = []; k = 0
    for s in S:
        lp = []
        for l in s["legs"]:
            nx = np.flatnonzero(~np.isnan(ms[k][m0 + 1:])); lp.append(ms[k][m0 + 1 + nx[0]] if len(nx) else fill[k][m0]); k += 1
        out.append(lp)
    return out


def groups_of(sides, stop):
    return [[s] for s in sides] if stop == "side" else [list(sides)]


def ensure_minutes(T):
    tried = set()
    for it in range(6):
        need = {}
        for t in T:
            for _, sides, stop in V:
                if stop and all(s in t["sides"] for s in sides): sim(t, groups_of(sides, stop), True, need)
        need = {d: {k: {x for x in ks if (d, k, x) not in tried} for k, ks in v.items()} for d, v in need.items()}
        need = {d: {k: ks for k, ks in v.items() if ks} for d, v in need.items()}; need = {d: v for d, v in need.items() if v}
        print(f"round {it}: 需補 1 分鐘資料 {len(need)} 日", flush=True)
        if not need: return
        from tools.opt_convert import index_sources
        src = index_sources(ZIP); jobs = [(d, src[d], {k: sorted(ks) for k, ks in v.items()}) for d, v in sorted(need.items()) if d in src]
        for d, v in need.items(): tried.update((d, k, x) for k, ks in v.items() for x in ks)
        with ProcessPoolExecutor(3) as ex:
            for i, (d, s) in enumerate(ex.map(_min_one, jobs)):
                if i % 50 == 0 or s.startswith("err"): print(" ", i, d, s, flush=True)
        _MC.clear()


def msh(x): x = np.asarray(x, float); return float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else 0.0


def main():
    T = pickle.load(open(CACHE, "rb")) if CACHE.exists() else build()
    T = [t for t in T if t["s"] >= IS0]; ensure_minutes(T)
    F = load_fut("2016-06-01"); days = F.index[(F.index <= END) & (F.index >= IS0)]
    # 共同樣本：所有變體各腿皆有報價的月份（公平比較；鐵兀鷹 call 買腿常缺報價）
    TC = [t for t in T if all(s in t["sides"] for s in SIDES)]
    print(f"月份：全部 {len(T)}，共同樣本 {len(TC)}")
    Sc, _, _, serc = run(TC, days); Sc.to_csv(RES / "dd_strangle_summary_common.csv", index=False)
    S, TR, Y, ser = run(T, days)
    S.to_csv(RES / "dd_strangle_summary.csv", index=False)
    pd.set_option("display.width", 400); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
    for lab, X_ in (("各變體自身可交易月份", S), ("共同樣本", Sc)):
        for nm in ("IS", "OOS"):
            print(f"\n=== {lab} {nm} ==="); print(X_[["variant"] + [c for c in X_.columns if c.startswith(nm + "_")]].round(2).to_string(index=False))
    print("\nfallback days", S[["variant", "fallback_days"]].to_string(index=False))
    print("\n逐年 call 腿損益（元）"); print(Y[Y.variant.str.contains("勒式|鐵兀鷹 2%/w4%$")].pivot_table(index="year", columns="variant", values="call").round(0).to_string())
    print("\n逐年總損益"); print(Y.pivot_table(index="year", columns="variant", values="net").round(0).to_string())
    big = TR[TR.variant == "勒式 P2%+C2%"].nsmallest(12, "call")[["X", "chg", "put", "call", "net"]]; print("\n勒式 P2%+C2% call 腿最差 12 月"); print(big.round(3).to_string(index=False))
    chart(serc)


def run(T, days):
    rows, trows, ser, mon = [], [], {}, {}
    for name, sides, stop in V:
        pc = pd.Series(0.0, days); wc = pd.Series(0.0, days); legd = {"P": pd.Series(0.0, days), "C": pd.Series(0.0, days)}; rec = []; need = {}
        for t in T:
            if not all(s in t["sides"] for s in sides): continue
            lp, w, why = sim(t, groups_of(sides, stop), bool(stop), need)
            tot = sum(lp.values()) * MULT; pc[t["hold"]] += tot; wc[t["hold"]] += w * MULT
            for k, v in lp.items(): legd[k[0]][t["hold"]] += v * MULT
            rec.append(dict(variant=name, X=t["X"], s=t["s"], Fs=t["Fs"], F_settle=t["Fc"][-1], chg=t["Fc"][-1] / t["Fs"] - 1, net=tot.sum(),
                            put=sum(v.sum() for k, v in lp.items() if k[0] == "P") * MULT, call=sum(v.sum() for k, v in lp.items() if k[0] == "C") * MULT, why=why))
        R = pd.DataFrame(rec); trows.append(R); ser[name] = pc
        mp = pc.groupby(pc.index.to_period("M")).sum(); mon[name] = mp
        r = dict(variant=name, fallback_days=len(need))
        for nm, a, b in (("IS", IS0, IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1))):
            sel = (pc.index >= a) & (pc.index < b); p = pc[sel]; w = wc[sel]; eq = p.cumsum()
            peak = np.maximum.accumulate(np.r_[0, eq.values])[:-1]; prev = eq.shift(fill_value=0).values
            dd = float((prev + w.values - peak).min()); ddc = mdd(eq.values)
            h = R[(R.s >= a) & (R.s < b)]; mm = mp[(mp.index >= pd.Period(a, "M")) & (mp.index < pd.Period(b, "M"))]
            wi = h.net.idxmin()
            r.update({f"{nm}_n": len(h), f"{nm}_net": p.sum(), f"{nm}_Sharpe": msh(mm), f"{nm}_MDD_intraday": dd, f"{nm}_MDD_close": ddc,
                      f"{nm}_worst": h.net.min(), f"{nm}_worst_X": h.X[wi], f"{nm}_win": (h.net > 0).mean(), f"{nm}_ret_MDD": p.sum() / -dd if dd < 0 else np.nan,
                      f"{nm}_corr_PC": h.put.corr(h.call) if (h.call != 0).any() else np.nan, f"{nm}_stops": h.why.str.contains("stop").sum()})
        rows.append(r)
    S = pd.DataFrame(rows); TR = pd.concat(trows); M = pd.DataFrame(mon)
    TR.to_csv(RES / "dd_strangle_trades.csv", index=False); M.to_csv(RES / "dd_strangle_monthly.csv")
    TR["year"] = TR.s.dt.year
    Y = TR.groupby(["variant", "year"])[["net", "put", "call"]].sum().reset_index(); Y.to_csv(RES / "dd_strangle_yearly_legs.csv", index=False)
    return S, TR, Y, ser


def chart(ser):
    plt = plt_style(); fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG); sty(ax)
    pick = [("裸賣 P2%", GOLD, "-"), ("勒式 P2%+C2%", "#e07b39", "-"), ("勒式 P2%+C3%", "#d36ac4", "-"), ("勒式 P3%+C3%", "#ff6b6b", ":"),
            ("put價差 2%/w4% SL2x", BLUE, "--"), ("鐵兀鷹 2%/w4%", "#9ad36a", "-"), ("鐵兀鷹 2%/w4% SL2x合計", WH, "--"), ("鐵兀鷹 2%/w4% SL2x各邊", GR, "--")]
    for k, c, ls in pick: ax.plot(ser[k].cumsum() / 1e4, color=c, ls=ls, lw=1.3, label=k)
    ax.axvline(IS_END, color="white", ls="--", lw=.8); ax.axhline(0, color=GR, lw=.6)
    ax.set_title("TXO 月選賣方：裸賣 put vs 勒式 vs 鐵兀鷹 累積淨損益（萬元，各 1 口，共同樣本月份；虛線左 IS / 右 OOS）")
    ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=9, loc="upper left")
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "dd_strangle_equity.png", dpi=110, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
