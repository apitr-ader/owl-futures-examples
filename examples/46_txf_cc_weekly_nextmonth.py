"""46 台指期 covered call 變體：(A) 週選 (B1/B2) 次月月選；對照 45 的近月月選 CC 與 TXF B&H（日K）。

共同框架同 45：名目無槓桿（起始 = 1 口 TXF 名目），每月月選結算日 E 的次一交易日開盤把 TXF 口數
重設為 權益/(200×開盤)；現金 rf 1.5%；收盤訊號、隔日日盤開盤成交；成本、MDD（盤中最差 / 濾離譜價 / 收盤）同 45。
(A) 週選：單一鏈。上一檔（任何種類）到期日收盤定履約價 → 次一交易日開盤賣出「最快到期」的 call 系列
    （YYYYMMWn 週三週選、2025 起 YYYYMMFn 週五週選、或月選若它最近），持有至結算；結算價近似 = 結算日 TXF 日盤收盤
    （週五週選同樣用週五 TXF 收盤；TXF 該日仍為近月合約，近似指數）。起點 2012-11-21（首檔週選）。
(B1) 每月 E+1 開盤賣出第二近月月選，持有至到期（約 2 個月）；兩腿重疊，每腿 call 口數 = cover×4×TXF口數×0.5，
     使總掩護維持 4 口/TXF（起始第一個月只有 1 腿 = 半掩護）。
(B2) 每月 E+1 開盤賣出第二近月月選（全掩護），下個月 E+1 開盤（該月份變近月時）以開盤價買回 + 1 tick 滑價，
     同時賣出新的次月。買回成本：手續費 30 元/口 + 稅 0.1%×權利金。
call 履約價：當天有成交（日盤開或收）的履約價中最接近目標者；無 day_open 用 day_close；整天無成交 → 順延。
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.optdaily import load_txo_daily, expiries

DATA = "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet"
OUT = Path("results")
RF, TAX, COMM, SLIP = 0.015, 2e-5, 50.0, 1.0
OCOMM, OTAX = 30.0, 0.001
START, END = pd.Timestamp("2010-01-01"), pd.Timestamp("2026-08-18")
W0 = pd.Timestamp("2012-11-21")
E1 = END + pd.Timedelta(days=1); I1 = pd.Timestamp("2020-01-01")
PER_M = (("FULL 2010-26/08", START, E1), ("IS 2010-19", START, I1), ("OOS 2020-26/08", I1, E1))
PER_W = (("FULL 2012/11-26/08", W0, E1), ("IS 2012/11-19", W0, I1), ("OOS 2020-26/08", I1, E1))


def tick(p):
    return 0.1 if p < 10 else 0.5 if p < 50 else 1 if p < 500 else 5 if p < 1000 else 10


def load_fut():
    d = pd.read_parquet(DATA)
    d.index = d["date"].dt.tz_localize(None).dt.normalize()
    d = d[(d.index >= START - pd.Timedelta(days=10)) & (d.index <= END)]
    cm = d.contractMonth.astype(str)
    roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    d = d[["open", "high", "low", "close"]].astype(float)
    d["roll"] = roll
    return d


def load_opt():
    o = load_txo_daily(START - pd.Timedelta(days=80), END, cp="C",
                       columns=["date", "expiry", "strike", "day_open", "day_high", "day_low", "day_close"])
    o["date"] = pd.to_datetime(o.date).dt.normalize()
    o = o[o.expiry.str.match(r"^\d{6}([WF]\d)?$")]
    return {k: g.drop(columns="expiry").set_index(["strike", "date"]).sort_index() for k, g in o.groupby("expiry")}


class Ctx:
    def __init__(s):
        s.F = load_fut(); s.O = load_opt(); s.days = s.F.index
        s.optdays = set().union(*[set(g.index.get_level_values("date")) for g in s.O.values()])
        e = expiries(); s.E = e[e.expiry_date <= END + pd.Timedelta(days=120)].sort_values("expiry_date").reset_index(drop=True)
        s.M = s.E[s.E.kind == "monthly"].reset_index(drop=True)
        # 月結算日（在 TXF 交易日內）
        s.mexp = [d for d in s.M.expiry_date if s.days[0] <= d <= END and d in s.days]


def nxt(ctx, d):
    p = ctx.days.searchsorted(d) + 1
    return p if p < len(ctx.days) else None


def schedule(ctx, mode):
    """回傳 entries：dict(sig, ix, expiry, settle, size, buyback_ix)。"""
    out = []
    if mode == "M0":
        for E in ctx.mexp:
            p = nxt(ctx, E); m = ctx.M[ctx.M.expiry_date > E]
            if p is None or m.empty or m.expiry_date.iloc[0] > END: continue
            out.append(dict(sig=E, ix=p, expiry=m.expiry.iloc[0], settle=m.expiry_date.iloc[0], size=1.0, bb=None))
    elif mode in ("B1", "B2"):
        for j, E in enumerate(ctx.mexp):
            p = nxt(ctx, E); m = ctx.M[ctx.M.expiry_date > E]
            if p is None or len(m) < 2: continue
            if mode == "B1":
                if m.expiry_date.iloc[1] > END: continue
                out.append(dict(sig=E, ix=p, expiry=m.expiry.iloc[1], settle=m.expiry_date.iloc[1], size=0.5, bb=None))
            else:
                if j + 1 >= len(ctx.mexp): continue
                bb = nxt(ctx, ctx.mexp[j + 1])
                if bb is None: continue
                out.append(dict(sig=E, ix=p, expiry=m.expiry.iloc[1], settle=m.expiry_date.iloc[1], size=1.0, bb=bb))
    elif mode == "W":
        E = W0
        while True:
            p = nxt(ctx, E)
            if p is None: break
            d = ctx.days[p]
            c = ctx.E[(ctx.E.expiry_date >= d) & (ctx.E.first_date <= d)]
            if c.empty or c.expiry_date.iloc[0] > END: break
            r = c.iloc[0]
            out.append(dict(sig=E, ix=p, expiry=r.expiry, settle=r.expiry_date, size=1.0, bb=None))
            E = r.expiry_date
            if E not in ctx.days:  # 結算日不在 TXF 交易日（資料缺）→ 以前一交易日
                E = ctx.days[ctx.days.searchsorted(E) - 1]
    return out


def pick(ctx, en, mode, x, log):
    F = ctx.F; days = ctx.days
    sub = ctx.O.get(en["expiry"])
    if sub is None: return None
    target = F.close.loc[en["sig"]] + x if mode == "pt" else F.close.loc[en["sig"]] * (1 + x)
    ix = en["ix"]; kind = "open"
    dates = sub.index.get_level_values("date")
    while True:
        d = days[ix]
        if d in ctx.optdays:
            day = sub.xs(d, level="date") if d in dates else sub.iloc[:0]
            live = day[day.day_open.notna() | day.day_close.notna()]
            if len(live):
                ks = live.index.values.astype(float)
                K = ks[np.argmin(np.abs(ks - target) - (ks >= target) * 1e-6)]
                row = live.loc[K]
                px = row.day_open if not np.isnan(row.day_open) else row.day_close
                if np.isnan(row.day_open): kind = "close_fallback"
                break
            kind = "next_day"
        else:
            kind = "missing_day_next"
        ix += 1
        if ix >= len(days) or days[ix] >= en["settle"]: return None
    log.append(dict(expiry=en["expiry"], fill=kind, delay=ix - en["ix"]))
    return dict(en, K=K, prem=px, ix=ix, ser=sub.xs(K, level="strike"), kind=kind)


def simulate(ctx, mode=None, strike=None, cover=1.0, log=None, fills=None):
    F, days = ctx.F, ctx.days; n = len(days)
    o, h, l, c, roll = (F[k].values for k in ("open", "high", "low", "close", "roll"))
    reb = {nxt(ctx, E) for E in ctx.mexp}
    start_ix = nxt(ctx, ctx.mexp[0])
    ents = {}
    if mode:
        for en in schedule(ctx, mode):
            ents.setdefault(en["ix"], []).append(en)
    eq = np.full(n, np.nan); wq = eq.copy(); wq2 = eq.copy()
    E = o[start_ix] * 200.0; q = 0.0; live = []; pend = []; legs = []; bb_fb = 0
    log = log if log is not None else []
    for t in range(start_ix, n):
        d = days[t]; pc = c[t - 1]; pnl = 0.0
        if not roll[t]: pnl += q * 200 * (o[t] - pc)
        elif q: pnl -= q * 2 * (COMM + 200 * (TAX * o[t] + SLIP))
        if t in reb:
            tq = (E + pnl) / (200 * o[t]); pnl -= abs(tq - q) * (COMM + 200 * (TAX * o[t] + SLIP)); q = tq
        oc = ow = ow2 = 0.0
        # 買回（開盤）
        for lg in list(live):
            if lg["bb"] is not None and t >= lg["bb"]:
                s = lg["ser"]; r = s.loc[d] if d in s.index else None
                px = np.nan
                if r is not None:
                    px = r.day_open if not np.isnan(r.day_open) else r.day_close
                    if np.isnan(r.day_open) and not np.isnan(px): bb_fb += 1
                if np.isnan(px):
                    if t == lg["bb"]: bb_fb += 1
                    continue
                buy = px + tick(px)
                v = -lg["nc"] * 50 * (buy - lg["mark"]) - lg["nc"] * (OCOMM + OTAX * buy * 50)
                oc += v; ow += v; ow2 += v
                legs.append(dict(expiry=lg["expiry"], entry=days[lg["ix"]], exit=d, K=lg["K"], prem=lg["sell"], exitpx=buy,
                                 pnl_pts=lg["sell"] - buy, itm=c[t] > lg["K"], prem_pct=lg["sell"] / c[lg["ix"] - 1], how="buyback"))
                live.remove(lg)
        # 賣出
        if mode:
            for en in ents.get(t, []):
                lg = pick(ctx, en, strike[1], strike[2], log)
                if lg is not None: pend.append(lg)
            for lg in [p for p in pend if p["ix"] == t]:
                pend.remove(lg)
                sell = lg["prem"] - tick(lg["prem"])
                lg.update(sell=sell, mark=sell, nc=cover * 4 * q * lg["size"])
                pnl -= lg["nc"] * (OCOMM + OTAX * sell * 50)
                live.append(lg)
        # 期貨日內
        fc = q * 200 * (c[t] - o[t]); fl = q * 200 * (l[t] - o[t])
        for lg in list(live):
            s = lg["ser"]; r = s.loc[d] if d in s.index else None
            dh = r.day_high if r is not None else np.nan
            dc = r.day_close if r is not None else np.nan
            do = r.day_open if r is not None else np.nan
            m0 = lg["mark"]; nc = lg["nc"]; K = lg["K"]
            if d >= lg["settle"]:
                pay = max(c[t] - K, 0.0)
                hi = max(pay, dh) if not np.isnan(dh) else max(pay, m0)
                cap = 2 * max(m0, pay) + max(h[t] - K, 0.0)
                oc += -nc * 50 * (pay - m0) - (nc * OCOMM if pay > 0 else 0)
                ow += -nc * 50 * (hi - m0); ow2 += -nc * 50 * (min(hi, cap) - m0)
                legs.append(dict(expiry=lg["expiry"], entry=days[lg["ix"]], exit=d, K=K, prem=lg["sell"], exitpx=pay,
                                 pnl_pts=lg["sell"] - pay, itm=pay > 0, prem_pct=lg["sell"] / c[lg["ix"] - 1], how="settle"))
                live.remove(lg)
            else:
                newm = dc if not np.isnan(dc) else max(m0, c[t] - K, 0.0)
                hi = dh if not np.isnan(dh) else max(newm, m0, h[t] - K)
                hi = max(hi, newm)
                cap = 2 * np.nanmax([m0, newm, do]) + max(h[t] - K, 0.0)
                oc += -nc * 50 * (newm - m0); ow += -nc * 50 * (hi - m0); ow2 += -nc * 50 * (min(hi, cap) - m0)
                lg["mark"] = newm
        pre = E + pnl + E * RF / 252
        wq[t] = pre + fl + ow; wq2[t] = pre + fl + ow2
        E = pre + fc + oc; eq[t] = E
    S = lambda a: pd.Series(a, days).dropna()
    return S(eq), S(wq), pd.DataFrame(legs), S(wq2), bb_fb


def stats(e, w, w2, lo, hi):
    m = (e.index >= lo) & (e.index < hi); e, w, w2 = e[m], w[m], w2[m]
    r = e.pct_change().dropna(); y = len(r) / 252
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / y) - 1; x = r - RF / 252
    pk = e.cummax(); pks = pk.shift().fillna(e.iloc[0])
    mc = (e / pk - 1).min(); mw = min((w / pks - 1).min(), mc); mw2 = min((w2 / pks - 1).min(), mc)
    return dict(CAGR=cagr, Sharpe=x.mean() / x.std() * math.sqrt(252), MDD_intra=mw, MDD_intra_cap=mw2, MDD_close=mc, Calmar=cagr / abs(mw))


def main():
    OUT.mkdir(exist_ok=True)
    ctx = Ctx()
    sw = schedule(ctx, "W"); print("weekly legs", len(sw), sw[0]["expiry"], sw[0]["ix"], sw[-1]["expiry"], sw[-1]["settle"].date(),
                                   pd.Series([s["expiry"][6:7] or "M" for s in sw]).value_counts().to_dict())
    SM = [("ATM", "pt", 0), ("+200", "pt", 200), ("+500", "pt", 500), ("+2%", "pct", .02), ("+5%", "pct", .05)]
    SW = [("ATM", "pt", 0), ("+100", "pt", 100), ("+200", "pt", 200), ("+300", "pt", 300), ("+1%", "pct", .01), ("+2%", "pct", .02), ("+3%", "pct", .03)]
    runs = {"TXF B&H": ("-", None, None, 0)}
    for mode, SS in (("M0", SM), ("B1", SM), ("B2", SM), ("W", SW)):
        for st in SS:
            for cv in (1.0, 0.5, 0.25):
                runs[f"{mode} {st[0]} x{int(cv*4)}"] = (mode, mode, st, cv)
    rows = []; legs = []; fl = []; res = {}
    for name, (fam, mode, st, cv) in runs.items():
        lg = []
        e, w, L, w2, bb = simulate(ctx, mode, st, cv, lg)
        res[name] = (e, w)
        if lg and cv == 1.0: fl.append(pd.DataFrame(lg).assign(variant=name))
        for pset in (PER_M, PER_W):
            for per, lo, hi in pset:
                if fam == "W" and pset is PER_M: continue
                if per.startswith("OOS") and pset is PER_W and fam != "W": continue
                r = dict(variant=name, family=fam, cover=cv, period=per, **stats(e, w, w2, lo, hi))
                if len(L):
                    Lp = L[(L.exit >= lo) & (L.exit < hi)]
                    r.update(legs=len(Lp), win=(Lp.pnl_pts > 0).mean(), itm=int(Lp.itm.sum()), prem=Lp.prem.mean(),
                             prem_pct=Lp.prem_pct.mean(), leg_pnl=Lp.pnl_pts.mean(), bb_fallback=bb)
                rows.append(r)
        if len(L): legs.append(L.assign(variant=name))
    T = pd.DataFrame(rows); T.to_csv(OUT / "cc2_summary.csv", index=False)
    pd.concat(legs).to_csv(OUT / "cc2_legs.csv", index=False)
    FL = pd.concat(fl); FL.to_csv(OUT / "cc2_fills.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    V = T.copy()
    for k in ("CAGR", "MDD_intra", "MDD_intra_cap", "MDD_close", "win", "prem_pct"): V[k] = (V[k] * 100).round(2)
    print(V[(V.cover == 1.0) | (V.family == "-")].drop(columns=["family", "cover"]).round(2).to_string(index=False))
    print("\n部分掩護（FULL）"); print(V[(V.cover < 1) & V.period.str.startswith("FULL")][["variant", "period", "CAGR", "Sharpe", "MDD_intra", "MDD_close"]].round(2).to_string(index=False))
    print("\nfills"); print(FL.groupby([FL.variant.str.split().str[0], "fill"]).size().unstack(fill_value=0))
    print(FL[FL.fill != "open"].drop_duplicates(["expiry"])[["variant", "expiry", "fill", "delay"]].to_string())
    chart(res, T)


def chart(res, T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, ax = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw=dict(height_ratios=[3, 2]))
    keys = ["TXF B&H", "M0 +5% x4", "M0 ATM x4", "B1 +5% x4", "B2 +5% x4", "W +3% x4", "W +1% x4", "W ATM x4"]
    for a, lo, ttl in ((ax[0, 0], START, "Equity 2010-2026/08 (log)"), (ax[0, 1], W0, "Equity 2012/11-2026/08, rebased (log)")):
        for k in keys:
            e = res[k][0]; e = e[e.index >= lo]
            if k.startswith("W") and lo == START: continue
            a.plot(e.index, e / e.iloc[0], lw=1.6 if k == "TXF B&H" else 1, label=k)
        a.set_yscale("log"); a.set_title(ttl); a.legend(fontsize=8); a.grid(alpha=.2)
    sel = T[(T.cover == 1.0) | (T.family == "-")]
    for a, per, ttl in ((ax[1, 0], "FULL 2012/11-26/08", "2012/11-2026/08: CAGR / MDD intraday (%)"),):
        s = sel[sel.period == per].set_index("variant"); xs = np.arange(len(s))
        a.bar(xs - .2, s.CAGR * 100, .4, label="CAGR"); a.bar(xs + .2, s.MDD_intra * 100, .4, label="MDD intraday")
        a.set_xticks(xs); a.set_xticklabels(s.index.str.replace(" x4", ""), rotation=60, fontsize=7); a.legend(); a.grid(alpha=.2); a.set_title(ttl)
    a = ax[1, 1]; s = sel[sel.period == "FULL 2012/11-26/08"].set_index("variant"); xs = np.arange(len(s))
    a.bar(xs, s.Sharpe, color=["#8dd3c7" if v == "TXF B&H" else "#fb8072" if v.startswith("W") else "#80b1d3" for v in s.index])
    a.set_xticks(xs); a.set_xticklabels(s.index.str.replace(" x4", ""), rotation=60, fontsize=7); a.grid(alpha=.2); a.set_title("Sharpe 2012/11-2026/08")
    fig.tight_layout(); fig.savefig(OUT / "cc2_equity_bars.png", dpi=120); print("chart", OUT / "cc2_equity_bars.png")


if __name__ == "__main__":
    main()
