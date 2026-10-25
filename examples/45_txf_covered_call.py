"""45 台指期 covered call：1 口 TXF 多單 + 每月賣出 TXO 月選 call（ATM / 價外）vs 單純 TXF B&H（日K）。

時序（Kevin 規則：收盤算訊號、隔日開盤成交）：
- 月選結算日 E（該月合約最後交易日）收盤：以 TXF 收盤 F_E 決定下個月 call 履約價（ATM 或 +N 點 / +x%）。
- E+1 日盤開盤（day_open）賣出次月月選 call（只用 YYYYMM 標準月合約）；同時以開盤調整 TXF 口數。
  無 day_open → 用當日 day_close（盤中成交）；當日完全無成交/缺資料 → 順延至下一個有開盤的交易日。
- 持有至該月結算日。最後結算價近似：結算日 TXF（仍為到期月合約）日盤收盤
  （官方為收盤前 30 分鐘指數平均）。call 收益 = 權利金 − max(S−K,0)。
資本：名目、無槓桿。每次換月（E+1 開盤）把 TXF 口數設為 權益/(200×開盤價)（可為小數），
  call 口數 = cover × 4 × TXF 口數（cover=1 即 4 口 call 對 1 口大台 = 完全掩護）。現金賺 rf 1.5%/年。
成本：期貨沿用 37：50 元/口/邊、期交稅 2e-5、滑價 1 點；連續合約換月：持有口數兩邊成本，換月日只算新合約開到收。
  選擇權：手續費 30 元/口/邊（到期價內結算再收一次）、期交稅 0.1%×權利金、賣出滑價 1 tick（TXO 跳動點）。
MDD_intraday_cap：盤中 call 高點上限 = 2×max(前值, 收, 開) + 期貨高點內含價值（濾掉 2015-08-24 那種 800 點離譜成交）。
MDD：收盤 MTM（call 以 day_close 評價，無成交沿用前值或內含價值較大者）；
  盤中最差：TXF 以當日最低、call 以當日最高（day_high）同時發生。
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
PERIODS = (("FULL", START, END + pd.Timedelta(days=1)), ("IS 2010-19", START, pd.Timestamp("2020-01-01")),
           ("OOS 2020-26/08", pd.Timestamp("2020-01-01"), END + pd.Timedelta(days=1)))
STRIKES = [("ATM", "pt", 0), ("+100", "pt", 100), ("+200", "pt", 200), ("+300", "pt", 300), ("+500", "pt", 500),
           ("+1%", "pct", .01), ("+2%", "pct", .02), ("+3%", "pct", .03), ("+5%", "pct", .05)]
COVERS = (1.0, 0.5, 0.25)


def tick(p):
    return 0.1 if p < 10 else 0.5 if p < 50 else 1 if p < 500 else 5 if p < 1000 else 10


def load_fut():
    d = pd.read_parquet(DATA)
    d.index = d["date"].dt.tz_localize(None).dt.normalize()
    d = d[(d.index >= START - pd.Timedelta(days=10)) & (d.index <= END)]
    cm = d.contractMonth.astype(str)
    d["roll"] = (cm.ne(cm.shift()) & cm.shift().notna()).values
    return d[["open", "high", "low", "close", "roll"]].astype(float).assign(roll=d.roll)


def load_opt():
    o = load_txo_daily(START - pd.Timedelta(days=40), END, expiry="monthly", cp="C",
                       columns=["date", "expiry", "strike", "cp", "day_open", "day_high", "day_low", "day_close", "day_volume"])
    o["date"] = pd.to_datetime(o.date).dt.normalize()
    return o.set_index(["expiry", "strike", "date"]).sort_index()


def build_cycles(F, O):
    """月選結算日序列與各月份每個履約價設定的賣出 call（含 fallback 記錄）。"""
    e = expiries(); e = e[(e.kind == "monthly")]
    exp = e[(e.expiry_date >= F.index[0]) & (e.last_date < END)].sort_values("expiry_date")
    days = F.index
    optdays = set(O.index.get_level_values("date").unique())
    cycles = []
    for i in range(len(exp) - 1):
        E = exp.expiry_date.iloc[i]
        if E not in days:
            continue
        nxt = exp.iloc[i + 1]
        p = days.searchsorted(E) + 1
        if p >= len(days) or nxt.expiry_date > END:
            break
        cycles.append(dict(E=E, entry_ix=p, expiry=nxt.expiry, settle=nxt.expiry_date, F_sig=F.close.loc[E]))
    return cycles, optdays


def pick_leg(F, O, cyc, mode, x, optdays, log):
    days = F.index
    target = cyc["F_sig"] + x if mode == "pt" else cyc["F_sig"] * (1 + x)
    try:
        sub = O.loc[cyc["expiry"]]
    except KeyError:
        return None
    ix = cyc["entry_ix"]; kind = "open"
    while True:
        d = days[ix]
        if d in optdays:
            day = sub.xs(d, level="date") if d in sub.index.get_level_values("date") else sub.iloc[:0]
            live = day[day.day_open.notna() | day.day_close.notna()]
            if len(live):
                ks = live.index.values.astype(float)
                K = ks[np.argmin(np.abs(ks - target) - (ks >= target) * 1e-6)]  # 等距時取較高
                row = live.loc[K]
                if np.isnan(row.day_open):
                    px = row.day_close; kind = "close_fallback"
                else:
                    px = row.day_open
                break
            kind = "next_day"
        else:
            kind = "missing_day_next"
        ix += 1
        if days[ix] >= cyc["settle"]:
            return None
    s = sub.xs(K, level="strike")
    log.append(dict(expiry=cyc["expiry"], fill=kind, delay=ix - cyc["entry_ix"]))
    return dict(K=K, prem=px, ix=ix, ser=s, kind=kind)


def simulate(F, O, cycles, optdays, strike=None, cover=0.0, log=None):
    """strike=None → 單純 B&H（同樣每月 E+1 開盤再平衡）。回傳每日 close/worst 權益與每月 call 記錄。"""
    days = F.index; n = len(days)
    o, h, l, c, roll = (F[k].values for k in ("open", "high", "low", "close", "roll"))
    start_ix = cycles[0]["entry_ix"]
    eq = np.full(n, np.nan); worst = np.full(n, np.nan); worst2 = np.full(n, np.nan)
    E = 1.0; q = 0.0; legs = []
    ent = {cy["entry_ix"]: cy for cy in cycles}
    leg = None; pend = None; mark = 0.0; ncall = 0.0
    sidx = {cy["settle"]: None for cy in cycles}
    rfd = RF / 252
    # 以「點×乘數」計價，權益單位：元，初始 = 1 口名目
    E = o[start_ix] * 200.0
    for t in range(start_ix, n):
        d = days[t]; pc = c[t - 1]
        pnl = 0.0; wpnl = 0.0
        # 1) 開盤前：舊部位 前收→開（換月日跳空不計）
        gap = 0.0 if roll[t] else q * 200 * (o[t] - pc)
        pnl += gap
        if roll[t] and q:
            pnl -= q * 2 * (COMM + 200 * (TAX * o[t] + SLIP))
        # 2) 月初再平衡 + 決定待賣 call
        if t in ent:
            tq = (E + pnl) / (200 * o[t])
            if t == start_ix or True:
                dq = abs(tq - q); pnl -= dq * (COMM + 200 * (TAX * o[t] + SLIP)); q = tq
            if strike is not None:
                cy = ent[t]
                lg = pick_leg(F, O, cy, strike[1], strike[2], optdays, log if log is not None else [])
                if lg is not None:
                    lg.update(cy=cy, nc=cover * 4 * q)
                    pend = lg
        # 3) 賣出 call（可能順延）
        if pend is not None and pend["ix"] == t:
            leg = pend; pend = None
            sell = leg["prem"] - tick(leg["prem"])
            leg["sell"] = sell; ncall = leg["nc"]
            pnl -= ncall * (OCOMM + OTAX * sell * 50)
            mark = sell  # 以成交價為基準 MTM
            open_mark = sell
        else:
            open_mark = mark
        # 4) 期貨 開→收 / 開→低
        fut_c = q * 200 * (c[t] - o[t]); fut_l = q * 200 * (l[t] - o[t])
        # 5) call MTM
        opt_c = opt_w = opt_w2 = 0.0
        if leg is not None:
            s = leg["ser"]
            if d == leg["cy"]["settle"]:
                S = c[t]; payoff = max(S - leg["K"], 0.0)
                dh = s.loc[d, "day_high"] if d in s.index else np.nan
                hi = max(payoff, dh) if not np.isnan(dh) else max(payoff, open_mark)
                opt_c = -ncall * 50 * (payoff - open_mark)
                opt_w = -ncall * 50 * (hi - open_mark)
                opt_w2 = -ncall * 50 * (min(hi, 2 * max(open_mark, payoff) + max(h[t] - leg["K"], 0.0)) - open_mark)
                if payoff > 0:
                    opt_c -= ncall * OCOMM
                legs.append(dict(expiry=leg["cy"]["expiry"], entry=days[leg["ix"]], settle=d, F_sig=leg["cy"]["F_sig"], K=leg["K"],
                                 prem=leg["sell"], S=S, payoff=payoff, pnl_pts=leg["sell"] - payoff, itm=payoff > 0,
                                 fill=leg["kind"], prem_pct=leg["sell"] / leg["cy"]["F_sig"]))
                leg = None; ncall = 0.0; mark = 0.0
            else:
                dc = s.loc[d, "day_close"] if d in s.index else np.nan
                dh = s.loc[d, "day_high"] if d in s.index else np.nan
                intrinsic = max(c[t] - leg["K"], 0.0)
                newm = dc if not np.isnan(dc) else max(mark, intrinsic)
                hi = dh if not np.isnan(dh) else max(newm, open_mark, max(h[t] - leg["K"], 0.0))
                hi = max(hi, newm)
                do_ = s.loc[d, "day_open"] if d in s.index else np.nan
                cap = 2 * np.nanmax([open_mark, newm, do_]) + max(h[t] - leg["K"], 0.0)
                opt_w2 = -ncall * 50 * (min(hi, cap) - open_mark)
                opt_c = -ncall * 50 * (newm - open_mark); opt_w = -ncall * 50 * (hi - open_mark)
                mark = newm
        intr = E * rfd
        eq_pre = E + pnl + intr
        worst[t] = eq_pre + fut_l + opt_w
        worst2[t] = eq_pre + fut_l + opt_w2
        E = eq_pre + fut_c + opt_c
        eq[t] = E
    s = pd.Series(eq, days).dropna(); w = pd.Series(worst, days).dropna(); w2 = pd.Series(worst2, days).dropna()
    return s, w, pd.DataFrame(legs), w2


def stats(eq, wq, lo, hi, wq2=None):
    m = (eq.index >= lo) & (eq.index < hi)
    e = eq[m]; w = wq[m]; w2 = wq2[m]
    r = e.pct_change().dropna()
    y = len(r) / 252
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / y) - 1
    x = r - RF / 252
    pk = e.cummax()
    mdd_c = (e / pk - 1).min()
    mdd_w = (w / pk.shift().fillna(e.iloc[0]).clip(lower=0) - 1).min()
    mdd_w = min(mdd_w, mdd_c)
    mdd_w2 = min((w2 / pk.shift().fillna(e.iloc[0]) - 1).min(), mdd_c)
    return dict(CAGR=cagr, vol=r.std() * math.sqrt(252), Sharpe=x.mean() / x.std() * math.sqrt(252),
                MDD_close=mdd_c, MDD_intraday=mdd_w, MDD_intraday_cap=mdd_w2, Calmar=cagr / abs(mdd_w))


def main():
    OUT.mkdir(exist_ok=True)
    F = load_fut(); O = load_opt()
    cycles, optdays = build_cycles(F, O)
    print("cycles", len(cycles), cycles[0]["E"].date(), cycles[-1]["settle"].date())
    res = {}; rows = []; legrows = []; fills = []
    bh = simulate(F, O, cycles, optdays)
    res["TXF B&H"] = bh
    for nm, md, x in STRIKES:
        for cv in COVERS:
            lg = []
            r = simulate(F, O, cycles, optdays, (nm, md, x), cv, lg)
            name = f"CC {nm} x{int(cv*4)}"
            res[name] = r
            if cv == 1.0:
                fl = pd.DataFrame(lg); fl["strike"] = nm; fills.append(fl)
    for name, (e, w, L, w2) in res.items():
        for per, lo, hi in PERIODS:
            st = stats(e, w, lo, hi, w2)
            if len(L):
                Lp = L[(L.settle >= lo) & (L.settle < hi)]
                st.update(months=len(Lp), call_win=(Lp.pnl_pts > 0).mean(), itm_months=int(Lp.itm.sum()),
                          avg_prem_pts=Lp.prem.mean(), avg_prem_pct=Lp.prem_pct.mean(), avg_call_pnl_pts=Lp.pnl_pts.mean())
            rows.append(dict(variant=name, period=per, **st))
        if len(L):
            legrows.append(L.assign(variant=name))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "cc_summary.csv", index=False)
    pd.concat(legrows).to_csv(OUT / "cc_legs.csv", index=False)
    FL = pd.concat(fills); FL.to_csv(OUT / "cc_fills.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    show = T[T.variant.str.endswith("x4") | (T.variant == "TXF B&H")]
    fmt = show.copy()
    for k in ("CAGR", "vol", "MDD_close", "MDD_intraday", "MDD_intraday_cap", "call_win", "avg_prem_pct"):
        fmt[k] = (fmt[k] * 100).round(2)
    print(fmt.round(2).to_string(index=False))
    print("\n部分掩護（x2 / x1）FULL：")
    print(T[(T.period == "FULL") & ~T.variant.str.endswith("x4")].round(3).to_string(index=False))
    print("\nfill 統計（每個履約價設定 x4）："); print(FL.groupby(["strike", "fill"]).size().unstack(fill_value=0))
    chart(res, T)


def chart(res, T):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    for f in ("Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
        try:
            from matplotlib import font_manager as fm
            fm.findfont(f, fallback_to_default=False); plt.rcParams["font.family"] = f; break
        except Exception:
            pass
    fig, ax = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw=dict(height_ratios=[3, 2]))
    a = ax[0, 0]
    keys = ["TXF B&H", "CC ATM x4", "CC +200 x4", "CC +500 x4", "CC +2% x4", "CC +5% x4"]
    for k in keys:
        e = res[k][0]; a.plot(e.index, e / e.iloc[0], lw=1.6 if k == "TXF B&H" else 1.1, label=k)
    a.set_yscale("log"); a.set_title("Equity (unlevered, notional = 1 TXF, log)"); a.legend(fontsize=8); a.grid(alpha=.2)
    a = ax[0, 1]
    for k in keys:
        e, w, _, _ = res[k]; a.plot(e.index, (w / e.cummax().shift() - 1) * 100, lw=.8, label=k)
    a.set_title("Intraday-worst drawdown % (TXF low + call high)"); a.grid(alpha=.2); a.legend(fontsize=7)
    full = T[(T.period == "FULL") & (T.variant.str.endswith("x4") | (T.variant == "TXF B&H"))].set_index("variant")
    xs = np.arange(len(full))
    a = ax[1, 0]
    a.bar(xs - .2, full.CAGR * 100, .4, label="CAGR %"); a.bar(xs + .2, full.MDD_intraday * 100, .4, label="MDD intraday %")
    a.set_xticks(xs); a.set_xticklabels(full.index.str.replace(" x4", ""), rotation=45, fontsize=8); a.legend(); a.grid(alpha=.2)
    a.set_title("FULL 2010-2026/08: CAGR vs intraday MDD")
    a = ax[1, 1]
    for i, per in enumerate(T.period.unique()):
        s = T[(T.period == per) & (T.variant.str.endswith("x4") | (T.variant == "TXF B&H"))].set_index("variant").Sharpe
        a.bar(xs + (i - 1) * .27, s.values, .27, label=per)
    a.set_xticks(xs); a.set_xticklabels(full.index.str.replace(" x4", ""), rotation=45, fontsize=8); a.legend(); a.grid(alpha=.2)
    a.set_title("Sharpe by period")
    fig.tight_layout(); fig.savefig(OUT / "cc_equity_bars.png", dpi=120)
    print("chart", OUT / "cc_equity_bars.png")


if __name__ == "__main__":
    main()
