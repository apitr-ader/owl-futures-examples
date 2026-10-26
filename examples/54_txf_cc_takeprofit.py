"""54 台指期 covered call + 停利：short call 收盤 ≤ (1−tp)×收到權利金 → 隔日開盤買回。

沿用 45/51/53：收盤訊號、隔日開盤成交、成本（30 元/口/邊、期交稅 0.1%×權利金、1 tick 滑價）、盤中最差 MDD。
買回：隔日 call day_open（無則 day_close，再無則 max(前值, 內含價值)）+ 1 tick。
停利後：預設只持有期貨到下次月選換倉；resell 變體：同日開盤以「訊號日收盤 F×(1+OTM)」賣同到期月新 call。
停利等級：tp ∈ {0.5, 0.6, 0.7, 0.8}（0.7 = 已賺 70% 權利金）。
"""
import importlib.util, math
from pathlib import Path
import numpy as np, pandas as pd

def _load(n, f):
    s = importlib.util.spec_from_file_location(n, Path(__file__).with_name(f)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cc = _load("cc45", "45_txf_covered_call.py"); vf = _load("vf51", "51_txf_cc_volfilter.py"); iv53 = _load("iv53", "53_txf_cc_iv_trend.py")
OUT = Path("results"); RF = cc.RF; tick = cc.tick
PICK = vf.make_pick(lambda c: True)


def simulate(F, O, cycles, optdays, decide=None, tp=None, resell=False):
    """decide(cy)->OTM% 或 None；tp=None 不停利。回傳 eq, worst, legs, tp_events。"""
    days = F.index; n = len(days)
    o, h, l, c, roll = (F[k].values for k in ("open", "high", "low", "close", "roll"))
    start_ix = cycles[0]["entry_ix"]; ent = {cy["entry_ix"]: cy for cy in cycles}
    eq = np.full(n, np.nan); worst = np.full(n, np.nan)
    q = 0.0; legs = []; tps = []; leg = None; pend = None; mark = 0.0; ncall = 0.0; buyback = False
    E = o[start_ix] * 200.0; rfd = RF / 252
    def opx(s, d, col):
        return s.loc[d, col] if d in s.index else np.nan
    for t in range(start_ix, n):
        d = days[t]; pc = c[t - 1]; pnl = 0.0
        pnl += 0.0 if roll[t] else q * 200 * (o[t] - pc)
        if roll[t] and q: pnl -= q * 2 * (cc.COMM + 200 * (cc.TAX * o[t] + cc.SLIP))
        open_mark = mark; opt_c = opt_w = 0.0
        # A) 停利買回（開盤）
        if buyback and leg is not None:
            s = leg["ser"]; po = opx(s, d, "day_open")
            if np.isnan(po): po = opx(s, d, "day_close")
            if np.isnan(po): po = max(mark, max(o[t] - leg["K"], 0.0)); fk = "fallback"
            else: fk = "open"
            buy = po + tick(po)
            opt_c += -ncall * 50 * (buy - mark); opt_w += -ncall * 50 * (buy - mark)
            pnl -= ncall * (cc.OCOMM + cc.OTAX * buy * 50)
            cy = leg["cy"]; S_set = c[days.get_loc(cy["settle"])]; payoff_set = max(S_set - leg["K"], 0.0)
            tps.append(dict(expiry=cy["expiry"], sig=days[t - 1], fill=d, hold_days=t - leg["ix"], prem=leg["sell"], sig_close=leg["sig_mark"],
                            buy=buy, gap=buy - leg["sig_mark"], K=leg["K"], settle_payoff=payoff_set,
                            avoided_itm=payoff_set > 0, avoided_pts=payoff_set - buy, fk=fk))
            legs.append(dict(expiry=cy["expiry"], settle=d, K=leg["K"], prem=leg["sell"], pnl_pts=leg["sell"] - buy, itm=False, tp=True, F_sig=cy["F_sig"]))
            last_cy = cy; leg = None; ncall = 0.0; mark = 0.0; open_mark = 0.0; buyback = False
            if resell and decide is not None and d < last_cy["settle"]:
                ot = last_cy.get("otm_used")
                ncy = dict(last_cy, entry_ix=t, F_sig=c[t - 1])
                lg = PICK(F, O, ncy, "pct", ot / 100, optdays, [])
                if lg is not None and lg["ix"] < days.get_loc(ncy["settle"]):
                    lg.update(cy=ncy, nc=4 * q); pend = lg
        # B) 月初再平衡 + 決定 call
        if t in ent:
            tq = (E + pnl) / (200 * o[t]); pnl -= abs(tq - q) * (cc.COMM + 200 * (cc.TAX * o[t] + cc.SLIP)); q = tq
            pend = None
            if decide is not None:
                cy = ent[t]; ot = decide(cy)
                if ot is not None:
                    cy["otm_used"] = ot
                    lg = PICK(F, O, cy, "pct", ot / 100, optdays, [])
                    if lg is not None: lg.update(cy=cy, nc=4 * q); pend = lg
        # C) 賣出
        if pend is not None and pend["ix"] == t and leg is None:
            leg = pend; pend = None
            sell = leg["prem"] - tick(leg["prem"]); leg["sell"] = sell; ncall = leg["nc"]
            pnl -= ncall * (cc.OCOMM + cc.OTAX * sell * 50); mark = sell; open_mark = sell
        fut_c = q * 200 * (c[t] - o[t]); fut_l = q * 200 * (l[t] - o[t])
        # D) call MTM
        if leg is not None:
            s = leg["ser"]
            if d == leg["cy"]["settle"]:
                S = c[t]; payoff = max(S - leg["K"], 0.0); dh = opx(s, d, "day_high")
                hi = max(payoff, dh) if not np.isnan(dh) else max(payoff, open_mark)
                hi = min(hi, 2 * max(open_mark, payoff) + max(h[t] - leg["K"], 0.0))
                opt_c += -ncall * 50 * (payoff - open_mark); opt_w += -ncall * 50 * (hi - open_mark)
                if payoff > 0: opt_c -= ncall * cc.OCOMM
                legs.append(dict(expiry=leg["cy"]["expiry"], settle=d, K=leg["K"], prem=leg["sell"], pnl_pts=leg["sell"] - payoff, itm=payoff > 0, tp=False, F_sig=leg["cy"]["F_sig"]))
                leg = None; ncall = 0.0; mark = 0.0
            else:
                dc = opx(s, d, "day_close"); dh = opx(s, d, "day_high"); do_ = opx(s, d, "day_open")
                newm = dc if not np.isnan(dc) else max(mark, max(c[t] - leg["K"], 0.0))
                hi = dh if not np.isnan(dh) else max(newm, open_mark, max(h[t] - leg["K"], 0.0)); hi = max(hi, newm)
                hi = min(hi, 2 * np.nanmax([open_mark, newm, do_]) + max(h[t] - leg["K"], 0.0))
                opt_c += -ncall * 50 * (newm - open_mark); opt_w += -ncall * 50 * (hi - open_mark); mark = newm
                if tp is not None and newm <= (1 - tp) * leg["sell"] and t + 1 < n and days[t + 1] <= leg["cy"]["settle"]:
                    if days[t + 1] < leg["cy"]["settle"]:
                        buyback = True; leg["sig_mark"] = newm
        eq_pre = E + pnl + E * rfd
        worst[t] = eq_pre + fut_l + opt_w
        E = eq_pre + fut_c + opt_c; eq[t] = E
    return pd.Series(eq, days).dropna(), pd.Series(worst, days).dropna(), pd.DataFrame(legs), pd.DataFrame(tps)


def stats(e, w, lo, hi):
    m = (e.index >= lo) & (e.index < hi); e = e[m]; w = w[m]; r = e.pct_change().dropna(); x = r - RF / 252
    pk = e.cummax(); mdd = min((w / pk.shift().fillna(e.iloc[0]) - 1).min(), (e / pk - 1).min())
    return dict(CAGR=(e.iloc[-1] / e.iloc[0]) ** (252 / len(r)) - 1, vol=r.std() * math.sqrt(252),
                Sharpe=x.mean() / x.std() * math.sqrt(252), MDD=mdd)


def main():
    OUT.mkdir(exist_ok=True)
    F = cc.load_fut(); O = cc.load_opt(); cycles, optdays = cc.build_cycles(F, O)
    full = pd.read_parquet(cc.DATA); full.index = full["date"].dt.tz_localize(None).dt.normalize(); full = full[full.index <= cc.END]
    cm = full.contractMonth.astype(str); roll = (cm.ne(cm.shift()) & cm.shift().notna()).values
    r = np.log(full.close / full.close.shift()); r[roll] = np.log(full.close / full.open)[roll]
    rv20 = r.rolling(20).std() * math.sqrt(252); rvmed = rv20.rolling(252, min_periods=252).median()
    iv = iv53.daily_iv(full); ivpct = iv.rolling(252, min_periods=200).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    lc = r.fillna(0).cumsum()
    S = pd.DataFrame(dict(rv=rv20, rvmed=rvmed, iv=iv.reindex(full.index), ivpct=ivpct.reindex(full.index),
                          up20=lc > lc.rolling(20).mean(), up60=lc > lc.rolling(60).mean(), m20=lc - lc.shift(20), m60=lc - lc.shift(60)))
    S["spread"] = S.iv - S.rv; S["ratio"] = S.iv / S.rv
    for cy in cycles: cy["s"] = S.loc[cy["E"]]
    IVF = [("none", None)] + [(f"spread>{t}", ("spread", t / 100)) for t in (0, 2, 4)] + \
          [(f"ratio>{t}", ("ratio", t)) for t in (1.0, 1.2, 1.4)] + [(f"ivpct>{t}", ("ivpct", t)) for t in (30, 50, 70)]
    TRF = [("none", None, None)] + [(f"{k}:{a}", k, a) for k in ("up20", "up60", "m20>0", "m60>0") for a in ("skip", "otm10")]
    def mk(otm, ivf, trk, act):
        def dec(cy):
            s = cy["s"]
            if ivf is not None:
                v = s[ivf[0]]
                if np.isnan(v) or not v > ivf[1]: return None
            if trk is not None:
                on = bool(s[trk]) if trk.startswith("up") else (s[trk[:3]] > 0 if not np.isnan(s[trk[:3]]) else False)
                if on: return None if act == "skip" else max(otm, 10)
            return otm
        return dec
    rvdec = lambda cy: 5 if (not np.isnan(cy["s"].rvmed) and cy["s"].rv > cy["s"].rvmed) else None
    BASE = {"always5": mk(5, None, None, None), "always10": mk(10, None, None, None), "RVfilter5": rvdec,
            "IV+trend(otm5|spread>2|up60:otm10)": mk(5, ("spread", .02), "up60", "otm10")}
    TPS = [("noTP", None, False), ("TP50", .5, False), ("TP60", .6, False), ("TP70", .7, False), ("TP80", .8, False), ("TP70+resell", .7, True)]
    res = {"B&H": simulate(F, O, cycles, optdays)}
    for b, dec in BASE.items():
        for tn, tp, rs in TPS:
            res[f"{b}|{tn}"] = simulate(F, O, cycles, optdays, dec, tp, rs)
    PER = list(cc.PERIODS) + [("WF 2015+", pd.Timestamp("2015-01-01"), cc.END + pd.Timedelta(days=1))]
    rows = []
    for k, (e, w, L, TP) in res.items():
        for pn, lo, hi in PER:
            st = stats(e, w, lo, hi)
            st["CAGR_vs_BH"] = st["CAGR"] - stats(res["B&H"][0], res["B&H"][1], lo, hi)["CAGR"]
            if len(L):
                Lp = L[(L.settle >= lo) & (L.settle < hi)]
                st.update(legs=len(Lp), call_pts=Lp.pnl_pts.sum(), call_pct=(Lp.pnl_pts / Lp.F_sig).sum() * 100, itm=int(Lp.itm.sum()))
            if len(TP):
                Tp = TP[(TP.fill >= lo) & (TP.fill < hi)]
                st.update(tp_n=len(Tp), tp_hold=Tp.hold_days.mean(), tp_avoidITM=int(Tp.avoided_itm.sum()),
                          tp_net_vs_hold_pts=Tp.avoided_pts.sum(), tp_gap_avg=Tp.gap.mean())
            rows.append(dict(name=k, period=pn.split()[0], **st))
    T = pd.DataFrame(rows); T.to_csv(OUT / "cc_tp_summary.csv", index=False)
    pd.concat([v[3].assign(name=k) for k, v in res.items() if len(v[3])]).to_csv(OUT / "cc_tp_events.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    f = T.copy()
    for k in ("CAGR", "vol", "MDD", "CAGR_vs_BH"): f[k] = (f[k] * 100).round(2)
    print(f.round(2).to_string(index=False))
    # tp 事件：避損 vs 放棄
    print("\n== TP70 事件：之後到期價內（避損）vs 放棄剩餘權利金 ==")
    for k in [x for x in res if x.endswith("|TP70")]:
        Tp = res[k][3]
        if not len(Tp): continue
        print(k, "N", len(Tp), "後來價內", int(Tp.avoided_itm.sum()), "| 停利後剩餘:放棄點數(到期價外，buy 價) %.0f" % Tp[~Tp.avoided_itm].buy.sum(),
              "| 避損點數(payoff−buy, 價內) %.0f" % Tp[Tp.avoided_itm].avoided_pts.sum(),
              "| 跳空 avg %.2f max %.1f" % (Tp.gap.mean(), Tp.gap.max()), "| fallback", int((Tp.fk != "open").sum()))
    # walk-forward：A) 參數池 = 53 網格 × {noTP, TP70}；B) 53 網格 WF 選出後套 TP70
    combos = []
    for otm in (5, 7, 10):
        for inm, ivf in IVF:
            for tnm, trk, act in TRF:
                if otm == 10 and act == "otm10": continue
                combos.append((f"otm{otm}|{inm}|{tnm}", mk(otm, ivf, trk, act)))
    WR = {}
    for nm, dec in combos:
        for tn, tp in (("noTP", None), ("TP70", .7)):
            WR[f"{nm}|{tn}"] = simulate(F, O, cycles, optdays, dec, tp)[0].pct_change()
    WR = pd.DataFrame(WR); WR["B&H"] = res["B&H"][0].pct_change()
    for k in ("always5|noTP", "always10|noTP", "RVfilter5|noTP", "RVfilter5|TP70", "always5|TP70", "always10|TP70"): WR[k] = res[k][0].pct_change()
    def wf(pool):
        out = []; picks = []
        for Y in range(2015, cc.END.year + 1):
            tr = WR.loc[f"{Y-5}-01-01":f"{Y-1}-12-31", pool]; x = tr - RF / 252; sh = x.mean() / x.std() * math.sqrt(252); b = sh.idxmax()
            out.append(WR.loc[f"{Y}-01-01":f"{Y}-12-31", b]); picks.append((Y, b))
        return pd.concat(out), picks
    noTP = [c for c in WR if c.endswith("|noTP") and c.startswith("otm")]
    wA, pA = wf(noTP + [c.replace("noTP", "TP70") for c in noTP])
    wN, pN = wf(noTP)
    wB = pd.concat([WR.loc[f"{Y}-01-01":f"{Y}-12-31", b.replace("noTP", "TP70")] for Y, b in pN])
    def rs(rr):
        rr = rr.dropna(); e = (1 + rr).cumprod(); x = rr - RF / 252
        return dict(CAGR=e.iloc[-1] ** (252 / len(rr)) - 1, vol=rr.std() * math.sqrt(252), Sharpe=x.mean() / x.std() * math.sqrt(252), MDD_close=(e / e.cummax() - 1).min())
    idx = wN.index
    W = pd.DataFrame([dict(name="WF 53 (noTP pool)", **rs(wN)), dict(name="WF pool incl TP70", **rs(wA)), dict(name="WF 53 picks + TP70", **rs(wB))] +
                     [dict(name=k, **rs(WR.loc[idx, k])) for k in ("B&H", "always5|noTP", "always5|TP70", "always10|noTP", "always10|TP70", "RVfilter5|noTP", "RVfilter5|TP70")])
    print(f"\n== walk-forward {idx[0].date()} ~ {idx[-1].date()} =="); print(W.assign(**{c: (W[c] * 100).round(2) for c in ("CAGR", "vol", "MDD_close")}).round(3).to_string(index=False))
    print("WF pool-incl-TP picks:", pA)
    chart(res, wN, wA, wB, WR, idx)


def chart(res, wN, wA, wB, WR, idx):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for a, b in zip(ax.flat, ("always5", "always10", "RVfilter5", "IV+trend(otm5|spread>2|up60:otm10)")):
        e = res["B&H"][0]; a.plot(e.index, e / e.iloc[0], c="k", lw=1.5, label="B&H")
        for tn in ("noTP", "TP50", "TP70", "TP80", "TP70+resell"):
            e = res[f"{b}|{tn}"][0]; a.plot(e.index, e / e.iloc[0], lw=1, label=tn)
        a.set_yscale("log"); a.set_title(b); a.legend(fontsize=8); a.grid(alpha=.3); a.axvline(pd.Timestamp("2020-01-01"), c="gray", ls="--")
    fig.tight_layout(); fig.savefig(OUT / "cc_tp_equity.png", dpi=110)
    fig, a = plt.subplots(figsize=(13, 6))
    for nm, rr in (("WF 53 noTP", wN), ("WF pool incl TP70", wA), ("WF 53 picks + TP70", wB), ("B&H", WR.loc[idx, "B&H"]), ("always10 noTP", WR.loc[idx, "always10|noTP"])):
        e = (1 + rr.fillna(0)).cumprod(); a.plot(e.index, e, lw=1.6 if nm == "B&H" else 1.1, label=nm)
    a.set_yscale("log"); a.legend(); a.grid(alpha=.3); a.set_title("Walk-forward 2015+ (yearly pick by prior-5y Sharpe)")
    fig.tight_layout(); fig.savefig(OUT / "cc_tp_wf.png", dpi=110)
    print("charts results/cc_tp_equity.png results/cc_tp_wf.png")


if __name__ == "__main__":
    main()
