"""51 台指期 covered call + 波動率濾網：1 口 TXF 多單，只在高波動月份賣 5% 價外月選 call。

沿用 45（資料、換月、TXO OHLC 定價、成本、盤中最差 MDD：期貨當日低 + call 當日高）。
訊號：每個月選結算日 E 收盤，用 TXF 日報酬（換月日用 開→收，避免價差跳空）算 20 日實現波動率（年化），
  與「過去 252 交易日的滾動 20 日 RV」中位數比較（只用 ≤E 的資料，無前視）。
  RV20 > 中位數 → E+1 開盤賣 1 份（4 口 TXO = 1 口 TXF 完全掩護）次月月選 call，
  履約價 = ≥ F_E×1.05 的最小履約價；持有至結算。否則該月只持有期貨。
比較：(a) TXF B&H (b) 每月都賣 +5% (c) 高波動才賣 (d) 反向：低波動才賣。
"""
import sys, importlib.util, math
from pathlib import Path
import numpy as np, pandas as pd

_p = Path(__file__).with_name("45_txf_covered_call.py")
_s = importlib.util.spec_from_file_location("cc45", _p); cc = importlib.util.module_from_spec(_s); _s.loader.exec_module(cc)
OUT = Path("results")


def make_pick(allow):
    """回傳取代 cc.pick_leg 的函式：allow(cycle)=False 則不賣；履約價取 ≥ 目標的最小值。"""
    def pick(F, O, cyc, mode, x, optdays, log):
        if not allow(cyc):
            return None
        days = F.index; target = cyc["F_sig"] * (1 + x)
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
                    ks = np.sort(live.index.values.astype(float))
                    up = ks[ks >= target - 1e-9]
                    K = up[0] if len(up) else ks[-1]
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
        log.append(dict(expiry=cyc["expiry"], fill=kind, delay=ix - cyc["entry_ix"]))
        return dict(K=K, prem=px, ix=ix, ser=sub.xs(K, level="strike"), kind=kind)
    return pick


def vol_signal(F):
    c, o, roll = F.close, F.open, F["roll"].astype(bool)
    r = np.log(c / c.shift()); r[roll] = np.log(c / o)[roll]
    rv = r.rolling(20).std() * math.sqrt(252)
    med = rv.rolling(252, min_periods=252).median()
    return rv, med


def main():
    OUT.mkdir(exist_ok=True)
    F = cc.load_fut()
    # 波動率需要 E 之前的長歷史：另讀完整日 K（同一檔）
    full = pd.read_parquet(cc.DATA); full.index = full["date"].dt.tz_localize(None).dt.normalize()
    cm = full.contractMonth.astype(str); full["roll"] = (cm.ne(cm.shift()) & cm.shift().notna()).values
    rv, med = vol_signal(full[full.index <= cc.END])
    O = cc.load_opt(); cycles, optdays = cc.build_cycles(F, O)
    for cy in cycles:
        cy["rv"] = rv.get(cy["E"], np.nan); cy["med"] = med.get(cy["E"], np.nan)
    hi = lambda cy: not np.isnan(cy["med"]) and cy["rv"] > cy["med"]
    lo = lambda cy: not np.isnan(cy["med"]) and cy["rv"] <= cy["med"]
    nan_n = sum(np.isnan(cy["med"]) for cy in cycles)
    print("cycles", len(cycles), "無中位數(暖機不足)月數", nan_n)
    V = {"(a) TXF B&H": None, "(b) 每月賣+5%": lambda cy: True, "(c) 高波動才賣+5%": hi, "(d) 低波動才賣+5%": lo}
    res = {}; orig = cc.pick_leg
    for nm, f in V.items():
        if f is None:
            res[nm] = cc.simulate(F, O, cycles, optdays)
        else:
            cc.pick_leg = make_pick(f)
            res[nm] = cc.simulate(F, O, cycles, optdays, ("+5%", "pct", .05), 1.0, [])
    cc.pick_leg = orig
    bh = res["(a) TXF B&H"][0]
    rows = []
    for nm, (e, w, L, w2) in res.items():
        for per, a, b in cc.PERIODS:
            st = cc.stats(e, w, a, b, w2)
            m = (e.index >= a) & (e.index < b)
            # call 貢獻 ≈ 策略與 B&H 的年化報酬差（同期貨部位、同再平衡）
            st["CAGR_diff_vs_BH"] = st["CAGR"] - cc.stats(bh, res["(a) TXF B&H"][1], a, b, res["(a) TXF B&H"][3])["CAGR"]
            if len(L):
                Lp = L[(L.settle >= a) & (L.settle < b)]
                st.update(months=len(Lp), itm=int(Lp.itm.sum()), call_pts_sum=Lp.pnl_pts.sum(),
                          call_pts_avg=Lp.pnl_pts.mean(), win=(Lp.pnl_pts > 0).mean())
            else:
                st.update(months=0)
            rows.append(dict(variant=nm, period=per, **st))
    T = pd.DataFrame(rows); T.to_csv(OUT / "cc_volfilter_summary.csv", index=False)
    pd.concat([r[2].assign(variant=k) for k, r in res.items() if len(r[2])]).to_csv(OUT / "cc_volfilter_legs.csv", index=False)
    pd.DataFrame([{k: cy[k] for k in ("E", "expiry", "F_sig", "rv", "med")} for cy in cycles]).to_csv(OUT / "cc_volfilter_signal.csv", index=False)
    pd.set_option("display.width", 250)
    fmt = T.copy()
    for k in ("CAGR", "vol", "MDD_close", "MDD_intraday", "MDD_intraday_cap", "CAGR_diff_vs_BH", "win"):
        fmt[k] = (fmt[k] * 100).round(2)
    print(fmt.drop(columns=["Calmar"]).round(2).to_string(index=False))
    # 濾網條件下 call 損益分佈
    L = res["(b) 每月賣+5%"][2].copy()
    sig = pd.DataFrame([dict(expiry=cy["expiry"], rv=cy["rv"], med=cy["med"]) for cy in cycles])
    L = L.merge(sig, on="expiry"); L["hiv"] = np.where(L.med.isna(), "warmup", np.where(L.rv > L.med, "high", "low"))
    L["oos"] = L.settle >= pd.Timestamp("2020-01-01")
    print("\n(b) 各月 call 損益點數，按波動狀態："); print(L.groupby(["oos", "hiv"]).pnl_pts.agg(["count", "mean", "sum"]).round(1))
    chart(res)


def chart(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    lab = {"(a) TXF B&H": "a TXF B&H", "(b) 每月賣+5%": "b always sell +5% call",
           "(c) 高波動才賣+5%": "c sell +5% only if RV20>median", "(d) 低波動才賣+5%": "d sell +5% only if RV20<=median"}
    for k, (e, w, _, _) in res.items():
        ax[0].plot(e.index, e / e.iloc[0], lw=1.3, label=lab[k])
        ax[1].plot(e.index, (w / e.cummax().shift() - 1) * 100, lw=.7, label=lab[k])
    ax[0].set_yscale("log"); ax[0].axvline(pd.Timestamp("2020-01-01"), color="gray", ls="--")
    ax[0].set_title("TXF covered call, vol filter (unlevered, log)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_title("Intraday-worst drawdown %"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "cc_volfilter.png", dpi=120); print("chart", OUT / "cc_volfilter.png")


if __name__ == "__main__":
    main()
