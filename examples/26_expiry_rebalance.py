"""26 個股期貨策略：在每月到期結算日換倉（持有到到期、以最後結算價結算，新倉開在新近月）

時序（無前視）：
  到期日 x（第三個週三；遇假日順延，以資料中「多數契約換月的日子」前一交易日為準，見 expiry_calendar）。
  x 收盤後計算訊號（只用 ≤ x 收盤的資料：調整價、量、TXF）。舊部位在 x 以最後結算價結算 → 以 x 收盤價近似
  （真正最後結算價為標的股票收盤前 60 分鐘均價；期貨收盤已收斂到現貨，誤差小，無方向偏差）。
  x+1（新近月第一天）開盤以新近月開盤價建新倉。x 收盤 → x+1 開盤之間空手。
  若選中的契約在 x+1 沒有成交（不在當天換月），該槽位空手（不知道它沒成交才換別檔，避免前視）。
  這樣不再有換月成本：每月成本 = 到期結算（手續費 + 期交稅，無滑價）＋ 新倉開倉（手續費 + 稅 + 滑價）。
成本假設：
  期貨交易稅 2/100000 於到期結算時依最後結算價課徵（期交稅條例：到期結算亦課稅）。
  手續費：多數期貨商對到期結算仍收一筆手續費（與平倉同），保守假設每口 50 元也適用於結算。
  滑價：開倉每邊 1 檔（敏感度 0.5 檔）；結算無滑價。
節奏：每月到期（cad=1）或每 3 個到期（cad=3，季）。季節奏下中間到期仍必須結算並在新近月重開同名同權重
  （無法續抱過到期），這部分照樣計成本。波動目標只在換倉（到期）時調整；月中不交易。
策略（5 檔）：DM（21 原 pick、25 DM re-pick、重挑）、DMS（lb 5/10/15/20 各自 plateau 挑選）、REV、RRES。
挑參：IS 2014-2019 plateau（IS 平均空槽 < 30%），OOS 2020-2026/10，滑價 1 檔下挑選，再看 0.5 檔。
products.csv 只含現存契約 → 倖存者偏差（結果偏樂觀）。
"""
import argparse
import importlib.util
import itertools
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("rr24", Path(__file__).with_name("24_reversal_residual_momentum.py"))
m24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m24)
o22, dm, D = m24.o22, m24.dm, m24.D
IS0, IS1, EX1, END = m24.IS0, m24.IS1, m24.EX1, m24.END
PERIODS = (("IS", IS0, IS1), ("OOS", IS1, END), ("OOS_ex2526", IS1, EX1))
N = 5
SL = [1.0]
SC = ["EW", "IV", "MV", "ERC"]
G = {
    "DM": dict(lb=[63, 126, 252], skip=[0, 21], cad=[1, 3], buf=[0, 1], mkt=[0, 1], vt=[0.0, 0.15], scheme=SC),
    "DMS": dict(lb=[5, 10, 15, 20], skip=[0, 1], cad=[1, 3], buf=[0, 1], mkt=[0, 1], vt=[0.0, 0.15], scheme=SC),
    "REV": dict(lb=[5, 10, 20], skip=[0, 1], cad=[1, 3], buf=[0, 1], liq=[0, 1], mkt=[0, 1], vt=[0.0, 0.15], scheme=SC),
    "RRES": dict(lb=[63, 126, 252], skip=[0, 21], win=[252], scaled=[0, 1], absf=["raw", "res"], cad=[1, 3],
                 mkt=[0, 1], vt=[0.0, 0.15], scheme=SC),
}


def expiry_calendar():
    """新近月第一天 = 多數（>30%）有成交契約同日換月的日子；到期日 = 其前一交易日。"""
    roll, tr = D["M"]["ROLL"].values, D["M"]["traded"].values
    frac = roll.sum(1) / np.maximum(tr.sum(1), 1)
    first = np.nonzero(frac > 0.3)[0]
    return [int(k) - 1 for k in first if k >= 1]


def score(fam, P, i, U):
    if fam in ("DM", "DMS"):
        m = D["I"][i - P["skip"], U] / D["I"][i - P["lb"], U] - 1
        return m, m > 0
    return (m24.score_rev if fam == "REV" else m24.score_res)(P, i, U)


def simulate(fam, P, lo=None, hi=None):
    days, mult, A, O, R, RO = D["days"], D["mult"], D["A"], D["O"], D["R"], D["RO"]
    roll = D["M"]["ROLL"].values
    slip = SL[0]
    X = D["exp"]
    lo_, hi_ = (IS0 if lo is None else lo), (END if hi is None else hi)
    X = [x for x in X if lo_ - pd.Timedelta(days=40) <= days[x] < hi_]
    nd = len(days)
    minvol, minpx = m24.LIQ[P["liq"]] if fam == "REV" else (50, 0)
    eq = 1.0
    vals = np.full(nd, np.nan)
    turn = c_open = c_set = 0.0
    held, w_held = [], None
    slots = 0
    nreb = 0
    for n, x in enumerate(X):
        k0 = x + 1
        if k0 >= nd:
            break
        k1 = X[n + 1] + 1 if n + 1 < len(X) else nd     # 持有 k0..X[n+1]（含到期日）
        if n == 0 or days[x] < lo_ - pd.Timedelta(days=40):
            pass
        rebal = (n % P["cad"] == 0)
        if rebal:
            nreb += 1
            sel = []
            U = o22.univ(x, minvol, minpx)
            need = max(P["lb"] + 1, P.get("win", 0))
            if len(U) >= dm.MIN_UNIVERSE and not (P["mkt"] and not D["txf_mom"][x] > 0) and x - need >= 0:
                sc, ok = score(fam, P, x, U)
                rk = U[np.argsort(-sc)]
                okd = dict(zip(U, ok))
                pos = {s: r for r, s in enumerate(rk)}
                keep = [s for s in held if s in pos and pos[s] < 2 * N and okd[s]][:N] if P.get("buf") else []
                sel = keep + [s for s in rk if s not in keep and okd[s]][:N - len(keep)]
            slots += N - len(sel)
            sel = sorted(int(s) for s in sel)
            w = o22.wts(x, P["scheme"], tuple(sel), len(sel) / N, P["vt"]) if sel else np.zeros(0)
            held, w_held = sel, w
        sel, w = held, w_held
        # 新近月在 k0 必須有成交（當天換月且有開盤）；否則該槽空手
        okf = [j for j, s in enumerate(sel) if roll[k0, s] and not np.isnan(RO[k0, s])] if sel else []
        q = np.zeros(len(D["cols"]))
        if okf:
            s_ = np.array([sel[j] for j in okf]); w_ = np.asarray(w)[okf]
            q[s_] = w_ * eq / (RO[k0, s_] * mult[s_])
            px = RO[k0, s_]
            c = q[s_] @ (dm.COMMISSION + mult[s_] * (dm.TAX * px + slip * o22.tick_arr(px)))
            turn += (q[s_] * mult[s_] * px).sum() / eq
            eq -= c; c_open += c / eq
            h = s_
            vals[k0:k1] = eq + (A[k0:k1, h] - O[k0, h]) @ (q[h] * mult[h])
            eq = vals[k1 - 1]
            if k1 - 1 < nd and n + 1 < len(X):      # 到期結算（以到期日收盤近似最後結算價）
                sp = np.nan_to_num(R[k1 - 1, h], nan=np.nan_to_num(D["M"]["Rf"].values[k1 - 1, h]))
                c = q[h] @ (dm.COMMISSION + mult[h] * dm.TAX * sp)
                eq -= c; c_set += c / eq
                vals[k1 - 1] = eq
        else:
            vals[k0:k1] = eq
        if n == 0 and k0 > 0:
            vals[k0 - 1] = 1.0
    e = pd.Series(vals, days).dropna()
    r = e.pct_change().fillna(0.0)
    if lo is not None:
        r = r[(r.index >= lo) & (r.index < hi)]
    yrs = len(r) / 252
    return dict(r=r, turn=turn, c_open=c_open, c_set=c_set, cost=c_open + c_set, cash=slots / max(nreb * N, 1), yrs=yrs)


def _work(args):
    fam, P = args
    o = simulate(fam, P)
    row = dict(P)
    for nm, lo, hi in PERIODS:
        row[nm + "_sharpe"] = m24.stats(m24.win(o["r"], lo, hi))["sharpe"]
    row["cash"] = o["cash"]
    return row


def evaluate(fam, P, slip, sr0, variant):
    SL[0] = slip
    full = simulate(fam, P)
    rows = []
    for per, lo, hi in PERIODS:
        o = simulate(fam, P, lo, hi)
        s = m24.stats(m24.win(full["r"], lo, hi))
        y = max(o["yrs"], 1e-9)
        rows.append(dict(strategy=fam, variant=variant, slip=slip, period=per, cfg=" ".join(f"{k}={v}" for k, v in P.items()),
                         CAGR=s["CAGR"], vol=s["vol"], sharpe=s["sharpe"], mdd=s["mdd"], turnover_yr=o["turn"] / y,
                         cost_pct_yr=o["cost"] / y * 100, open_cost_pct_yr=o["c_open"] / y * 100,
                         settle_cost_pct_yr=o["c_set"] / y * 100, roll_cost_pct_yr=0.0, cash_slots=o["cash"],
                         DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan, sr0=sr0))
    SL[0] = 1.0
    return rows, full["r"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stock_futures")
    ap.add_argument("--txf", default="data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    o22.prepare(a.data, a.txf)
    D["txv"] = D["txr"].values; D["txr_s"] = D["txr"]
    D["exp"] = expiry_calendar()
    ex = pd.DataFrame(dict(expiry=D["days"][D["exp"]], new_front_first_day=D["days"][[min(x + 1, len(D["days"]) - 1) for x in D["exp"]]]))
    ex["weekday"] = ex.expiry.dt.day_name()
    ex.to_csv(out / "exp_calendar.csv", index=False)
    print("到期日（非週三）：", ex[ex.weekday != "Wednesday"].expiry.dt.date.tolist())
    pd.set_option("display.width", 250)

    tab = []
    bm = {"TXF B&H": D["txr"]}
    mes = [D["days"].get_loc(t) for t in dm.month_ends(D["days"]) if IS0 - pd.Timedelta(days=40) <= t < END]
    ew = pd.Series(0.0, D["days"])
    for n, i in enumerate(mes[:-1]):
        U = o22.univ(i, 50, 0); k0, k1 = i + 1, mes[n + 1] + 1
        if len(U) >= dm.MIN_UNIVERSE:
            ew.iloc[k0:k1] = D["ret"][k0:k1, U].mean(1)
    bm["EW universe (gross)"] = ew
    for n, r in bm.items():
        for per, lo, hi in PERIODS:
            s = m24.stats(m24.win(r, lo, hi))
            tab.append(dict(strategy=n, variant="benchmark", period=per, CAGR=s["CAGR"], vol=s["vol"], sharpe=s["sharpe"], mdd=s["mdd"]))

    curves, info = {}, {}
    for fam, g in G.items():
        C = [(fam, dict(zip(g, v))) for v in itertools.product(*g.values())]
        C = [c for c in C if not (fam == "REV" and c[1]["skip"] >= c[1]["lb"]) and not (fam == "RRES" and c[1]["lb"] > c[1]["win"])]
        with Pool(a.procs) as pool:
            res = pd.DataFrame(pool.map(_work, C, chunksize=8))
        res["plateau"] = m24.plateau(res, g)
        res.to_csv(out / f"exp_grid_{fam.lower()}.csv", index=False)
        s0 = dm.expected_max_sr(len(res), float(res.IS_sharpe.var()))
        rho = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]
        info[fam] = (len(res), s0, rho)
        print(f"\n== {fam}: 試驗 {len(res)} SR0 {s0:.3f} Spearman(IS,OOS) {rho:.3f}")
        print("IS", res.IS_sharpe.describe().round(3).to_dict()); print("OOS", res.OOS_sharpe.describe().round(3).to_dict())
        el = res[res.cash < 0.3].sort_values(["plateau", "IS_sharpe"], ascending=False)
        picks = [("repick", el.iloc[0])]
        if fam == "DMS":
            picks += [(f"repick_lb{lb}", el[el.lb == lb].iloc[0]) for lb in g["lb"]]
        todo = [(v, {k: (p[k].item() if hasattr(p[k], "item") else p[k]) for k in g}) for v, p in picks]
        if fam == "DM":
            todo += [("orig21_at_expiry", dict(lb=252, skip=21, cad=1, buf=0, mkt=1, vt=0.15, scheme="EW")),
                     ("dm25_repick_at_expiry", dict(lb=252, skip=0, cad=3, buf=1, mkt=0, vt=0.15, scheme="EW"))]
        for v, P in todo:
            for slip in (1.0, 0.5):
                rr, eqr = evaluate(fam, P, slip, s0, v)
                tab += rr
                if slip == 1.0 and v in ("repick", "orig21_at_expiry", "dm25_repick_at_expiry"):
                    curves[f"{fam} {v}"] = eqr
    for r in tab:
        if r.get("strategy") in info:
            r["grid_trials"], r["grid_sr0"], r["grid_spearman"] = info[r["strategy"]]
    tab = pd.DataFrame(tab)
    tab.to_csv(out / "exp_summary.csv", index=False)
    cols = ["strategy", "variant", "slip", "period", "CAGR", "sharpe", "mdd", "turnover_yr", "cost_pct_yr",
            "open_cost_pct_yr", "settle_cost_pct_yr", "DSR"]
    print(tab[cols].round(3).to_string())
    print(tab[tab.variant.str.startswith(("repick", "orig", "dm25"))].drop_duplicates(["strategy", "variant"])[["strategy", "variant", "cfg"]].to_string())

    # 與 25 NEW 成本比較
    c25 = pd.read_csv(out / "cost_summary.csv")
    c25 = c25[(c25.cost_model == "NEW") & (c25.period != "x")]
    m = {("DM", "orig21"): ("DM", "orig21_at_expiry"), ("DM", "repick"): ("DM", "dm25_repick_at_expiry"),
         ("DMS", "repick"): ("DMS", "repick"), ("REV", "repick"): ("REV", "repick"), ("RRES", "repick"): ("RRES", "repick")}
    for lb in (5, 10, 15, 20):
        m[("DMS", f"orig23_lb{lb}")] = ("DMS", f"repick_lb{lb}")
    comp = []
    for (s25, v25), (s26, v26) in m.items():
        for per in ("IS", "OOS", "OOS_ex2526"):
            a_ = c25[(c25.strategy == s25) & (c25.variant == v25) & (c25.period == per)]
            b_ = tab[(tab.strategy == s26) & (tab.variant == v26) & (tab.period == per) & (tab.slip == 1.0)]
            if len(a_) and len(b_):
                a_, b_ = a_.iloc[0], b_.iloc[0]
                comp.append(dict(s25=f"{s25}/{v25}", s26=f"{s26}/{v26}", period=per, sharpe25=a_.sharpe, sharpe26=b_.sharpe,
                                 CAGR25=a_.CAGR, CAGR26=b_.CAGR, cost25=a_.cost_pct_yr, cost26=b_.cost_pct_yr,
                                 mdd25=a_.mdd, mdd26=b_.mdd))
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "exp_vs_cost25.csv", index=False)
    print(comp.round(3).to_string())
    charts(out, tab, curves, bm, comp)


def charts(out, tab, curves, bm, comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for n, r in curves.items():
        r = m24.win(r, IS0, END); plt.plot(r.index, (1 + r).cumprod(), label=n, lw=1.5)
    for n, r in bm.items():
        r = m24.win(r, IS0, END); plt.plot(r.index, (1 + r).cumprod(), label=n, ls="--", c="k" if "TXF" in n else "gray")
    plt.axvline(IS1, color="k", ls="--", lw=.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title("Expiry-rebalanced stock-futures strategies (hold to settlement, open new front month; 1 tick slip)")
    plt.tight_layout(); plt.savefig(out / "exp_equity.png", dpi=110); plt.close()
    t = comp[comp.period == "OOS"].set_index("s26")[["sharpe25", "sharpe26"]]
    t.columns = ["script 25 NEW (month-end, spread roll)", "script 26 expiry rebalance"]
    ax = t.plot.bar(figsize=(11, 5))
    for n, r in bm.items():
        ax.axhline(m24.stats(m24.win(r, IS1, END))["sharpe"], ls="--", c="k" if "TXF" in n else "gray", lw=1, label=n)
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_ylabel("OOS Sharpe"); plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.set_title("OOS Sharpe: month-end + roll (25) vs expiry rebalance (26)")
    plt.tight_layout(); plt.savefig(out / "exp_vs_cost25.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
