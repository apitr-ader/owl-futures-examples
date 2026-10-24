"""25 個股期貨策略：較便宜、較實際的成本模型 + 檔位/價格比可選池濾網（不修改 21-24）

成本模型（每口）：
  OLD   ：21-24 原設定。調倉每邊 手續費 50 + 稅 2/100000 + 1 檔滑價；換月拆兩腳各自付 1 檔（＝2 檔）＋兩腳手續費/稅。
  NEW   ：換月用價差（calendar spread）單：整筆 1 檔滑價（不是每腳 1 檔）＋兩腳手續費/稅；調倉同 OLD（每邊 1 檔）。
  NEW05 ：同 NEW 但滑價每邊 0.5 檔、換月價差 0.5 檔（敏感度）。
檔位/價格比濾網 tf：訊號日只留 tick(收盤)/收盤 <= tf 的契約（tf=1 表示不濾；0.2%、0.1%）。
  注意以 tick 表計：0.2% 會留下 25-50、50-100（價>=50）、250-500、500-1000 各價帶的上半段；>=1000 價帶 tick=5 → 要價>=2500。

策略與重新挑參（NEW 成本、IS 2014-2019 plateau、IS 平均空槽 < 30%）：
  DM  ：雙動能（21/22 家族）lb 63/126/252、skip 0/21、5 檔、月/季、緩衝 0/1、大盤濾網、波動目標、EW/IV/MV/ERC、tf
  DMS ：短回看雙動能（23）lb 5-20、週/月、skip 0/1、…、tf
  REV ：短期反轉（24）lb 5/10/20、skip 0/1、週/月、降週轉 無/緩衝/最短持有、流動性 2 檔、…、tf
  RRES：殘差動能（24）lb 63/126/252、skip 0/21、窗 252、vol-scaled、絕對濾網、月/季、…、tf
原始挑選（21 原 pick、22 best_N5、23 各 lb chosen、24 rev/rres chosen）在 OLD/NEW/NEW05 下重評。
成本拆成 換月 vs 調倉（換月-only：把調倉手續費/稅/滑價設 0 重跑；調倉 = 總 − 換月）。
DSR：重挑用本檔各家族的試驗數；原 pick 用原研究的試驗數與 IS Sharpe 變異。
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
TFS = [1.0, 0.002, 0.001]
COMM0, TAX0 = dm.COMMISSION, dm.TAX

G = {
    "DM": dict(lb=[63, 126, 252], skip=[0, 21], freq=[1, 3], buf=[0, 1], mkt=[0, 1], vt=[0.0, 0.15],
               scheme=["EW", "IV", "MV", "ERC"], tf=TFS),
    "DMS": dict(lb=[5, 10, 15, 20], reb=["W", "M"], skip=[0, 1], mkt=[0, 1], vt=[0.0, 0.15],
                scheme=["EW", "IV", "MV", "ERC"], tf=TFS),
    "REV": dict(lb=[5, 10, 20], skip=[0, 1], reb=["W", "M"], tv=["none", "buf", "hold"], liq=[0, 1], mkt=[0, 1],
                vt=[0.0, 0.15], scheme=["EW", "IV", "MV", "ERC"], tf=TFS),
    "RRES": dict(lb=[63, 126, 252], skip=[0, 21], win=[252], scaled=[0, 1], absf=["raw", "res"], reb=["M", "Q"],
                 mkt=[0, 1], vt=[0.0, 0.15], scheme=["EW", "IV", "MV", "ERC"], tf=TFS),
}

# ---------- 成本 ----------
def set_cost(model):
    slip = 0.5 if model == "NEW05" else 1.0
    mult, R, roll = D["mult"], np.nan_to_num(D["R"]), D["M"]["ROLL"].values
    t = o22.tick_arr(R)
    legs = 2 * (COMM0 + mult * TAX0 * R)
    rc = legs + mult * t * (2.0 if model == "OLD" else slip)
    rc = np.where(roll, rc, 0.0)
    D["CR"] = np.vstack([np.zeros(len(D["cols"])), np.cumsum(rc, 0)])
    D["tickRO"] = slip * o22.tick_arr(np.nan_to_num(D["RO"]))
    D["_tick_save"] = D["tickRO"]
    dm.COMMISSION, dm.TAX = COMM0, TAX0


def rebal_costs(on):
    if on:
        dm.COMMISSION, dm.TAX, D["tickRO"] = COMM0, TAX0, D["_tick_save"]
    else:
        dm.COMMISSION, dm.TAX, D["tickRO"] = 0.0, 0.0, np.zeros_like(D["_tick_save"])


# ---------- 檔位/價格比濾網 ----------
TF = [1.0]
_orig_univ = o22.univ
_UT = {}


def univ_tf(i, minvol, minpx):
    U = _orig_univ(i, minvol, minpx)
    t = TF[0]
    if t >= 1:
        return U
    k = (i, minvol, minpx, t)
    if k not in _UT:
        R = np.nan_to_num(D["R"][i, U])
        _UT[k] = U[(R > 0) & (o22.tick_arr(R) / np.where(R > 0, R, 1) <= t)]
    return _UT[k]


o22.univ = univ_tf


# ---------- 模擬 ----------
def sched_dm(kind, lo, hi):
    days = D["days"]
    s = pd.Series(np.arange(len(days)), days)
    iso = days.isocalendar()
    key = [iso.year.values, iso.week.values] if kind == "W" else [days.year, days.month]
    last = s.groupby(key).last().sort_values().values
    lo = pd.Timestamp("2013-12-01") if lo is None else lo - pd.Timedelta(days=10)
    hi = END if hi is None else hi
    return [int(i) for i in last if lo <= days[i] < hi]


def simulate(fam, P, lo=None, hi=None):
    TF[0] = P["tf"]
    if fam in ("DM", "DMS"):
        D["mes"] = sched_dm(P.get("reb", "M"), lo, hi)
        Q = dict(lb=P["lb"], skip=P["skip"], ntop=5, freq=P.get("freq", 1), buf=P.get("buf", 0), minvol=P.get("_minvol", 50), minpx=P.get("_minpx", 0),
                 thr=P.get("_thr", "0"), mkt=P["mkt"], vt=P["vt"], scheme=P["scheme"])
        o = o22.run(Q, keep_path=True)
        if hi is not None:
            o["r"] = o["r"][(o["r"].index >= lo) & (o["r"].index < hi)]
            o["yrs"] = len(o["r"]) / 252
        return o
    return m24.run(P, "rev" if fam == "REV" else "res", lo, hi)


def _work(args):
    fam, P = args
    o = simulate(fam, P)
    row = dict(P)
    for nm, lo, hi in PERIODS:
        row[nm + "_sharpe"] = m24.stats(m24.win(o["r"], lo, hi))["sharpe"]
    row["cash"] = o["cash"]
    return row


def evaluate(fam, P, model, sr0, tag, variant):
    set_cost(model)
    rows, full = [], simulate(fam, P)
    for per, lo, hi in PERIODS:
        rebal_costs(True); o = simulate(fam, P, lo, hi)
        rebal_costs(False); orl = simulate(fam, P, lo, hi)
        rebal_costs(True)
        s = m24.stats(m24.win(full["r"], lo, hi))
        yrs = max(o["yrs"], 1e-9)
        rows.append(dict(strategy=fam, variant=variant, cost_model=model, period=per,
                         cfg=" ".join(f"{k}={v}" for k, v in P.items()), CAGR=s["CAGR"], vol=s["vol"],
                         sharpe=s["sharpe"], mdd=s["mdd"], turnover_yr=o["turn"] / yrs,
                         cost_pct_yr=o["cost"] / yrs * 100, roll_cost_pct_yr=orl["cost"] / yrs * 100,
                         rebal_cost_pct_yr=(o["cost"] - orl["cost"]) / yrs * 100, cash_slots=o["cash"],
                         DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan,
                         sr0=sr0))
    return rows, full["r"]


def originals():
    """原研究挑選 + 原試驗數 / IS Sharpe 變異（給 DSR）。"""
    R = Path("results")
    out = []
    g = pd.read_csv(R / "dm_grid.csv")
    out.append(("DM", "orig21", dict(lb=252, skip=21, freq=1, buf=0, mkt=1, vt=0.15, scheme="EW", tf=1.0),
                dm.expected_max_sr(len(g), float(g.IS_sharpe.var()))))
    res = pd.read_parquet(R / "dmo_grid.parquet")
    b5, sc, _ = o22.pick(res, list(range(2014, 2020)), res.ntop == 5)
    out.append(("DM", "orig22", dict(lb=int(b5.lb), skip=int(b5.skip), freq=int(b5.freq), buf=int(b5.buf),
                                      mkt=int(b5.mkt), vt=float(b5.vt), scheme=b5.scheme, tf=1.0,
                                      _minvol=int(b5.minvol), _minpx=int(b5.minpx), _thr=b5.thr),
                dm.expected_max_sr(len(res), float(sc.var()))))
    g = pd.read_csv(R / "dms_grid.csv"); s0 = dm.expected_max_sr(len(g), float(g.IS_sharpe.var()))
    sm = pd.read_csv(R / "dms_summary.csv")
    for lb, c in sm[(sm.row == "chosen") & (sm.period == "IS")][["lb", "cfg"]].values:
        reb, sk, mk, vt, sch = c.split()
        out.append(("DMS", f"orig23_lb{lb}", dict(lb=int(lb), reb=reb, skip=int(sk[4:]), mkt=int(mk[3:]),
                                                  vt=float(vt[2:]), scheme=sch, tf=1.0), s0))
    for fam, f, pre in (("REV", "rev", "orig24_rev"), ("RRES", "rres", "orig24_rres")):
        g = pd.read_csv(R / f"{f}_grid.csv"); s0 = dm.expected_max_sr(len(g), float(g.IS_sharpe.var()))
        c = pd.read_csv(R / f"{f}_summary.csv").query("row=='chosen'").cfg.iloc[0]
        P = {}
        for kv in c.split():
            k, v = kv.split("=")
            P[k] = v if k in ("reb", "tv", "absf", "scheme") else (float(v) if k == "vt" else int(v))
        P["tf"] = 1.0
        out.append((fam, pre, P, s0))
    return out


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
    for k in ("W", "M", "Q"):
        m24.schedule(k)
    pd.set_option("display.width", 250)

    # 可選池檔數（月底、量>=50）
    mes = sched_dm("M", IS0, END)
    rows = []
    for t in TFS:
        TF[0] = t
        for i in mes:
            rows.append(dict(tf=t, year=D["days"][i].year, n=len(univ_tf(i, 50, 0)), names=set(univ_tf(i, 50, 0))))
    u = pd.DataFrame(rows)
    uy = u.groupby(["year", "tf"]).agg(avg_names=("n", "mean"), min_names=("n", "min"),
                                         distinct=("names", lambda s: len(set().union(*s)))).unstack("tf")
    uy.columns = [f"{a}_tf{b:g}" for a, b in uy.columns]
    uy.to_csv(out / "cost_universe.csv")
    print("可選池檔數（每年月底平均/最少/不重複）\n", uy.round(1).to_string())

    # 基準
    bm = {"TXF B&H": D["txr"]}
    ew = pd.Series(0.0, D["days"]); TF[0] = 1.0
    for n, i in enumerate(mes[:-1]):
        U = univ_tf(i, 50, 0); k0, k1 = i + 1, mes[n + 1] + 1
        if len(U) >= dm.MIN_UNIVERSE:
            ew.iloc[k0:k1] = D["ret"][k0:k1, U].mean(1)
    bm["EW universe (gross)"] = ew
    tab = []
    for n, r in bm.items():
        for per, lo, hi in PERIODS:
            s = m24.stats(m24.win(r, lo, hi))
            tab.append(dict(strategy=n, variant="benchmark", period=per, CAGR=s["CAGR"], vol=s["vol"],
                            sharpe=s["sharpe"], mdd=s["mdd"]))

    # 原始挑選重評
    curves = {}
    for fam, nm, P, s0 in originals():
        P2 = P
        for model in ("OLD", "NEW", "NEW05"):
            r, eq = evaluate(fam, P2, model, s0, None, nm)
            tab += r
            if model == "NEW":
                curves[f"{nm} (NEW)"] = eq

    # 重新挑參（NEW 成本）
    set_cost("NEW")
    grid_info = {}
    for fam, g in G.items():
        C = [(fam, dict(zip(g, v))) for v in itertools.product(*g.values())]
        C = [c for c in C if not (fam == "REV" and c[1]["skip"] >= c[1]["lb"]) and not (fam == "RRES" and c[1]["lb"] > c[1]["win"])]
        with Pool(a.procs) as pool:
            res = pd.DataFrame(pool.map(_work, C, chunksize=16))
        res["plateau"] = m24.plateau(res, g)
        res.to_csv(out / f"cost_grid_{fam.lower()}.csv", index=False)
        s0 = dm.expected_max_sr(len(res), float(res.IS_sharpe.var()))
        rho = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]
        best = res[res.cash < 0.3].sort_values(["plateau", "IS_sharpe"], ascending=False).iloc[0]
        P = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in g}
        grid_info[fam] = dict(trials=len(res), sr0=s0, spearman=rho)
        print(f"\n== {fam} 重挑：試驗 {len(res)} SR0 {s0:.3f} Spearman {rho:.3f}  pick {P}")
        print(res.groupby("tf")[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median().round(3).to_string())
        for model in ("NEW", "NEW05"):
            r, eq = evaluate(fam, P, model, s0, None, "repick")
            tab += r
            if model == "NEW":
                curves[f"{fam} repick (NEW)"] = eq
    for r in tab:
        if r.get("variant") == "repick":
            r.update({f"grid_{k}": v for k, v in grid_info[r["strategy"]].items()})
    tab = pd.DataFrame(tab)
    tab.to_csv(out / "cost_summary.csv", index=False)
    cols = ["strategy", "variant", "cost_model", "period", "CAGR", "sharpe", "mdd", "turnover_yr", "cost_pct_yr",
            "roll_cost_pct_yr", "rebal_cost_pct_yr", "DSR"]
    print(tab[cols].round(3).to_string())
    print(tab[tab.variant == "repick"].drop_duplicates("strategy")[["strategy", "cfg"]].to_string())
    charts(out, tab, curves, bm, uy)


def charts(out, tab, curves, bm, uy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for n, r in curves.items():
        if "repick" in n or n.startswith(("orig21", "orig22", "orig24")):
            r = m24.win(r, IS0, END)
            plt.plot(r.index, (1 + r).cumprod(), label=n, lw=1.8 if "repick" in n else 1, ls="-" if "repick" in n else ":")
    for n, r in bm.items():
        r = m24.win(r, IS0, END); plt.plot(r.index, (1 + r).cumprod(), label=n, lw=1.4, ls="--", c="k" if "TXF" in n else "gray")
    plt.axvline(IS1, color="k", ls="--", lw=.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=7)
    plt.title("Stock-futures strategies under NEW cost model (spread rolls, 1 tick/side) vs benchmarks")
    plt.tight_layout(); plt.savefig(out / "cost_equity.png", dpi=110); plt.close()

    t = tab[(tab.period == "OOS") & (tab.variant != "benchmark")]
    pv = t.pivot_table(index=["strategy", "variant"], columns="cost_model", values="sharpe")[["OLD", "NEW", "NEW05"]]
    ax = pv.plot.bar(figsize=(13, 5.5))
    for n, r in bm.items():
        ax.axhline(m24.stats(m24.win(r, IS1, END))["sharpe"], ls="--", c="k" if "TXF" in n else "gray", lw=1, label=n + " OOS")
    ax.set_ylabel("OOS Sharpe"); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("OOS Sharpe by cost model")
    plt.xticks(rotation=35, ha="right", fontsize=8); plt.tight_layout(); plt.savefig(out / "cost_sharpe.png", dpi=110); plt.close()

    t = tab[(tab.period == "OOS") & (tab.variant != "benchmark")]
    pv = t.pivot_table(index=["strategy", "variant", "cost_model"], values=["roll_cost_pct_yr", "rebal_cost_pct_yr"])
    ax = pv.plot.barh(stacked=True, figsize=(10, 9)); ax.set_xlabel("% of equity per year (OOS)")
    ax.set_title("Annual cost split: roll vs rebalance"); ax.grid(alpha=.3)
    plt.yticks(fontsize=7); plt.tight_layout(); plt.savefig(out / "cost_split.png", dpi=110); plt.close()

    ax = uy[[c for c in uy.columns if c.startswith("avg")]].plot(marker="o", figsize=(9, 4.5))
    ax.set_ylabel("avg # names (month-ends)"); ax.grid(alpha=.3); ax.set_title("Universe size by tick/price filter")
    plt.tight_layout(); plt.savefig(out / "cost_universe.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
