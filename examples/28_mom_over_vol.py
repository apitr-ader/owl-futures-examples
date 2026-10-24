"""28 個股期貨：風險調整動能（複合動能 ÷ 波動）排名，沿用 27 的到期換倉框架（不修改 21-27）

框架同 26/27：到期日收盤算訊號 → 舊部位結算（無滑價）→ 次一交易日開盤開新近月；無換月成本；5 檔、單檔上限 40%。
分數 sc：
  raw  ：複合動能 c = mean(r21, r63, r126)（同 27 vs=0）
  leg  ：27 的逐腿波動縮放 mean(r_L / (σ_L·√L))（同 27 vs=1）
  s63/s126/s252 ：c / 年化日報酬標準差（63/126/252 日）
  d63/d126/d252 ：c / 年化下檔波動 sqrt(mean(min(r,0)^2))·√252
絕對濾網：原始 c > 0 或 > rf（70 日）；大盤濾網：TXF 的 c > 0；波動目標 無/15%（只在換倉調整）。
網格：sc 8 × thr 2 × cad 2 × buf 2 × mkt 2 × vt 2 × 權重 4 = 1024。
挑參：IS 2014-2019 plateau（鄰居只沿二元參數 thr/cad/buf/mkt/vt，sc 與權重視為類別），IS 平均空槽 < 30%。
products.csv 只含現存契約 → 倖存者偏差（結果偏樂觀）。
"""
import argparse
import importlib.util
import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("cm27", Path(__file__).with_name("27_composite_momentum.py"))
m27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m27)
m26, m24, o22, dm, D = m27.m26, m27.m24, m27.o22, m27.dm, m27.D
IS0, IS1, EX1, END, PERIODS = m26.IS0, m26.IS1, m26.EX1, m26.END, m26.PERIODS
SCS = ["raw", "leg", "s63", "s126", "s252", "d63", "d126", "d252"]
G = dict(sc=SCS, thr=["0", "rf"], cad=[1, 3], buf=[0, 1], mkt=[0, 1], vt=[0.0, 0.15], scheme=["EW", "IV", "MV", "ERC"])
_score27 = m26.score


def score(fam, P, i, U):
    if fam != "MOV":
        return _score27(fam, P, i, U)
    sc = P["sc"]
    s, raw = m27.composite(i, U, 0, 1 if sc == "leg" else 0)
    if sc[0] in "sd":
        W = int(sc[1:])
        r = D["ret"][i - W + 1:i + 1, U]
        v = (np.sqrt((np.minimum(r, 0) ** 2).mean(0)) if sc[0] == "d" else r.std(0)) * np.sqrt(252)
        s = raw / np.maximum(v, 1e-4)
    thr = (1 + dm.RF) ** (np.mean(m27.LEGS) / 252) - 1 if P["thr"] == "rf" else 0.0
    return s, raw > thr


m26.score = score


def market(fam):
    D["txf_mom"] = D["txf_cmp"] if fam in ("MOV", "CMP") else D["txf_252"]


def _work(P):
    market("MOV")
    o = m26.simulate("MOV", P)
    row = dict(P)
    for nm, lo, hi in PERIODS:
        row[nm + "_sharpe"] = m24.stats(m24.win(o["r"], lo, hi))["sharpe"]
    row["cash"] = o["cash"]
    return row


def plateau(res):
    keys = list(G)
    idx = {k: {x: j for j, x in enumerate(G[k])} for k in keys}
    code = pd.DataFrame({k: res[k].map(idx[k]) for k in keys})
    look = pd.Series(res.IS_sharpe.values, pd.MultiIndex.from_frame(code))
    tot, cnt = res.IS_sharpe.values.copy(), np.ones(len(res))
    for k in ("thr", "cad", "buf", "mkt", "vt"):
        for d in (-1, 1):
            nb = code.copy(); nb[k] += d
            v = look.reindex(pd.MultiIndex.from_frame(nb)).values; ok = ~np.isnan(v)
            tot[ok] += v[ok]; cnt[ok] += 1
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stock_futures")
    ap.add_argument("--txf", default="data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    o22.prepare(a.data, a.txf)
    D["txv"] = D["txr"].values; D["txr_s"] = D["txr"]
    D["exp"] = m26.expiry_calendar()
    tI = (1 + D["txr"]).cumprod()
    D["txf_cmp"] = np.mean([(tI / tI.shift(L) - 1).values for L in m27.LEGS], 0)
    D["txf_252"] = D["txf_mom"].copy()
    pd.set_option("display.width", 250)

    C = [dict(zip(G, v), lb=126, win=252) for v in itertools.product(*G.values())]
    with Pool(a.procs) as pool:
        res = pd.DataFrame(pool.map(_work, C, chunksize=8)).drop(columns=["lb", "win"])
    res["plateau"] = plateau(res)
    res.to_csv(out / "mv_grid.csv", index=False)
    s0 = dm.expected_max_sr(len(res), float(res.IS_sharpe.var()))
    rho = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]
    print(f"MOV 試驗 {len(res)} SR0 {s0:.3f} Spearman(IS,OOS) {rho:.3f}")
    sens = []
    for k in G:
        g = res.groupby(k, sort=False)[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median()
        sens += [dict(param=k, value=str(v), **r.to_dict()) for v, r in g.iterrows()]
    sens = pd.DataFrame(sens); sens.to_csv(out / "mv_sensitivity.csv", index=False)
    print(sens.round(3).to_string())
    bysc = res.groupby("sc", sort=False).apply(lambda g: pd.Series(dict(
        IS_med=g.IS_sharpe.median(), OOS_med=g.OOS_sharpe.median(), EX_med=g.OOS_ex2526_sharpe.median(),
        IS_max=g.IS_sharpe.max(), spearman=g[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1])))
    bysc.to_csv(out / "mv_by_score.csv")
    print(bysc.round(3).to_string())

    el = res[res.cash < 0.3].sort_values(["plateau", "IS_sharpe"], ascending=False)
    todo = [("MOV", "repick", el.iloc[0])] + [("MOV", f"best_{sc}", el[el.sc == sc].iloc[0]) for sc in SCS]
    todo = [(f, v, {**{k: (p[k].item() if hasattr(p[k], "item") else p[k]) for k in G}, "lb": 126, "win": 252}, s0) for f, v, p in todo]
    c27 = pd.read_csv(out / "cmp_summary.csv")
    def parse(fam, var, df):
        r = df[(df.strategy == fam) & (df.variant == var)].iloc[0]
        P = {}
        for kv in r.cfg.split():
            k, v = kv.split("=")
            P[k] = v if k in ("scheme", "thr") else (float(v) if k == "vt" else int(v))
        return P, r
    P, r = parse("CMP", "repick", c27); P["lb"] = 126
    todo.append(("CMP", "27 composite pick", P, m24.np.nan if False else dm.expected_max_sr(512, float(pd.read_csv(out / "cmp_grid.csv").IS_sharpe.var()))))
    P, r = parse("DMS", "26 lb20 monthly", c27)
    todo.append(("DMS", "26 lb20 monthly", P, float(pd.read_csv(out / "exp_summary.csv").query("strategy=='DMS'").grid_sr0.iloc[0])))
    tab, curves = [], {}
    for fam, var, P, s in todo:
        market(fam)
        for slip in (1.0, 0.5):
            rr, eq = m26.evaluate(fam, P, slip, s, var)
            for x in rr:
                x["cfg"] = " ".join(f"{k}={v}" for k, v in P.items() if k not in ("lb", "win") or fam == "DMS")
            tab += rr
            if slip == 1.0 and not var.startswith("best_"):
                curves[f"{fam} {var}"] = eq
    for x in tab:
        if x["strategy"] == "MOV":
            x["grid_trials"], x["grid_spearman"] = len(res), rho
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
    tab = pd.DataFrame(tab)
    tab.to_csv(out / "mv_summary.csv", index=False)
    print(tab[["strategy", "variant", "slip", "period", "CAGR", "sharpe", "mdd", "turnover_yr", "cost_pct_yr", "DSR"]].round(3).to_string())
    print(tab.drop_duplicates(["strategy", "variant"])[["strategy", "variant", "cfg"]].to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for n, r in curves.items():
        r = m24.win(r, IS0, END); plt.plot(r.index, (1 + r).cumprod(), label=n, lw=2 if "MOV" in n else 1.2)
    for n, r in bm.items():
        r = m24.win(r, IS0, END); plt.plot(r.index, (1 + r).cumprod(), label=n, ls="--", c="k" if "TXF" in n else "gray")
    plt.axvline(IS1, color="k", ls="--", lw=.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title("Momentum / volatility ranking (expiry rebalance, 1 tick) vs 27, 26 and benchmarks")
    plt.tight_layout(); plt.savefig(out / "mv_equity.png", dpi=110); plt.close()
    ax = bysc[["IS_med", "OOS_med", "EX_med"]].plot.bar(figsize=(10, 4.5))
    ax.set_ylabel("median Sharpe"); ax.grid(alpha=.3); ax.set_title("Median Sharpe by ranking score / vol window (128 configs each)")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(out / "mv_by_score.png", dpi=110); plt.close()
    plt.figure(figsize=(6, 5))
    for sc in SCS:
        g = res[res.sc == sc]; plt.scatter(g.IS_sharpe, g.OOS_sharpe, s=6, label=sc)
    plt.xlabel("IS Sharpe"); plt.ylabel("OOS Sharpe"); plt.grid(alpha=.3); plt.legend(fontsize=7, markerscale=2)
    plt.title(f"IS vs OOS (Spearman {rho:.2f})"); plt.tight_layout(); plt.savefig(out / "mv_is_vs_oos.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
