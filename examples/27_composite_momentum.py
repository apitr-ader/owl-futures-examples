"""27 個股期貨：複合動能（1/3/6 個月平均報酬）雙動能，沿用 26 的到期換倉框架（不修改 21-26）

框架（同 26）：到期日收盤算訊號 → 舊部位以結算價（到期日收盤近似）結算、無滑價 → 次一交易日開盤開新近月；無換月成本。
  成本：結算 手續費 50 + 期交稅；開倉 手續費 + 稅 + 滑價 1 檔（敏感度 0.5 檔）。5 檔、單檔上限 40%。
複合分數：c = mean(r21(skip s), r63, r126)，r_L = I[x-skip]/I[x-L]-1（1m 腿可 skip 0/5 日）。
  vs=1：每腿除以自身波動（該窗日報酬標準差·√L）再平均，避免 6m 腿主導排序。
絕對濾網：原始（未縮放）複合報酬 > 0 或 > rf（rf 換算成 70 日平均期間）。
大盤濾網：TXF 同樣的複合報酬 > 0。波動目標只在換倉時調整。
網格：skip 0/5、thr 0/rf、vs 0/1、cad 1/3（每到期/季）、buf 0/1、mkt 0/1、vt 0/15%、EW/IV/MV/ERC → 512。
挑參：IS 2014-2019 plateau（IS 平均空槽 < 30%），OOS 2020-2026/10；對照 26 最佳單一回看 pick 與 TXF B&H、等權可選池。
products.csv 只含現存契約 → 倖存者偏差（結果偏樂觀）。
"""
import argparse
import importlib.util
import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("ex26", Path(__file__).with_name("26_expiry_rebalance.py"))
m26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m26)
m24, o22, dm, D = m26.m24, m26.o22, m26.dm, m26.D
IS0, IS1, EX1, END, PERIODS = m26.IS0, m26.IS1, m26.EX1, m26.END, m26.PERIODS
G = dict(skip=[0, 5], thr=["0", "rf"], vs=[0, 1], cad=[1, 3], buf=[0, 1], mkt=[0, 1], vt=[0.0, 0.15],
         scheme=["EW", "IV", "MV", "ERC"])
LEGS = (21, 63, 126)
_score26 = m26.score


def composite(i, U, skip, vs):
    I, ret = D["I"], D["ret"]
    raw, sc = [], []
    for L in LEGS:
        s = skip if L == 21 else 0
        r = I[i - s, U] / I[i - L, U] - 1
        raw.append(r)
        if vs:
            v = ret[i - L + 1:i + 1, U].std(0) * np.sqrt(L)
            sc.append(r / np.maximum(v, 1e-6))
    raw = np.mean(raw, 0)
    return (np.mean(sc, 0) if vs else raw), raw


def score(fam, P, i, U):
    if fam != "CMP":
        return _score26(fam, P, i, U)
    sc, raw = composite(i, U, P["skip"], P["vs"])
    thr = (1 + dm.RF) ** (np.mean(LEGS) / 252) - 1 if P["thr"] == "rf" else 0.0
    return sc, raw > thr


m26.score = score


def use_market(fam):
    D["txf_mom"] = D["txf_cmp"] if fam == "CMP" else D["txf_252"]


def _work(P):
    use_market("CMP")
    o = m26.simulate("CMP", P)
    row = dict(P)
    for nm, lo, hi in PERIODS:
        row[nm + "_sharpe"] = m24.stats(m24.win(o["r"], lo, hi))["sharpe"]
    row["cash"] = o["cash"]
    return row


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
    D["txf_cmp"] = np.mean([(tI / tI.shift(L) - 1).values for L in LEGS], 0)
    D["txf_252"] = D["txf_mom"].copy()
    pd.set_option("display.width", 250)

    C = [dict(zip(G, v), lb=126) for v in itertools.product(*G.values())]
    with Pool(a.procs) as pool:
        res = pd.DataFrame(pool.map(_work, C, chunksize=8)).drop(columns="lb")
    res["plateau"] = m24.plateau(res, G)
    res.to_csv(out / "cmp_grid.csv", index=False)
    s0 = dm.expected_max_sr(len(res), float(res.IS_sharpe.var()))
    rho = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]
    print(f"CMP 試驗 {len(res)} SR0 {s0:.3f} Spearman(IS,OOS) {rho:.3f}")
    print("IS", res.IS_sharpe.describe().round(3).to_dict()); print("OOS", res.OOS_sharpe.describe().round(3).to_dict())
    sens = []
    for k in G:
        g = res.groupby(k)[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median()
        sens += [dict(param=k, value=str(v), **r.to_dict()) for v, r in g.iterrows()]
    sens = pd.DataFrame(sens); sens.to_csv(out / "cmp_sensitivity.csv", index=False)
    print(sens.round(3).to_string())

    el = res[res.cash < 0.3].sort_values(["plateau", "IS_sharpe"], ascending=False)
    picks = [("CMP", "repick", el.iloc[0])]
    for vs in (0, 1):
        picks.append(("CMP", f"repick_vs{vs}", el[el.vs == vs].iloc[0]))
    todo = [(f, v, {**{k: (p[k].item() if hasattr(p[k], "item") else p[k]) for k in G}, "lb": 126}, s0) for f, v, p in picks]
    e26 = pd.read_csv(out / "exp_summary.csv")
    def p26(fam, var):
        r = e26[(e26.strategy == fam) & (e26.variant == var)].iloc[0]
        P = {}
        for kv in r.cfg.split():
            k, v = kv.split("=")
            P[k] = v if k == "scheme" else (float(v) if k == "vt" else int(v))
        return P, float(r.grid_sr0)
    for fam, var, nm in (("DMS", "repick_lb20", "26 lb20 monthly"), ("DM", "repick", "26 DM lb252 quarterly")):
        P, s = p26(fam, var)
        todo.append((fam, nm, P, s))
    tab, curves = [], {}
    for fam, var, P, s in todo:
        use_market(fam)
        for slip in (1.0, 0.5):
            rr, eq = m26.evaluate(fam, P, slip, s, var)
            for r in rr:
                r["cfg"] = " ".join(f"{k}={v}" for k, v in P.items() if not (fam == "CMP" and k == "lb"))
            tab += rr
            if slip == 1.0:
                curves[f"{fam} {var}"] = eq
    for r in tab:
        if r["strategy"] == "CMP":
            r["grid_trials"], r["grid_spearman"] = len(res), rho
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
    tab.to_csv(out / "cmp_summary.csv", index=False)
    print(tab[["strategy", "variant", "slip", "period", "CAGR", "sharpe", "mdd", "turnover_yr", "cost_pct_yr", "DSR"]].round(3).to_string())
    print(tab.drop_duplicates(["strategy", "variant"])[["strategy", "variant", "cfg"]].to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for n, r in curves.items():
        r = m24.win(r, IS0, END); plt.plot(r.index, (1 + r).cumprod(), label=n, lw=2 if n == "CMP repick" else 1.2)
    for n, r in bm.items():
        r = m24.win(r, IS0, END); plt.plot(r.index, (1 + r).cumprod(), label=n, ls="--", c="k" if "TXF" in n else "gray")
    plt.axvline(IS1, color="k", ls="--", lw=.8); plt.yscale("log"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.title("Composite (1/3/6m) momentum, expiry rebalance, 1 tick slip vs single-lookback (26) and benchmarks")
    plt.tight_layout(); plt.savefig(out / "cmp_equity.png", dpi=110); plt.close()
    fig, axs = plt.subplots(2, 4, figsize=(15, 7)); axs = axs.ravel()
    for ax, k in zip(axs, G):
        g = res.groupby(k)[["IS_sharpe", "OOS_sharpe"]].median(); x = np.arange(len(g))
        ax.bar(x - .2, g.IS_sharpe, .4, label="IS"); ax.bar(x + .2, g.OOS_sharpe, .4, label="OOS")
        ax.set_xticks(x, [str(v) for v in g.index]); ax.set_title(k); ax.grid(alpha=.3); ax.axhline(0, c="k", lw=.5)
    axs[0].legend(); fig.suptitle(f"Composite momentum: median Sharpe by parameter ({len(res)} trials, IS-OOS rank corr {rho:.2f})")
    fig.tight_layout(); fig.savefig(out / "cmp_sensitivity.png", dpi=110); plt.close(fig)
    plt.figure(figsize=(6, 5)); plt.scatter(res.IS_sharpe, res.OOS_sharpe, s=6, alpha=.5)
    plt.xlabel("IS Sharpe"); plt.ylabel("OOS Sharpe"); plt.grid(alpha=.3); plt.title(f"IS vs OOS (Spearman {rho:.2f})")
    plt.tight_layout(); plt.savefig(out / "cmp_is_vs_oos.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
