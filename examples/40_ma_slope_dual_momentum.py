"""40 雙動能：把價格動能換成「均線增速」動能（日K），其餘沿用 32 的 '27 pick'（新可選池）。
基準 27 pick：任何上市股票期貨、持有 10 檔（第 1-10 名，過絕對濾網）、每到期換倉（26 框架：到期結算、次日開盤開新近月）、
  大盤濾網（TXF 1/3/6m 平均報酬 > 0）、目標波動 15%、等權、VS 複合（1/3/6m 報酬各除以自身波動）。
訊號：slope_{L,N}(t) = MA_L(t) / MA_L(t-N) - 1（MA 用換月調整後的連續價格指數 I）。
  單一：L ∈ {10,20,60} × N ∈ {5,10,20,60}；複合：MA20/21日、MA60/63日、MA120/126日 三腿平均（新上市用可用腿，需第一腿）。
  vs=1：每腿除以 (該窗日報酬標準差 · √N)（同 27/28 的波動調整）。
  相對動能 = 依 slope 排名；絕對動能 = slope 年化 (1+s)^(252/N)-1 > rf（thr=rf），或 slope > 0（thr=0）；複合用三腿平均的年化。
  大盤濾網：cmp（同 27：TXF 1/3/6m 平均 > 0）或 ma250（TXF 收盤 > 250 日均線，39 的發現）。
網格 13 訊號 × vs 2 × thr 2 × mkt 2 = 104。挑參：IS 2014-2019 平台（一次只改一個參數的鄰居 + 自身 IS Sharpe 中位數），
  IS 平均空槽 < 30%；OOS 2020-2026/10，另看 ex-2025-26；多重檢定 → DSR（以 104 組 IS Sharpe 變異數算期望最大 Sharpe）。
成本同 32：手續費 50、稅 2e-5、開倉滑價 1 檔、結算無滑價、無換月成本。
對照：價格動能基準（27 pick）、TXF B&H、39（TXF SMA250 1/0，期貨模擬含換月成本）；並報與 39 的日報酬相關係數與 50/50 組合。
存活者偏差：products.csv 只含現存契約 → 個股期貨結果偏樂觀（TXF 與 39 無此問題）。
"""
import importlib.util, itertools, math
from pathlib import Path
import numpy as np, pandas as pd


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


m32 = _load("nu32", "32_four_strategies_new_universe.py")
m31, m24, dm, D = m32.m31, m32.m24, m32.dm, m32.D
IS0, IS1, EX1, END, PERIODS = m32.IS0, m32.IS1, m32.EX1, m32.END, m32.PERIODS
OUT, NH = m32.OUT, m32.NH
LS, NS = (10, 20, 60), (5, 10, 20, 60)
CMP_LEGS = ((20, 21), (60, 63), (120, 126))
BASE = m32.STRATS["27 pick"]


def sigs():
    out = [(f"L{L}N{N}", ((L, N),)) for L in LS for N in NS]
    return out + [("CMP", CMP_LEGS)]


def setup():
    m32.setup()
    I, ret, nh = D["I"], D["ret"], D["M"]["nhist"].values
    tI = (1 + D["txr"]).cumprod()
    D["sig"]["ma250"] = (tI / tI.rolling(250).mean() - 1).fillna(-1).values
    MA = {L: pd.DataFrame(I).rolling(L, min_periods=L).mean().values for L in {l for _, g in sigs() for l, _ in g}}
    rfd = (1 + dm.RF)
    for x in D["X"]:
        U = D["univ"][x]; h = nh[x, U]
        for name, legs in sigs():
            raw = np.zeros(len(U)); vsc = np.zeros(len(U)); ann = np.zeros(len(U)); cnt = np.zeros(len(U))
            for L, N in legs:
                m = (h >= L + N) & (x - N >= 0)
                s = np.where(m, MA[L][x, U] / MA[L][max(x - N, 0), U] - 1, 0.0); s = np.nan_to_num(s)
                v = ret[max(x - N + 1, 0):x + 1, U].std(0) * math.sqrt(N)
                raw += s; vsc += np.where(m, s / np.maximum(v, 1e-6), 0); ann += np.where(m, np.sign(1 + s) * np.abs(1 + s) ** (252 / N) - 1, 0); cnt += m
            elig = (h >= legs[0][0] + legs[0][1])
            c = np.maximum(cnt, 1); raw, vsc, ann = raw / c, vsc / c, ann / c
            for vs in (0, 1):
                sc = np.where(elig, vsc if vs else raw, -np.inf)
                D["scores"][x][f"{name}|vs{vs}|rf"] = (sc, elig & (ann > rfd - 1))
                D["scores"][x][f"{name}|vs{vs}|0"] = (sc, elig & (raw > 0))


def cash(P, lo, hi):
    days = D["days"]; e = []
    for x in D["X"]:
        if not (lo <= days[x] < hi):
            continue
        U = D["univ"][x]
        if len(U) < dm.MIN_UNIVERSE or (P["mkt"] and not D["sig"][P["msig"]][x] > 0):
            e.append(1.0); continue
        sc, ok = D["scores"][x][P["score"]]; o = np.argsort(-sc)[:NH]
        e.append(1 - ok[o].sum() / NH)
    return float(np.mean(e))


def s39():
    m39 = _load("ma39", "39_txf_ma_scaling.py")
    X = m39.m37.load(); X = X[X.index >= pd.Timestamp("2010-01-01")]
    st = m39.state(X, 250)
    return m39.run(X, st, 1.0, 0.0)["r"]


def main():
    setup()
    pd.set_option("display.width", 250)
    grid, rets = [], {}
    for (name, _), vs, thr, mk in itertools.product(sigs(), (0, 1), ("rf", "0"), ("cmp", "ma250")):
        P = {**BASE, "score": f"{name}|vs{vs}|{thr}", "msig": mk}
        r = m32.simulate(P)["r"]; rets[P["score"] + "|" + mk] = r
        grid.append(dict(sig=name, vs=vs, thr=thr, mkt=mk, IS_cash=cash(P, IS0, IS1),
                         **{f"{nm}_sharpe": m24.stats(m24.win(r, lo, hi))["sharpe"] for nm, lo, hi in PERIODS},
                         **{f"{nm}_CAGR": m24.stats(m24.win(r, lo, hi))["CAGR"] for nm, lo, hi in PERIODS}))
    g = pd.DataFrame(grid)
    g["L"] = g.sig.str.extract(r"L(\d+)").astype(float); g["N"] = g.sig.str.extract(r"N(\d+)").astype(float)

    def nbrs(r):
        same = (g.vs == r.vs) & (g.thr == r.thr) & (g.mkt == r.mkt)
        if r.sig == "CMP":
            m = same & (g.sig == "CMP")
        else:
            li, ni = LS.index(int(r.L)), NS.index(int(r.N))
            m = same & (((g.L == r.L) & g.N.isin([NS[j] for j in (ni - 1, ni + 1) if 0 <= j < len(NS)])) |
                        ((g.N == r.N) & g.L.isin([LS[j] for j in (li - 1, li + 1) if 0 <= j < len(LS)])))
        m |= (g.sig == r.sig) & ((((g.vs != r.vs) & (g.thr == r.thr) & (g.mkt == r.mkt))) | ((g.vs == r.vs) & (g.thr != r.thr) & (g.mkt == r.mkt)) | ((g.vs == r.vs) & (g.thr == r.thr) & (g.mkt != r.mkt)))
        return g[m | ((g.sig == r.sig) & same)].IS_sharpe.median()
    g["plateau"] = [nbrs(r) for r in g.itertuples()]
    g.to_csv(OUT / "mas_grid.csv", index=False)
    sr0 = dm.expected_max_sr(len(g), float(g.IS_sharpe.var()))
    print(f"configs {len(g)}  SR0(expected max IS Sharpe under null) {sr0:.3f}")
    for mk in ("cmp", "ma250"):
        for thr in ("rf", "0"):
            print(f"IS Sharpe mkt={mk} thr={thr}  (rows sig, cols vs)")
            print(g[(g.mkt == mk) & (g.thr == thr)].pivot_table(index="sig", columns="vs", values=["IS_sharpe", "OOS_sharpe", "IS_cash"], sort=False).round(2).to_string())
    ok = g[g.IS_cash < 0.30].sort_values(["plateau", "IS_sharpe"], ascending=False)
    print("plateau top 10 (IS cash<30%):"); print(ok.head(10)[["sig", "vs", "thr", "mkt", "IS_cash", "plateau", "IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].round(3).to_string(index=False))
    b = ok.iloc[0]; hk = f"{b.sig}|vs{b.vs}|{b.thr}"
    cfgs = {f"HEAD {b.sig} vs{b.vs} thr{b.thr} mkt={b.mkt}": {**BASE, "score": hk, "msig": b.mkt},
            f"HEAD w/ other mkt": {**BASE, "score": hk, "msig": "ma250" if b.mkt == "cmp" else "cmp"},
            "27 pick (price mom, cmp mkt)": dict(BASE), "27 pick + ma250 mkt": {**BASE, "msig": "ma250"}}
    r39 = s39()
    curves, rows = {}, []
    for k, P in cfgs.items():
        full = m32.simulate(P); curves[k] = full["r"]
        for per, lo, hi in PERIODS:
            o = m32.simulate(P, lo, hi); s = m24.stats(m24.win(full["r"], lo, hi)); y = max(o["yrs"], 1e-9)
            rows.append(dict(strategy=k, period=per, CAGR=s["CAGR"], vol=s["vol"], sharpe=s["sharpe"], MDD=s["mdd"],
                             turnover_yr=o["turn"] / y, cost_pct_yr=o["cost"] / y * 100, gross=o["gross"], cash=cash(P, lo, hi),
                             DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" and k.startswith("HEAD ") else np.nan,
                             corr39=m24.win(full["r"], lo, hi).corr(r39.reindex(m24.win(full["r"], lo, hi).index).fillna(0))))
    hr = curves[list(cfgs)[0]]
    combo = 0.5 * hr + 0.5 * r39.reindex(hr.index).fillna(0)
    for k, r in {"TXF B&H": D["txr"], "39 TXF SMA250 1/0": r39, "50/50 HEAD + 39 (daily rebal)": combo}.items():
        curves[k] = r
        for per, lo, hi in PERIODS:
            s = m24.stats(m24.win(r, lo, hi))
            rows.append(dict(strategy=k, period=per, CAGR=s["CAGR"], vol=s["vol"], sharpe=s["sharpe"], MDD=s["mdd"]))
    T = pd.DataFrame(rows); T.to_csv(OUT / "mas_summary.csv", index=False)
    print(T.pivot_table(index="strategy", columns="period", values=["CAGR", "sharpe", "MDD"], sort=False).round(3).to_string())
    print(T[T.period == "OOS"][["strategy", "turnover_yr", "cost_pct_yr", "gross", "cash", "DSR", "corr39"]].round(3).to_string(index=False))
    print(T[T.period == "IS"][["strategy", "cash", "corr39"]].round(3).to_string(index=False))
    Y = pd.DataFrame({k: (1 + m24.win(r, IS0, END)).groupby(m24.win(r, IS0, END).index.year).prod() - 1 for k, r in curves.items()})
    Y.to_csv(OUT / "mas_yearly.csv"); print(Y.round(3).to_string())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for k, r in curves.items():
        r = m24.win(r, IS0, END); e = (1 + r).cumprod()
        st = dict(c="k", lw=2) if k == "TXF B&H" else dict(lw=1.1, ls="--" if "other" in k or "+ ma250" in k else "-")
        axs[0].plot(e.index, e, label=k, **st); axs[1].plot(e.index, e / e.cummax() - 1, **st)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=7, ncol=2); axs[0].set_title("MA-slope dual momentum vs price momentum, TXF B&H, script 39")
    axs[1].set_ylabel("drawdown")
    for a in axs:
        a.axvline(IS1, c="gray", lw=0.8)
    plt.tight_layout(); plt.savefig(OUT / "mas_equity_dd.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
