"""34 最佳雙動能（32 的 27 pick，新可選池）加個股出場：移動停利 + 停損。
基礎：VS 複合（1/3/6m 波動調整）> rf、每到期換倉、大盤濾網(TXF 1/3/6m>0)、vt 15%、EW、10 檔第 1-10 名。
部位報酬 g_t = 調整收盤 / 進場調整開盤 - 1（持有期間，每到期重設）。
  移動停利：峰值 G = max g（持有期間）；G > 0 且 g <= (1-trail)·G 時出場。
  停損：g <= -stop 時出場。
CLOSE 版：收盤判斷，隔日開盤出場（若隔日已是到期日之後則照常到期結算）。
HL 版：日內用日K高低：當日先以「前一日為止」的峰值計算停利價 E·(1+(1-trail)·G) 與停損價 E·(1-stop)，
  最低價觸及即以 min(開盤, 觸發價) 出場（跳空時用開盤），之後才用當日最高價更新峰值（保守，不假設先高後低）。
  未使用 1 分 K（可行但需大量讀檔；日K高低已近似）。
出場資金留現金到下次到期換倉（CASH）；或 REFILL：隔日開盤以當期排名下一個未持有且過濾網的標的、相同權重補進（每期最多補到 10 檔）。
成本：手續費 50/邊、交易稅 2e-5、滑價 1 tick（另報 0.5 tick）；無換月成本。存活者偏差：products.csv 只含現存商品。
"""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location("nu32", Path(__file__).with_name("32_four_strategies_new_universe.py"))
m32 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m32)
m31, m24, o22, dm, D = m32.m31, m32.m24, m32.o22, m32.dm, m32.D
OUT = Path("results")
NH = 10
IS0, IS1, EX1, END, PERIODS = m32.IS0, m32.IS1, m32.EX1, m32.END, m32.PERIODS
THR = (1 + dm.RF) ** (np.mean(m31.LEGS) / 252) - 1
COST = dict(slip=1.0)


def load_hl():
    days, cols = D["days"], D["cols"]
    H = pd.DataFrame(index=days, columns=cols, dtype=float); L = H.copy()
    for s in cols:
        a = pd.read_parquet(f"data/stock_futures/daily/{s}_adj.parquet")
        idx = a["time"].dt.tz_localize(None).dt.normalize()
        H[s] = pd.Series(a.high.values, idx).reindex(days); L[s] = pd.Series(a.low.values, idx).reindex(days)
    D["H"] = H.fillna(pd.DataFrame(D["A"], days, cols)).values
    D["L"] = L.fillna(pd.DataFrame(D["A"], days, cols)).values


def simulate(P, lo=None, hi=None):
    """P: trail (None 或 0.2..), stop (None 或 0.05..), mode CLOSE/HL, fill CASH/REFILL"""
    days, mult, A, O, R, RO, H, L = D["days"], D["mult"], D["A"], D["O"], D["R"], D["RO"], D["H"], D["L"]
    Rf = D["M"]["Rf"].values
    slip = COST["slip"]; comm = 50.0
    lo_, hi_ = (IS0 if lo is None else lo), (END if hi is None else hi)
    X = [x for x in D["X"] if lo_ - pd.Timedelta(days=40) <= days[x] < hi_]
    nd = len(days); eq = 1.0; vals = np.full(nd, np.nan)
    turn = cost = 0.0
    tr, st = P.get("trail"), P.get("stop")
    ev = dict(positions=0, trail=0, stop=0, refill=0, ret_trail=[], ret_stop=[], ret_expiry=[])

    def px_raw(k, s):
        v = RO[k, s]
        return v if not np.isnan(v) else Rf[k, s]

    def tc(q, s, p):
        return q * (comm + mult[s] * (dm.TAX * p + slip * o22.tick_arr(np.array([p]))[0]))

    for n, x in enumerate(X):
        k0 = x + 1
        if k0 >= nd:
            break
        k1 = X[n + 1] + 1 if n + 1 < len(X) else nd
        U = D["univ"][x]; sel, queue = [], []
        if len(U) >= dm.MIN_UNIVERSE and D["mktsig"][x] > 0:
            sc, ok = D["scores"][x]["VS"]
            o = np.argsort(-sc); rk = U[o]; okr = ok[o]
            cand = [int(s) for s, v in zip(rk, okr) if v]
            sel, queue = cand[:NH], cand[NH:]
        sel = sorted(s for s in sel if not np.isnan(RO[k0, s]))
        if n == 0 and k0 > 0:
            vals[k0 - 1] = 1.0
        if not sel:
            vals[k0:k1] = eq; continue
        w = dict(zip(sel, m31.weights(x, np.array(sel), "EW", 0.15)))
        pos = {}   # s -> dict(q, E(adj entry), Er(raw entry), G)

        def open_(s, k, wt):
            nonlocal eq, turn, cost
            p = px_raw(k, s)
            q = wt * eq / (p * mult[s]); c = tc(q, s, p)
            turn += q * mult[s] * p / eq; eq -= c; cost += c / eq
            pos[s] = dict(q=q, E=O[k, s], G=0.0, w=wt)
            ev["positions"] += 1

        for s in sel:
            open_(s, k0, w[s])
        cash_pnl = 0.0          # 已實現（相對 eq 起點）
        base = eq
        pending = []           # (s, reason) 於下一日開盤出場
        for k in range(k0, k1):
            # 1) 執行前一日收盤觸發的出場（開盤）
            if pending and k > k0:
                for s, why in pending:
                    p0 = pos.pop(s)
                    xa = O[k, s]; pr = px_raw(k, s)
                    cash_pnl += p0["q"] * mult[s] * (xa - p0["E"])
                    c = p0["q"] * (comm + mult[s] * (dm.TAX * pr + slip * o22.tick_arr(np.array([pr]))[0])); cash_pnl -= c; cost += c / max(eq, 1e-9)
                    ev["ret_" + why].append(xa / p0["E"] - 1)
                    if P.get("fill") == "REFILL":
                        while queue:
                            s2 = queue.pop(0)
                            if s2 not in pos and not np.isnan(RO[k, s2]):
                                eq_now = base + cash_pnl + sum(v["q"] * mult[t] * (O[k, t] - v["E"]) for t, v in pos.items())
                                pr2 = px_raw(k, s2); q2 = p0["w"] * eq_now / (pr2 * mult[s2])
                                c2 = tc(q2, s2, pr2); cash_pnl -= c2; cost += c2 / max(eq_now, 1e-9); turn += q2 * mult[s2] * pr2 / eq_now
                                pos[s2] = dict(q=q2, E=O[k, s2], G=0.0, w=p0["w"]); ev["refill"] += 1; ev["positions"] += 1
                                break
                pending = []
            # 2) HL 版：日內觸發
            if P.get("mode") == "HL" and k < k1 - 1:
                for s in list(pos):
                    v = pos[s]; E = v["E"]
                    lv = []
                    if st:
                        lv.append((E * (1 - st), "stop"))
                    if tr and v["G"] > 0:
                        lv.append((E * (1 + (1 - tr) * v["G"]), "trail"))
                    if lv and k > k0:
                        lvl, why = max(lv)
                        if L[k, s] <= lvl:
                            xa = min(O[k, s], lvl); pr = px_raw(k, s) * xa / max(O[k, s], 1e-9)
                            cash_pnl += v["q"] * mult[s] * (xa - E)
                            c = v["q"] * (comm + mult[s] * (dm.TAX * pr + slip * o22.tick_arr(np.array([pr]))[0]))
                            cash_pnl -= c; cost += c / max(eq, 1e-9)
                            ev[why] += 1; ev["ret_" + why].append(xa / E - 1); pos.pop(s)
                            continue
                    v["G"] = max(v["G"], H[k, s] / E - 1)
            # 3) 收盤評價與（CLOSE 版）觸發
            vals[k] = base + cash_pnl + sum(v["q"] * mult[s] * (A[k, s] - v["E"]) for s, v in pos.items())
            if P.get("mode") != "HL" and k < k1 - 2:   # 隔日開盤仍在本期（到期日前）
                for s, v in pos.items():
                    g = A[k, s] / v["E"] - 1; v["G"] = max(v["G"], g)
                    if st and g <= -st:
                        pending.append((s, "stop")); ev["stop"] += 1
                    elif tr and v["G"] > 0 and g <= (1 - tr) * v["G"]:
                        pending.append((s, "trail")); ev["trail"] += 1
        # 到期結算
        eq = vals[k1 - 1]
        if n + 1 < len(X):
            for s, v in pos.items():
                sp = np.nan_to_num(R[k1 - 1, s], nan=np.nan_to_num(Rf[k1 - 1, s]))
                c = v["q"] * (comm + mult[s] * dm.TAX * sp); eq -= c; cost += c / eq
                ev["ret_expiry"].append(A[k1 - 1, s] / v["E"] - 1)
            vals[k1 - 1] = eq
    r = pd.Series(vals, days).dropna().pct_change().fillna(0.0)
    if lo is not None:
        r = r[(r.index >= lo) & (r.index < hi)]
    return dict(r=r, turn=turn, cost=cost, yrs=len(r) / 252, ev=ev)


def row(P, name, slip=1.0):
    COST["slip"] = slip
    full = simulate(P); out = []
    for per, lo, hi in PERIODS:
        o = simulate(P, lo, hi); s = m24.stats(m24.win(full["r"], lo, hi)); y = max(o["yrs"], 1e-9); e = o["ev"]
        npos = max(e["positions"], 1)
        out.append(dict(variant=name, slip=slip, period=per, trail=P.get("trail"), stop=P.get("stop"), mode=P.get("mode"), fill=P.get("fill"),
                        CAGR=s["CAGR"], sharpe=s["sharpe"], MDD=s["mdd"], calmar=s["CAGR"] / abs(s["mdd"]), turnover_yr=o["turn"] / y,
                        cost_pct_yr=o["cost"] / y * 100, positions=e["positions"], trail_exits=e["trail"], stop_exits=e["stop"], refills=e["refill"],
                        trail_hit=e["trail"] / npos, stop_hit=e["stop"] / npos,
                        avg_ret_trail=np.mean(e["ret_trail"]) if e["ret_trail"] else np.nan,
                        avg_ret_stop=np.mean(e["ret_stop"]) if e["ret_stop"] else np.nan,
                        avg_ret_expiry=np.mean(e["ret_expiry"]) if e["ret_expiry"] else np.nan,
                        _r=full["r"]))
    COST["slip"] = 1.0
    return out


def main():
    m32.setup(); load_hl()
    pd.set_option("display.width", 260)
    rows = []
    V = {"baseline (no exits)": dict(trail=None, stop=None, mode="CLOSE", fill="CASH"),
         "trail30 stop10 close->open, cash": dict(trail=0.3, stop=0.1, mode="CLOSE", fill="CASH"),
         "trail30 stop10 daily H/L, cash": dict(trail=0.3, stop=0.1, mode="HL", fill="CASH"),
         "trail30 stop10 close->open, refill": dict(trail=0.3, stop=0.1, mode="CLOSE", fill="REFILL"),
         "trail30 only": dict(trail=0.3, stop=None, mode="CLOSE", fill="CASH"),
         "stop10 only": dict(trail=None, stop=0.1, mode="CLOSE", fill="CASH")}
    curves = {}
    for k, P in V.items():
        for slip in (1.0, 0.5):
            rr = row(P, k, slip); rows += rr
            if slip == 1.0:
                curves[k] = rr[0]["_r"]
    grid = []
    for tr, st in itertools.product([0.2, 0.3, 0.4, 0.5, None], [0.05, 0.1, 0.15, None]):
        for r in row(dict(trail=tr, stop=st, mode="CLOSE", fill="CASH"), f"grid t{tr} s{st}"):
            grid.append(r)
    for per, lo, hi in PERIODS:
        s = m24.stats(m24.win(D["txr"], lo, hi))
        rows.append(dict(variant="TXF B&H", period=per, CAGR=s["CAGR"], sharpe=s["sharpe"], MDD=s["mdd"], calmar=s["CAGR"] / abs(s["mdd"])))
    curves["TXF B&H"] = D["txr"]
    T = pd.DataFrame(rows).drop(columns="_r"); T.to_csv(OUT / "ts_summary.csv", index=False)
    Gd = pd.DataFrame(grid).drop(columns="_r"); Gd.to_csv(OUT / "ts_grid.csv", index=False)
    t = T[T.slip != 0.5]
    print(t.pivot_table(index="variant", columns="period", values=["CAGR", "sharpe", "MDD"], sort=False).round(3).to_string())
    print(t[t.period.isin(["IS", "OOS"])][["variant", "period", "turnover_yr", "cost_pct_yr", "positions", "trail_exits", "stop_exits", "refills",
                                            "trail_hit", "stop_hit", "avg_ret_trail", "avg_ret_stop", "avg_ret_expiry"]].round(3).to_string(index=False))
    print(T[(T.slip == 0.5) & (T.period == "OOS")][["variant", "CAGR", "sharpe", "MDD"]].round(3).to_string(index=False))
    Gd["t"] = Gd.trail.fillna(0).map(lambda v: f"{int(v*100)}%" if v else "none"); Gd["s"] = Gd.stop.fillna(0).map(lambda v: f"{int(v*100)}%" if v else "none")
    for per in ("IS", "OOS", "OOS_ex2526"):
        for m in ("sharpe", "CAGR", "MDD"):
            print(per, m); print(Gd[Gd.period == per].pivot_table(index="t", columns="s", values=m, sort=False).round(3).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for k, r in curves.items():
        r = m24.win(r, IS0, END); e = (1 + r).cumprod()
        sty = dict(c="k", lw=2) if k == "TXF B&H" else dict(lw=1.1)
        axs[0].plot(e.index, e, label=k, **sty); axs[1].plot(e.index, e / e.cummax() - 1, **sty)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=8); axs[1].set_ylabel("drawdown")
    for a in axs:
        a.axvline(IS1, c="gray", lw=0.8)
    plt.tight_layout(); plt.savefig(OUT / "ts_equity_dd.png", dpi=110); plt.close()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, per in zip(axs, ("IS", "OOS", "OOS_ex2526")):
        pv = Gd[Gd.period == per].pivot_table(index="t", columns="s", values="sharpe", sort=False)
        im = ax.imshow(pv.values, cmap="RdYlGn"); ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index); ax.set_xlabel("stop"); ax.set_ylabel("trail"); ax.set_title(f"{per} Sharpe")
        for (i, j), v in np.ndenumerate(pv.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "ts_grid_sharpe.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
