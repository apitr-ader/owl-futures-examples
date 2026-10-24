"""31 排名區間動能：只要上市交易就可買、持有 10 檔、選動能排名 5-14 / 10-19（對照 1-10）。

可選池（每個到期日 x）：
  - 契約當日有成交且收盤價 > 0；累計交易日 >= 22（足以計算最短的 21 日腿）。
  - 取消 252 日歷史、成交量門檻、永久換月跳空排除。
  - 同標的只取一檔（標準契約優先，當日無成交才用小型）。
  - 仍排除 ETF 與代號結尾 '1' 的調整股數契約（其標準契約通常同時存在，連續調整序列較短且不完整）。
  - 新上市股票：複合分數只用「歷史足夠」的腿（21 必需；63、126 有就用），lb20 需要 21 日。
  - 換倉日新近月需有開盤價，否則該槽空手（同 26）。
排名區間（10 檔）：B1_10 = 第 1-10 名；B5_14 =「5-15 名」取第 5-14 名；B10_19 =「10-20 名」取第 10-19 名。
排名在全體可選池上依分數由高到低；絕對動能濾網（原始動能 > 無風險門檻）後才持有，不遞補區間外名次。
緩衝：已持有標的若排名仍在 [lo-5, hi+5] 且過濾網則續抱，其餘由區間內依排名補滿 10 檔。
分數：RAW 複合（21/63/126 平均）、VS 複合（27 的波動調整）、LB20。大盤濾網：TXF 21/63/126 平均報酬 > 0。
權重：EW（每檔 1/10）或 IV（60 日波動倒數，總和 = 選中檔數/10）；目標波動 0 / 15% / 25%（最多 2 倍）。
成本：手續費 50 或 25 元/口/邊，交易稅 2e-5，滑價 1 或 0.5 tick，無換月成本（到期換倉）。
樣本內 2014-2019 依參數鄰域平台挑選（每個區間各挑一組），樣本外 2020-2026/10。
注意：products.csv 只含目前仍掛牌的商品 → 存活者偏差（下市股票期貨不在樣本中），結果偏樂觀。
"""
import importlib.util
import itertools
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location("mv28", Path(__file__).with_name("28_mom_over_vol.py"))
m28 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m28)
m26, m27, m24, o22, dm, D = m28.m26, m28.m27, m28.m24, m28.o22, m28.dm, m28.D
OUT = Path("results")
NH = 10
MINH = 22
LEGS = (21, 63, 126)
BANDS = {"B1_10": (1, 10), "B5_14": (5, 14), "B10_19": (10, 19)}
G = dict(score=["RAW", "VS", "LB20"], band=list(BANDS), mkt=[0, 1], vt=[0.0, 0.15, 0.25], scheme=["EW", "IV"], buf=[0, 1])
IS0, IS1, EX1, END = m26.IS0, m26.IS1, m26.EX1, m26.END
PERIODS = m26.PERIODS
COST = dict(slip=1.0, comm=50.0)
C = {}


def setup():
    dm.MIN_HIST = MINH            # load() 只保留至少 22 根日K 的商品（原本 252）
    o22.prepare("data/stock_futures", "data/TXF_1c_D_regular_2010-01-01_2026-10-22.parquet")
    D["exp"] = m26.expiry_calendar()
    tI = (1 + D["txr"]).cumprod()
    D["mktsig"] = np.mean([(tI / tI.shift(L) - 1).values for L in LEGS], 0)
    M = D["M"]
    traded, R = M["traded"].values, np.nan_to_num(M["R"].values)
    nh = M["nhist"].values
    mini = M["mini"].values.astype(bool)
    I, ret = D["I"], D["ret"]
    days = D["days"]
    X = [x for x in D["exp"] if IS0 - pd.Timedelta(days=40) <= days[x] < END]
    thr = (1 + dm.RF) ** (np.mean(LEGS) / 252) - 1
    univ, scores = {}, {}
    for x in X:
        ok = traded[x] & (R[x] > 0) & (nh[x] >= MINH)
        seen, U = set(), []
        for j in np.lexsort((mini, D["und"])):
            if ok[j] and D["und"][j] not in seen:
                seen.add(D["und"][j]); U.append(j)
        U = np.array(sorted(U), int)
        univ[x] = U
        h = nh[x, U]
        raw, vs, cnt = np.zeros(len(U)), np.zeros(len(U)), np.zeros(len(U))
        for L in LEGS:
            m = h >= L + 1
            r = np.where(m, I[x, U] / I[max(x - L, 0), U] - 1, 0.0)
            v = ret[x - L + 1:x + 1, U].std(0) * math.sqrt(L)
            raw += r; vs += np.where(m, r / np.maximum(v, 1e-6), 0.0); cnt += m
        raw, vs = raw / cnt, vs / cnt
        l20 = I[x, U] / I[x - 20, U] - 1
        scores[x] = {"RAW": (raw, raw > thr), "VS": (vs, raw > thr), "LB20": (l20, l20 > 0)}
    D.update(X=X, univ=univ, scores=scores)


def weights(x, sel, scheme, vt):
    X = D["ret"][x - 59:x + 1, sel]
    if scheme == "EW":
        w = np.full(len(sel), 1.0 / NH)
    else:
        iv = 1 / np.maximum(X.std(0), 1e-4); w = iv / iv.sum() * len(sel) / NH
    if vt:
        S = dm.ledoit_wolf(X)
        w = w * min(vt / math.sqrt(max(w @ S @ w, 1e-12) * 252), 2.0)
    return w


def simulate(P, lo=None, hi=None, trace=False):
    days, mult, A, O, R, RO = D["days"], D["mult"], D["A"], D["O"], D["R"], D["RO"]
    V = D["M"]["V"].values
    slip, comm = COST["slip"], COST["comm"]
    lo_, hi_ = (IS0 if lo is None else lo), (END if hi is None else hi)
    X = [x for x in D["X"] if lo_ - pd.Timedelta(days=40) <= days[x] < hi_]
    nd = len(days); eq = 1.0; vals = np.full(nd, np.nan)
    turn = c_open = c_set = 0.0
    held, gross, recs = [], [], []
    b0, b1 = BANDS[P["band"]]
    for n, x in enumerate(X):
        k0 = x + 1
        if k0 >= nd:
            break
        k1 = X[n + 1] + 1 if n + 1 < len(X) else nd
        U = D["univ"][x]
        sel = []
        if len(U) >= dm.MIN_UNIVERSE and not (P["mkt"] and not D["mktsig"][x] > 0):
            sc, ok = D["scores"][x][P["score"]]
            o = np.argsort(-sc); rk = U[o]; okr = ok[o]
            pos = {int(s): r + 1 for r, s in enumerate(rk)}
            okd = {int(s): bool(v) for s, v in zip(rk, okr)}
            keep = [s for s in held if s in pos and b0 - 5 <= pos[s] <= b1 + 5 and okd[s]] if P["buf"] else []
            band = [int(s) for s, v in zip(rk[b0 - 1:b1], okr[b0 - 1:b1]) if v and int(s) not in keep]
            sel = (keep + band)[:NH]
        sel = [s for s in sorted(sel) if not np.isnan(RO[k0, s])]
        held = sel
        if sel:
            s_ = np.array(sel); w_ = weights(x, s_, P["scheme"], P["vt"])
            gross.append(w_.sum())
            q = w_ * eq / (RO[k0, s_] * mult[s_]); px = RO[k0, s_]
            c = q @ (comm + mult[s_] * (dm.TAX * px + slip * o22.tick_arr(px)))
            turn += (q * mult[s_] * px).sum() / eq
            eq -= c; c_open += c / eq
            vals[k0:k1] = eq + (A[k0:k1, s_] - O[k0, s_]) @ (q * mult[s_])
            eq = vals[k1 - 1]
            if n + 1 < len(X):
                sp = np.nan_to_num(R[k1 - 1, s_], nan=np.nan_to_num(D["M"]["Rf"].values[k1 - 1, s_]))
                c = q @ (comm + mult[s_] * dm.TAX * sp)
                eq -= c; c_set += c / eq; vals[k1 - 1] = eq
            if trace:
                vol = np.nanmean(V[max(x - 19, 0):x + 1, s_], 0)
                recs += [dict(date=days[x], end=days[min(k1 - 1, nd - 1)], j=int(s), und=D["und"][s], w=float(w), adv20=float(v))
                         for s, w, v in zip(s_, w_, vol)]
        else:
            gross.append(0.0); vals[k0:k1] = eq
        if n == 0 and k0 > 0:
            vals[k0 - 1] = 1.0
    r = pd.Series(vals, days).dropna().pct_change().fillna(0.0)
    if lo is not None:
        r = r[(r.index >= lo) & (r.index < hi)]
    return dict(r=r, turn=turn, c_open=c_open, c_set=c_set, yrs=len(r) / 252, gross=float(np.mean(gross)) if gross else 0.0, recs=recs)


def _work(P):
    o = simulate(P)
    row = dict(P)
    for nm, lo, hi in PERIODS:
        row[nm + "_sharpe"] = m24.stats(m24.win(o["r"], lo, hi))["sharpe"]
    return row


def plateau(res):
    key = list(G); v = res.set_index(key).IS_sharpe; out = []
    for idx in v.index:
        nb = [v[idx]]
        for i, k in enumerate(key):
            if k == "band":
                continue
            for alt in G[k]:
                t = list(idx); t[i] = alt; t = tuple(t)
                if t != idx and t in v.index:
                    nb.append(v[t])
        out.append(np.median(nb))
    return out


def winners():
    days, I, cols = D["days"], D["I"], D["cols"]
    traded = D["M"]["traded"].values
    mini = D["M"]["mini"].values.astype(bool)
    rep = {}
    for j in np.lexsort((mini, D["und"])):
        rep.setdefault(D["und"][j], j)
    ye = pd.Series(range(len(days)), days).groupby(days.year).last()
    rows = []
    for u, j in rep.items():
        for y in range(2014, 2027):
            a, b = ye.get(y - 1), ye.get(y)
            if a is None or b is None:
                continue
            t = traded[a + 1:b + 1, j]
            if t.sum() < 20:
                continue
            first = a + 1 + np.argmax(t)
            rows.append(dict(und=u, year=y, ret=I[b, j] / I[max(a, first - 1), j] - 1))
    yr = pd.DataFrame(rows)
    return set(map(tuple, yr.sort_values("ret", ascending=False).groupby("year").head(10)[["year", "und"]].values))


def capture(recs, W, lo, hi):
    """強勢股月份中被持有的比例（期間以結束日年份歸年，同 30）。"""
    days = D["days"]
    X = [x for x in D["X"] if pd.Timestamp(lo) <= days[x] < pd.Timestamp(hi)]
    held = {(r["date"], r["und"]) for r in recs}
    tot = hit = 0
    for n, x in enumerate(X):
        k1 = X[n + 1] if n + 1 < len(X) else len(days) - 1
        y = days[k1].year
        for (yy, u) in W:
            if yy == y:
                tot += 1; hit += (days[x], u) in held
    return hit / max(tot, 1)


def evaluate(P, sr0, W, variant):
    rows = []
    full = simulate(P, trace=True)
    for slip, comm in ((1.0, 50.0), (0.5, 50.0), (1.0, 25.0), (0.5, 25.0)):
        COST.update(slip=slip, comm=comm)
        full = simulate(P, trace=True)
        for per, lo, hi in PERIODS:
            o = simulate(P, lo, hi, trace=True)
            s = m24.stats(m24.win(full["r"], lo, hi)); y = max(o["yrs"], 1e-9)
            rec = pd.DataFrame(o["recs"])
            rows.append(dict(variant=variant, band=P["band"], slip=slip, comm=comm, period=per, cfg=" ".join(f"{k}={v}" for k, v in P.items()),
                             CAGR=s["CAGR"], vol=s["vol"], sharpe=s["sharpe"], mdd=s["mdd"], turnover_yr=o["turn"] / y,
                             cost_pct_yr=(o["c_open"] + o["c_set"]) / y * 100, gross=o["gross"],
                             DSR=dm.dsr(s["sharpe"], s["n"], s["skew"], s["kurt"], sr0) if per == "OOS" else np.nan, sr0=sr0,
                             winner_capture=capture(o["recs"], W, lo, hi),
                             held_adv20_median=rec.adv20.median() if len(rec) else np.nan,
                             held_lt50_share=(rec.adv20 < 50).mean() if len(rec) else np.nan,
                             held_lt50_weight=(rec.w * (rec.adv20 < 50)).sum() / rec.w.sum() if len(rec) else np.nan))
        if slip == 1.0 and comm == 50.0:
            curve, holds = full["r"], pd.DataFrame(full["recs"])
    COST.update(slip=1.0, comm=50.0)
    return rows, curve, holds


def main():
    setup()
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    Cfg = [dict(zip(G, v)) for v in itertools.product(*G.values())]
    with Pool(6) as pool:
        res = pd.DataFrame(pool.map(_work, Cfg, chunksize=6))
    res["plateau"] = plateau(res)
    res.to_csv(OUT / "band_grid.csv", index=False)
    sr0 = dm.expected_max_sr(len(res), float(res.IS_sharpe.var()))
    rho = res[["IS_sharpe", "OOS_sharpe"]].rank().corr().iloc[0, 1]
    print(f"trials {len(res)} SR0 {sr0:.3f} Spearman(IS,OOS) {rho:.3f}")
    sens = pd.concat([res.groupby(k)[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median().assign(param=k) for k in G])
    sens.to_csv(OUT / "band_sensitivity.csv"); print(sens.round(3).to_string())
    W = winners()
    tab, curves, holds = [], {}, {}
    for b in BANDS:
        p = res[res.band == b].sort_values(["plateau", "IS_sharpe"], ascending=False).iloc[0]
        P = {k: (p[k].item() if hasattr(p[k], "item") else p[k]) for k in G}
        rr, cv, hd = evaluate(P, sr0, W, f"pick {b}"); tab += rr; curves[f"pick {b}"] = cv; holds[b] = hd
    # 固定設定對照：RAW 分數、無大盤濾網、無目標波動、EW、無緩衝（純「排名區間」效果）
    for b in BANDS:
        P = dict(score="RAW", band=b, mkt=0, vt=0.0, scheme="EW", buf=0)
        rr, cv, _ = evaluate(P, sr0, W, f"plain {b}"); tab += rr; curves[f"plain {b}"] = cv
    T = pd.DataFrame(tab)
    # 基準
    bm = {"TXF B&H": D["txr"]}
    ew = pd.Series(0.0, D["days"]); X = D["X"]
    for n, x in enumerate(X[:-1]):
        U = D["univ"][x]; ew.iloc[x + 1:X[n + 1] + 1] = D["ret"][x + 1:X[n + 1] + 1, U].mean(1)
    bm["EW universe (any listed)"] = ew
    for k, r in bm.items():
        for per, lo, hi in PERIODS:
            s = m24.stats(m24.win(r, lo, hi))
            T = pd.concat([T, pd.DataFrame([dict(variant=k, band="-", slip=np.nan, comm=np.nan, period=per, CAGR=s["CAGR"], vol=s["vol"],
                                                 sharpe=s["sharpe"], mdd=s["mdd"], gross=1.0)])])
    prior = pd.read_csv(OUT / "mv_summary.csv")
    prior = prior[(prior.slip == 1.0) & prior.variant.isin(["27 composite pick", "26 lb20 monthly", "best_s126"])]
    prior = prior.assign(variant="prior " + prior.variant, band="-", comm=50.0)
    T = pd.concat([T, prior[[c for c in T.columns if c in prior.columns]]])
    T.to_csv(OUT / "band_summary.csv", index=False)
    for b, h in holds.items():
        h.to_csv(OUT / f"band_holdings_{b}.csv", index=False)
    cols = ["variant", "slip", "comm", "period", "CAGR", "sharpe", "mdd", "turnover_yr", "cost_pct_yr", "gross", "DSR", "winner_capture",
            "held_adv20_median", "held_lt50_share", "held_lt50_weight"]
    print(T[(T.slip.isna()) | ((T.slip == 1.0) & (T.comm == 50.0))][cols].round(3).to_string(index=False))
    print("\ncost sensitivity OOS sharpe/CAGR:")
    print(T[(T.period == "OOS") & T.variant.str.startswith(("pick", "plain"))].pivot_table(index="variant", columns=["slip", "comm"], values=["sharpe", "CAGR"]).round(3).to_string())
    print("\npicks:"); print(T[(T.period == "IS") & (T.slip == 1.0) & (T.comm == 50.0) & T.variant.str.startswith("pick")][["variant", "cfg"]].to_string(index=False))
    print("universe size median IS/OOS:", np.median([len(D["univ"][x]) for x in X if D["days"][x] < IS1]), np.median([len(D["univ"][x]) for x in X if D["days"][x] >= IS1]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    for k, r in {**curves, **bm}.items():
        r = m24.win(r, IS0, END)
        ax.plot(r.index, (1 + r).cumprod(), label=k, lw=1.2, ls="--" if k in bm else "-")
    ax.axvline(IS1, c="gray", lw=0.8); ax.set_yscale("log"); ax.legend(fontsize=8); ax.set_title("Rank-band momentum (10 names, any listed stock future), 1 tick / 50 TWD")
    plt.tight_layout(); plt.savefig(OUT / "band_equity.png", dpi=110); plt.close()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, sc in zip(axs, G["score"]):
        g = res[res.score == sc]
        for b in BANDS:
            gg = g[g.band == b]; ax.scatter(gg.IS_sharpe, gg.OOS_sharpe, s=12, label=b)
        ax.set_title(sc); ax.set_xlabel("IS Sharpe"); ax.axhline(0, c="k", lw=0.5)
    axs[0].set_ylabel("OOS Sharpe"); axs[0].legend(); plt.tight_layout(); plt.savefig(OUT / "band_is_vs_oos.png", dpi=110); plt.close()
    sb = res.groupby(["band", "score"])[["IS_sharpe", "OOS_sharpe", "OOS_ex2526_sharpe"]].median().unstack(1)
    sb.to_csv(OUT / "band_by_score.csv"); print(sb.round(3).to_string())


if __name__ == "__main__":
    main()
