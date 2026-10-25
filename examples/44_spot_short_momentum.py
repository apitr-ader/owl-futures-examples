"""44 原始雙動能（43 #1：EW 前 10、無目標波動、無停損、波動調整排名、絕對動能 > rf）改用 1-4 週回看。
回看：單一 5/10/15/20 日；複合 5/10/15/20 日波動調整平均；對照 21/63/126（43 #1）。
換股：月（月底收盤）與週（每週最後交易日收盤）。絕對動能：同回看平均報酬 > 同期 rf（1.5%/年）。
濾網：主 cmp（TXF 21/63/126 平均報酬 > 0），另報無濾網。成本：手續費 0.07125%/邊、稅 0.3%、滑價 1 tick/邊；流動性 5,000 萬。
診斷：反轉組合 = 同分數排名最後 10 名（不設絕對動能門檻），僅供檢查短期反轉，不是候選策略。
IS 2010-2019、OOS 2020-01 ~ 最新。DSR 以本檔全部動能試驗數計（仍低估累積研究的試驗數）。
"""
import importlib.util, math
from pathlib import Path
import numpy as np, pandas as pd

_s = importlib.util.spec_from_file_location("m42", Path(__file__).with_name("42_spot_dual_momentum.py"))
m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m)
OUT = m.OUT; COMM, TAX, LIQ, N = 0.001425 * 0.5, 0.003, 5e7, 10
LB = {"1w": (5,), "2w": (10,), "3w": (15,), "4w": (20,), "1-4w comp": (5, 10, 15, 20), "1/3/6m ref": (21, 63, 126)}


def simulate(D, sig, legs, freq, rev=False):
    days, A, P, R, R0, traded, nhist, t20, lastpx = D
    if freq == "M":
        reb = np.r_[days[1:].month != days[:-1].month, True]
    else:
        wk = days.to_period("W-SUN"); reb = np.r_[wk[1:] != wk[:-1], True]
    thr = (1 + m.RF) ** (np.mean(legs) / 252) - 1
    w = np.zeros(A.shape[1]); out = np.zeros(len(days)); expo = np.zeros(len(days)); turn = cost = 0.0
    for t in range(len(days)):
        if t > 0:
            pr = w @ R0[t]; out[t] = pr; g = 1 + pr
            w = w * (1 + R0[t]) / g if g > 0 else w * 0
        expo[t] = w.sum()
        if not reb[t] or t < 130 or days[t] < m.IS0 - pd.Timedelta(days=40):
            continue
        tgt = np.zeros_like(w)
        if sig[t]:
            ok = np.where(traded[t] & (nhist[t] >= 127) & (t20[t] >= LIQ))[0]
            raw_m = np.zeros(len(ok)); vs = np.zeros(len(ok))
            for L in legs:
                rL = A[t, ok] / A[t - L, ok] - 1
                v = np.nanstd(R[t - L + 1:t + 1, ok], 0) * math.sqrt(L)
                raw_m += rL; vs += rL / np.maximum(v, 1e-6)
            raw_m /= len(legs); vs /= len(legs)
            fin = np.isfinite(vs) & np.isfinite(raw_m)
            if rev:
                cand = ok[fin][np.argsort(vs[fin])][:N]
            else:
                good = fin & (raw_m > thr); cand = ok[good][np.argsort(-vs[good])][:N]
            tgt[cand] = 1.0 / N
        d = tgt - w
        px = np.where(np.isnan(lastpx[t]), 1.0, lastpx[t]); slip = m.tick(px) / px
        c = (np.clip(d, 0, None) * (COMM + slip)).sum() + (np.clip(-d, 0, None) * (COMM + TAX + slip)).sum()
        out[t] -= c; cost += c; turn += np.abs(d).sum() / 2; w = tgt
    keep = days >= m.IS0
    return pd.Series(out, days)[keep], pd.Series(expo, days)[keep], turn, cost


def main():
    pd.set_option("display.width", 250)
    adj, raw, tov, X, _ = m.load()
    days = adj.index; A = adj.values; P = raw.values
    R = np.vstack([np.full(A.shape[1], np.nan), A[1:] / A[:-1] - 1]); traded = ~np.isnan(P)
    D = (days, A, P, R, np.nan_to_num(R), traded, np.cumsum(traded, 0),
         pd.DataFrame(np.where(traded, tov.values, 0.0)).rolling(20).mean().values, pd.DataFrame(P).ffill().values)
    tI = pd.Series((1 + m.m39.sim(X, np.ones(len(X)))["r"]).cumprod())
    sigs = m.market_signals(tI, days)
    rows, curves, srs = [], {}, []
    for f in ("cmp", "none"):
        for freq in ("M", "W"):
            for lk, legs in LB.items():
                for rev in (False, True):
                    if rev and lk == "1/3/6m ref":
                        continue
                    r, e, turn, cost = simulate(D, sigs[f], legs, freq, rev); y = len(r) / 252
                    key = f"{'REV ' if rev else ''}{lk} {freq} [{f}]"; curves[key] = r
                    if not rev:
                        x = m.win(r, m.IS1, None) - m.RF / 252; srs.append(x.mean() / x.std())
                    for per, lo, hi in m.PER:
                        rows.append(dict(name=key, lookback=lk, rebal=freq, filter=f, kind="reversal" if rev else "momentum", period=per,
                                         **m.stats(m.win(r, lo, hi)), turnover_yr=turn / y, cost_pct_yr=cost / y * 100,
                                         invested=float((m.win(e, lo, hi) > 0.01).mean())))
                    print(key, "done", flush=True)
    for row in rows:
        if row["period"] == "OOS" and row["kind"] == "momentum":
            row["DSR"] = m.dsr(m.win(curves[row["name"]], m.IS1, None), srs, len(srs))
    end = days[-1]
    refs = {"0050 B&H": m.bench_0050(), "39 TXF SMA250": m.m39.run(X, m.m39.state(X, 250), 1, 0)["r"]}
    for k, r in refs.items():
        r = m.win(r, m.IS0, end + pd.Timedelta(days=1)); curves[k] = r
        for per, lo, hi in m.PER:
            rows.append(dict(name=k, kind="bench", period=per, **m.stats(m.win(r, lo, hi))))
    T = pd.DataFrame(rows); T.to_csv(OUT / "spotw_summary.csv", index=False)
    pv = T.pivot_table(index="name", columns="period", values=["CAGR", "sharpe", "MDD"], sort=False)
    print(pv.round(3).to_string())
    print(T[T.period == "OOS"].dropna(subset=["turnover_yr"])[["name", "turnover_yr", "cost_pct_yr", "invested", "DSR"]].round(3).to_string(index=False))
    Y = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in curves.items() if "[cmp]" in k or "B&H" in k or "SMA" in k})
    Y.to_csv(OUT / "spotw_yearly.csv"); print("data end", end.date())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    show = [k for k in curves if ("[cmp]" in k and not k.startswith("REV") and (" M " in k or "1-4w" in k)) or k in refs or k == "REV 1w W [cmp]"]
    for k in show:
        e = (1 + curves[k]).cumprod(); s = dict(c="k", lw=2) if k == "0050 B&H" else dict(c="gray", lw=1) if "SMA" in k else dict(lw=1.1, ls="--" if "REV" in k or " W " in k else "-")
        axs[0].plot(e.index, e, label=k, **s); axs[1].plot(e.index, e / e.cummax() - 1, **s)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=7, ncol=2); axs[1].set_ylabel("drawdown")
    axs[0].set_title("44 spot dual momentum, 1-4 week lookbacks (EW top10, cmp filter, comm 50% off)")
    for a in axs:
        a.axvline(m.IS1, c="gray", lw=.8); a.axvline(m.EX1, c="gray", lw=.6, ls=":")
    plt.tight_layout(); plt.savefig(OUT / "spotw_equity_dd.png", dpi=110)


if __name__ == "__main__":
    main()
