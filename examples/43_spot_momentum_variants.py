"""43 現股動能五變體（同 42 資料/可選池：上市櫃普通股含已下市、total-return 還原、20 日均成交金額 >= 5,000 萬）。
成本：手續費 0.1425%×5折 = 0.07125%/邊、證交稅 0.3%（賣）、滑價 1 tick/邊。月底收盤換股、前 10 名、絕對動能 > rf。
大盤濾網主版 = 舊 cmp（TXF 21/63/126 平均報酬 > 0）；另報無濾網。
  1 EW：波動調整 1/3/6m 複合排名、每檔 10%、不做目標波動。
  2 EW+停損：收盤 g = 收盤/進場價 - 1 <= -10% → 隔日開盤出場，資金留現金到下次換股。
  3 EW+停損+移動停利（同 35 CLOSE 版）：峰值 G = max g；G >= 10% 啟動，g <= (1-0.3)·G 出場；另 10% 停損。
  4 目標波動 15%（上限 1.0x，= 42 主設定，重算成本）。
  5 原始報酬平均排名：21/63/126 日報酬簡單平均（不經波動調整），EW、無停損。
成交：tools/spotsim.py（預設 t+1 開盤、漲跌停隔日重試；--fill close = 收盤成交）。停損/停利進場價 = 成交價，每次換股重設。下市：最後交易日後報酬記 0。
IS 2010-2019、OOS 2020-01 ~ 最新、OOS 除 2025-26。DSR：5 變體 × 2 濾網 = 10 試驗（且延續 42 等先前研究，實際試驗數更多）。
"""
import importlib.util, math
from pathlib import Path
import numpy as np, pandas as pd

_s = importlib.util.spec_from_file_location("m42", Path(__file__).with_name("42_spot_dual_momentum.py"))
m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m)
OUT = m.OUT; COMM, TAX, LIQ, N, ACT, TRAIL, STOP = 0.001425 * 0.5, 0.003, 5e7, 10, 0.10, 0.30, 0.10
V = {"1 EW": dict(), "2 EW+stop10": dict(stop=STOP), "3 EW+stop10+trail(act10,30%)": dict(stop=STOP, trail=TRAIL),
     "4 vol target 15%": dict(vt=0.15), "5 raw 1/3/6m avg": dict(rawrank=True)}


def main():
    pd.set_option("display.width", 250)
    adj, raw, tov, X, _ = m.load(); D = m.prep(); days = D["days"]
    tI = pd.Series((1 + m.m39.sim(X, np.ones(len(X)))["r"]).cumprod())
    sigs = m.market_signals(tI, days)
    rows, curves, srs = [], {}, []
    for f in ("cmp", "none"):
        for k, p in V.items():
            r, e, turn, cost, ev = m.ss.run(D, sigs[f], liq=LIQ, comm=COMM, tax=TAX, fill=m.FILL, act=ACT, **p); y = len(r) / 252; key = f"{k} [{f}]"; curves[key] = r
            x = m.win(r, m.IS1, None) - m.RF / 252; srs.append(x.mean() / x.std())
            for per, lo, hi in m.PER:
                ex = m.win(e, lo, hi)
                rows.append(dict(name=key, filter=f, period=per, **m.stats(m.win(r, lo, hi)), turnover_yr=turn / y, cost_pct_yr=cost / y * 100,
                                 invested=float((ex > 0.01).mean()), avg_expo=float(ex.mean()), positions=ev["pos"],
                                 stop_hit=ev["stop"] / max(ev["pos"], 1), trail_hit=ev["trail"] / max(ev["pos"], 1), blocked_up=ev["blocked_up"], blocked_dn=ev["blocked_dn"], orders=ev["orders"]))
            print(key, "done", flush=True)
    for row in rows:
        if row["period"] == "OOS":
            row["DSR"] = m.dsr(m.win(curves[row["name"]], m.IS1, None), srs, len(srs))
    end = days[-1]
    refs = {"0050 B&H": m.bench_0050(), "TXF B&H": m.m39.sim(X, np.ones(len(X)))["r"], "39 TXF SMA250": m.m39.run(X, m.m39.state(X, 250), 1, 0)["r"]}
    for k, r in refs.items():
        r = m.win(r, m.IS0, end + pd.Timedelta(days=1)); curves[k] = r
        for per, lo, hi in m.PER:
            rows.append(dict(name=k, filter="bench", period=per, **m.stats(m.win(r, lo, hi))))
    T = pd.DataFrame(rows); T.to_csv(OUT / "spot5_summary.csv", index=False)
    print(T.pivot_table(index="name", columns="period", values=["CAGR", "sharpe", "MDD"], sort=False).round(3).to_string())
    print(T[T.period == "OOS"].dropna(subset=["turnover_yr"])[["name", "turnover_yr", "cost_pct_yr", "invested", "avg_expo", "stop_hit", "trail_hit", "DSR"]].round(3).to_string(index=False))
    Y = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in curves.items() if "[none]" not in k})
    Y.to_csv(OUT / "spot5_yearly.csv"); print(Y.round(3).to_string()); print("data end", end.date())
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for k, r in curves.items():
        if "[none]" in k:
            continue
        e = (1 + r).cumprod(); s = dict(c="k", lw=2) if k == "0050 B&H" else dict(c="gray", lw=1) if "TXF" in k else dict(lw=1.2)
        axs[0].plot(e.index, e, label=k, **s); axs[1].plot(e.index, e / e.cummax() - 1, **s)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=8); axs[1].set_ylabel("drawdown")
    axs[0].set_title(f"43 spot momentum variants (cmp filter, top10, comm 50% off, tax 0.3%, 1 tick, fill={m.FILL})")
    for a in axs:
        a.axvline(m.IS1, c="gray", lw=.8); a.axvline(m.EX1, c="gray", lw=.6, ls=":")
    plt.tight_layout(); plt.savefig(OUT / "spot5_equity_dd.png", dpi=110)


if __name__ == "__main__":
    main()
