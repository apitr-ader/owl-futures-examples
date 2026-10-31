"""77 月選 put 賣方：價平 / 價外 1% / 2%（x3% 對照）× 裸賣 / 價差(2%,4%) × {無停損, 2 倍點數停損}（承 76，同引擎）。

沿用 76 的 build / sim / 1 分鐘停損判定 / 成本 / 資金模擬；不做停利（76 顯示無益）。
ATM = 訊號日 TXF 收盤下方最近履約價（build 只取 K<F，故為略價外的價平）。
價差在買腿入場日無報價時跳過該月；輸出列出各結構缺月數。
輸出：results/dd_put_strike_{grid,equity,yearly,random,missing}.csv、dd_put_strike.png；快取 data/cache/（不入 repo）。
"""
import importlib.util, itertools, pickle, sys
import numpy as np, pandas as pd
from _rqo import *
spec = importlib.util.spec_from_file_location("m76", Path(__file__).parent / "76_put_tp_sl.py"); m = importlib.util.module_from_spec(spec); sys.modules["m76"] = m; spec.loader.exec_module(m)

m.STRUCTS = [("x", x, w) for x in (0.0, .01, .02, .03) for w in (0, .02, .04)]
m.TPS = [None]; m.CACHE = ROOT / "data/cache/dd_put_strike_trades.pkl"
END, IS0 = m.END, m.IS0
PER = (("IS", IS0, IS_END), ("OOS", IS_END, END + pd.Timedelta(days=1)))


def main():
    T = pickle.load(open(m.CACHE, "rb")) if m.CACHE.exists() else m.build()
    T = [t for t in T if t["s"] >= IS0]
    F = load_fut("2016-06-01"); days = F.index[(F.index <= END) & (F.index >= IS0)]
    m.ensure_minutes(T, days)
    G = {}
    for t in T: G.setdefault(t["st"], []).append(t)
    miss = []
    for st in m.STRUCTS:
        if st[2]:
            nX = set(t["X"] for t in G[(st[0], st[1], 0)]); sX = set(t["X"] for t in G.get(st, []))
            miss.append(dict(struct=m.sname(st), naked_n=len(nX), spread_n=len(sX), missing=len(nX - sX), missing_months=" ".join(sorted(nX - sX))))
    MS = pd.DataFrame(miss); MS.to_csv(RES / "dd_put_strike_missing.csv", index=False)
    rows, eqrows, yrows, ser, rb = [], [], [], {}, []
    for st in m.STRUCTS:
        for sl in m.SLS:
            need = {}; pc, wc, rec = m.run_variant(G[st], None, sl, days, need)
            key = (m.sname(st), "SL2x" if sl else "SL無"); ser[key] = pc
            r = dict(struct=key[0], sl=key[1], fallback_days=len(need))
            for nm, a, b in PER:
                for k, v in m.stats(pc, wc, rec, a, b).items(): r[f"{nm}_{k}"] = v
            m20 = rec[rec.X.isin(["202003", "202004"])]
            r.update(m2020_net=m20.net1.sum(), m2020_trades="/".join(m20.X + ":" + m20.why))
            for u in (.10, .25):
                pe, we, re = m.run_variant(G[st], None, sl, days, {}, u=u)
                for nm, a, b in PER:
                    s_ = m.stats(pe, we, re, a, b, eq0=m.CAP0) if len(pe[pe.index >= a]) else {}
                    r[f"{nm}_CAGR_u{int(u*100)}"] = s_.get("CAGR"); r[f"{nm}_MDD_u{int(u*100)}"] = s_.get("MDDw_pct")
                r[f"ruined_u{int(u*100)}"] = bool((m.CAP0 + pe.cumsum()).min() <= 0 or (len(re) and re.s.max() < T[-1]["s"] - pd.Timedelta(days=60)))
            if sl:
                rate = r["OOS_stop_rate"]
                sh = [m.stats(*m.run_variant(G[st], None, None, days, {}, rand=rate), IS_END, END + pd.Timedelta(days=1))["Sharpe"] for _ in range(200)]
                r["rand_med_OOS_Sharpe"] = np.median(sh); r["rand_pct"] = (np.array(sh) < r["OOS_Sharpe"]).mean()
            rows.append(r)
            for Y, g in rec[rec.s >= IS_END].groupby(rec.s.dt.year): yrows.append(dict(struct=key[0], sl=key[1], year=Y, n=len(g), net=g.net1.sum(), stops=(g.why == "stop").sum()))
    R = pd.DataFrame(rows); R.to_csv(RES / "dd_put_strike_grid.csv", index=False)
    Yd = pd.DataFrame(yrows); Yd.to_csv(RES / "dd_put_strike_yearly.csv", index=False)
    R[["struct", "sl", "rand_med_OOS_Sharpe", "rand_pct"]].dropna().to_csv(RES / "dd_put_strike_random.csv", index=False)
    pd.set_option("display.width", 400); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
    print(MS.to_string(index=False)); print(R.round(3).to_string(index=False))
    print(Yd.pivot_table(index=["struct", "sl"], columns="year", values="net").round(0).to_string())
    print(f"Spearman IS→OOS Sharpe {R.IS_Sharpe.rank().corr(R.OOS_Sharpe.rank()):.2f}")
    chart(ser)


def chart(ser):
    plt = plt_style(); fig, axs = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG); cols = [GOLD, BLUE, "#e07b39", GR]
    for ax, (ttl, ks) in zip(axs, (("裸賣（無停損 實線／2倍停損 虛線）", [(f"裸賣 x{x:.2f}", s) for x in (0, .01, .02, .03) for s in ("SL無", "SL2x")]),
                                   ("價差 w2% + 2倍停損（實線）／w4% + 2倍停損（虛線）", [(f"價差 x{x:.2f} w{w:.0%}", "SL2x") for x in (0, .01, .02, .03) for w in (.02, .04)]))):
        sty(ax)
        for i, k in enumerate(ks):
            s = ser[k]; s = s[s.index >= IS_END]; ax.plot(s.cumsum() / 1e4, color=cols[i // 2], lw=1.3, ls="-" if i % 2 == 0 else "--", label=" ".join(k))
        ax.set_title(f"OOS 2022+ 累積淨損益（萬元，1 組）：{ttl}"); ax.legend(facecolor=BG, labelcolor=WH, edgecolor=GR, fontsize=7)
    foot(fig); fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(RES / "dd_put_strike.png", dpi=110, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
