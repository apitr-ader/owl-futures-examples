"""61 指數成分股調整事件（research/reddit_quant_ideas.md #1 #2）：台灣50（FTSE TWSE Taiwan 50）納入／剔除個股。
事件名單：zh.wikipedia「臺灣50指數」成份股變動表（次級來源，只有「年月」）；不自行補造事件。只用 2011-03 起（季度 3/6/9/12 月）。
生效：FTSE 季度審核於 3/6/9/12 月第三個週五收盤後生效 → 指數基金於第三個週五收盤（F）調整。日期由規則推得（非逐筆公告日）。
  宣布日約在 F 前 2 週（未取得逐筆日期），故「宣布後追價」以 F−k 開盤進場近似。MSCI Taiwan 名單無可靠公開歷史 → 不測。
策略（相對 TAIEX 超額報酬，代表以 TXF/ETF 避險；個股日資料還原價）：
  pre：F−k 開盤進場（k=5/10），F 收盤出場；做多納入、放空剔除（順著指數需求）。
  post：F+1 開盤進場、F+h 收盤出場（h=5/10/20）；fade：放空納入、做多剔除。
  side：add / del / both。成本：手續費 0.1425%×50% 每邊、證交稅 0.3%（賣出邊）、滑價 1 tick 忽略；放空另扣借券費 1%/年。
  放空限制：融券可能受限（停券期間如除權息、股東會），且平盤下不得放空規則已放寬；實務可用個股期貨替代。
IS：2011-2019；OOS：2020+。隨機基準：同日期、隨機抽同市值區（當日成交值前 150 名）非事件股。
另測 TXF：F 日（調整日）日盤報酬與 F+1 反轉。
"""
import re, sys, math, itertools, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import _rq as R
sys.path.insert(0, "/workspace/fubon-threads/tools"); import stockdata as sd

RAW = Path("data/index/tw50_wiki_raw.txt"); COMM = 0.001425 * 0.5; TAX = 0.003; BORROW = 0.01
URL = "https://zh.wikipedia.org/w/index.php?title=%E8%87%BA%E7%81%A350%E6%8C%87%E6%95%B8&action=raw"


def events():
    if not RAW.exists(): RAW.write_bytes(urllib.request.urlopen(URL).read())
    t = RAW.read_text(); t = t[t.index("== 成份股變動 =="):]; t = t[:t.index("|}")]
    ev = []
    for row in t.split("|-"):
        m = re.match(r"\s*\|\s*(\d{4})年(\d{1,2})月\s*\|\|(.*?)\|\|(.*)", row.strip("\n"), re.S)
        if not m: continue
        y, mo = int(m[1]), int(m[2])
        for side, txt in (("add", m[3]), ("del", m[4])):
            for code in re.findall(r"(?:^|<br>|\s)(\d{4})\s", txt + " "): ev.append((y, mo, side, code))
    E = pd.DataFrame(ev, columns=["y", "m", "side", "code"])
    E = E[(E.y * 100 + E.m >= 201103) & E.m.isin([3, 6, 9, 12])]
    fr = lambda y, m: pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D").to_series().pipe(lambda s: s[(s.dt.month == m) & (s.dt.dayofweek == 4)]).iloc[2]
    E["F"] = [fr(y, m) for y, m in zip(E.y, E.m)]
    return E.reset_index(drop=True)


def main():
    E = events(); print("events", len(E), E.groupby("side").size().to_dict(), E.F.min().date(), E.F.max().date())
    tx = pd.read_parquet("data/index/taiex_daily.parquet"); tx["date"] = pd.to_datetime(tx.date); tx = tx.set_index("date")
    days = tx.index[tx.index >= "2010-06-01"]; tc = tx.close.reindex(days); to = tx.open.reindex(days)
    cache = {}
    def px(code):
        if code not in cache:
            try: a = sd.load_adjusted(code); a.index = pd.to_datetime(a.index); cache[code] = a.reindex(days)
            except FileNotFoundError: cache[code] = None
        return cache[code]
    def fpos(F):
        i = days.searchsorted(F); return i if i < len(days) and days[i] == F else i - 1   # 週五休市 → 前一交易日
    def ret(code, i0, i1):   # i0 開盤 → i1 收盤，超額 TAIEX
        a = px(code)
        if a is None or i0 < 0 or i1 >= len(days): return np.nan
        r = a.close.iloc[i1] / a.open.iloc[i0] - 1; m = tc.iloc[i1] / to.iloc[i0] - 1
        return r - m if r == r else np.nan
    rows = []
    for (ph, k), (_, e) in itertools.product([("pre", 5), ("pre", 10), ("post", 5), ("post", 10), ("post", 20)], E.iterrows()):
        f = fpos(e.F); i0, i1 = (f - k + 1, f) if ph == "pre" else (f + 1, f + k)
        x = ret(e.code, i0, i1)
        sgn = (1 if e.side == "add" else -1) * (1 if ph == "pre" else -1)
        short = sgn < 0; c = 2 * COMM + TAX + (BORROW * k / 252 if short else 0)
        rows.append(dict(F=e.F, y=e.F.year, side=e.side, code=e.code, ph=ph, k=k, ex=x, net=sgn * x - c))
    X = pd.DataFrame(rows).dropna(subset=["ex"])
    X.to_csv(R.OUT / "rq61_event_trades.csv", index=False)
    print("\nmean excess return vs TAIEX by phase/side (raw, not signed):")
    print(X.assign(per=np.where(X.y < 2020, "IS", "OOS")).groupby(["ph", "k", "side", "per"]).ex.agg(["mean", "count", lambda s: s.mean() / s.std() * math.sqrt(len(s))]).round(4).to_string())
    G = []
    for (ph, k), sides in itertools.product(X.groupby(["ph", "k"]).groups, (("add",), ("del",), ("add", "del"))):
        s = X[(X.ph == ph) & (X.k == k) & X.side.isin(sides)]
        ev = s.groupby("F").net.mean()   # 每事件日等權
        i, o = ev[ev.index.year < 2020], ev[ev.index.year >= 2020]
        sh = lambda v: v.mean() / v.std() * 2 if len(v) > 2 and v.std() > 0 else np.nan   # 年化：每年 4 次
        G.append(dict(key=f"{ph}{k}_{'+'.join(sides)}", IS_n=len(i), OOS_n=len(o), IS_mean=i.mean(), OOS_mean=o.mean(), IS_Sharpe=sh(i), OOS_Sharpe=sh(o),
                      IS_t=i.mean() / i.std() * math.sqrt(len(i)), OOS_t=o.mean() / o.std() * math.sqrt(len(o)), ev=ev))
    G = pd.DataFrame(G); sp = G.IS_Sharpe.rank().corr(G.OOS_Sharpe.rank())
    print("\n", G.drop(columns="ev").round(4).to_string(), "\nSpearman IS/OOS %.3f" % sp)
    best = G.sort_values("IS_Sharpe", ascending=False).iloc[0]; print("BEST IS", best.key)
    # WF：每年用過去 3 年事件挑最佳
    wf = []
    for Y in range(2020, 2026):
        tr = [g.ev[(g.ev.index.year >= Y - 3) & (g.ev.index.year < Y)] for g in G.itertuples()]
        b = int(np.nanargmax([t.mean() / t.std() if len(t) > 2 else np.nan for t in tr]))
        te = G.ev.iloc[b][G.ev.iloc[b].index.year == Y]; wf.append(te); print(Y, G.key.iloc[b], "test mean %.4f n %d" % (te.mean(), len(te)))
    wf = pd.concat(wf); print("WF stitched mean %.4f Sharpe(ann, 4/yr) %.3f n %d" % (wf.mean(), wf.mean() / wf.std() * 2, len(wf)))
    # 隨機基準（best 設定）：同 F、同階段，從當日成交值前 150 名非事件股隨機抽
    ph = best.key[:3] if best.key.startswith("pre") else "post"; k = int(re.findall(r"\d+", best.key)[0]); side = best.key.split("_")[1].split("+")
    tov = sd.panel("turnover", adjusted=False, start="2010-06-01").reindex(days).rolling(20).mean()
    rng = np.random.default_rng(1); sims = []
    evd = E[E.side.isin(side)]
    for rep in range(200):
        vals = []
        for e in evd.itertuples():
            f = fpos(e.F); top = tov.iloc[f - 15].nlargest(150).index.difference(E[E.F == e.F].code)
            cd = rng.choice(top); i0, i1 = (f - k + 1, f) if ph == "pre" else (f + 1, f + k)
            x = ret(cd, i0, i1); sgn = (1 if e.side == "add" else -1) * (1 if ph == "pre" else -1)
            if x == x: vals.append((e.F, sgn * x - 2 * COMM - TAX))
        v = pd.DataFrame(vals, columns=["F", "n"]).groupby("F").n.mean(); vo = v[v.index.year >= 2020]; sims.append(vo.mean() / vo.std() * 2)
    print("random OOS Sharpe mean %.3f [5%%,95%%] [%.3f, %.3f]" % (np.mean(sims), np.percentile(sims, 5), np.percentile(sims, 95)))
    # TXF：F 日日盤報酬、F+1 反轉
    D = pd.read_parquet(R.DATAD); D.index = D.date.dt.tz_localize(None).dt.normalize()
    Fs = sorted(set(E.F)); allF = [pd.Timestamp(f) for f in pd.date_range("2011-03-01", "2026-10-01", freq="MS") if f.month in (3, 6, 9, 12)]
    allF = [pd.date_range(f, periods=31, freq="D").to_series().pipe(lambda s: s[(s.dt.month == f.month) & (s.dt.dayofweek == 4)]).iloc[2] for f in allF]
    txr = []
    for F in allF:
        i = D.index.searchsorted(F)
        if i + 1 >= len(D) or D.index[i] != F: continue
        r0 = D.close.iloc[i] - D.open.iloc[i]; r1 = D.close.iloc[i + 1] - D.open.iloc[i + 1]
        txr.append(dict(F=F, fri=r0, next=r1, fade=-np.sign(r0) * r1 - R.cost_pts(D.open.iloc[i + 1], D.close.iloc[i + 1])))
    T = pd.DataFrame(txr); print("\nTXF rebalance Friday: mean day-session pts %.1f | next-day fade net mean %.1f pts, hit %.2f, n %d | IS %.1f OOS %.1f" % (
        T.fri.mean(), T.fade.mean(), (T.fade > 0).mean(), len(T), T[T.F.dt.year < 2020].fade.mean(), T[T.F.dt.year >= 2020].fade.mean()))
    plt = R.style(); fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for ph_, k_ in (("pre", 10), ("post", 20)):
        for sd_ in ("add", "del"):
            s = X[(X.ph == ph_) & (X.k == k_) & (X.side == sd_)].sort_values("F").groupby("F").ex.mean()
            ax[0].plot(s.index, s.cumsum() * 100, label=f"{ph_}{k_} {sd_}")
    ax[0].axvline(pd.Timestamp("2020-01-01"), c=R.GRAY, ls="--"); ax[0].set_title("台灣50 調整事件 累積超額報酬（未扣成本、未帶方向）"); ax[0].set_ylabel("%"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].bar(G.key, G.IS_Sharpe, alpha=.7, label="IS 2011-19"); ax[1].bar(G.key, G.OOS_Sharpe, alpha=.7, label="OOS 2020+", width=.4)
    ax[1].tick_params(axis="x", rotation=70, labelsize=7); ax[1].legend(); ax[1].set_title("各設定 Sharpe（扣成本，每年 4 事件年化）"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(R.OUT / "rq_61_tw50_events.png", dpi=100); print("chart results/rq_61_tw50_events.png")


if __name__ == "__main__":
    main()
