"""58 Rogue Overnight Volume Profile（夜盤分價量 OVAH / OPOC / OVAL 首觸）台指期 1 分 K 回測 + 樣本內最佳化。

參考：TradingView「Rogue Overnight Volume Profile」(v5pFekBV, RogueLLC33)；research/tradingview_strategies.md 第二輪 I（建議優先回測第 5 項）。
該腳本是「指標」，非 strategy（頁面無任何績效統計）。頁面描述的計算：
  - 夜盤（overnight）時段內建立 volume profile（最多 1000 列）、Value Area 預設 70%、POC；
  - 畫 OVAH / OPOC / OVAL 並延伸到 RTH（紐約盤）；
  - First Touch：RTH 期間第一次碰到 OVAH / OPOC / OVAL 標記；
  - Break & Retest：價格在水平線外「接受」(acceptance) 後回測，標出多空回測；
  - 另有 Weekly VWAP、5 期 MA、volume surge、HVN（本回測未實作，屬裝飾 / 另外的濾網）。
  頁面沒有給出交易規則，下列規則是我依「Best Used For：fade and acceptance trades around OVAH/OPOC/OVAL、break-and-retest」改寫成可回測的版本。

規則（台北時間；框架、資料、成本、成交慣例同 55/56）：
  Profile：該交易日之前的夜盤（15:00–05:00，單一合約，與隔日日盤同一契約 → 換月不會跨在任何一筆交易中）所有 1 分 K，
    成交量均分到該 K 的 low..high 每一個 1 點價列（TXF 最小跳動 1 點）。POC = 量最大價列（同量取最接近中點）；
    VA 以 POC 起算，每次比較上方 2 列與下方 2 列的量和，加入較大者，直到 ≥ VA%（60/70/80）。OVAH/OVAL = VA 上/下緣。
  首觸：日盤 08:45 起，第一根 K 其範圍（含前一根收盤→本根開盤的跳空）穿過水平線即為首觸；接近方向 ap = 前一根收盤（首根用開盤）在線的哪一側。
  進場（訊號在 K 收盤，下一根開盤成交 + 1 點滑價；13:00 以後不開新倉，同時只持 1 部位，每條水平線每日最多 1 筆）：
    hold_any   ：首觸那根收盤即反向（接近方向的反方向）= 做「水平線守住」的 fade（自 VAH 下方碰 → 空，自上方碰 → 多）。
    hold_rej   ：同上，但首觸 K 需收在接近側（被拒絕）才進場，否則該線作廢。
    brk_acc    ：首觸後連續 3 根 1 分 K 收盤在線的另一側（接受 acceptance）→ 順突破方向進場。
    brk_retest ：acceptance 後，價格回測水平線（low ≤ 線，多單）且該根收盤仍在線外 → 進場（Break & Retest）。回測收回線內則重新等 acceptance。
  水平線集合：VAH / VAL / POC / VA（VAH+VAL 兩條）。
  停損：進場價 ∓ s×D，D = 前 20 個日盤的平均 (高−低)，s ∈ {0.1,0.2,0.4}（無單位，跨指數水位 10000→46000 可比）。
  停利：m×停損距離（m = 1, 2；限價，無滑價）或不設。時間出場：進場後 60 / 180 分，或日盤收盤（皆最晚日盤收盤強平，市價 1 點滑價）。
  同根 K 同時碰停損與停利 → 停損；成交當根碰停損 → 停損、成交當根不算停利；跳空越過停損 → 以開盤價 − 滑價。
  成本：每邊 50 元（0.25 點）+ 2e-5×價格 期交稅 + 市價 / 停損出場 1 點滑價；1 點 = 200 元（TX 大台）。
  換月：全部持倉均在日盤內平倉，不跨夜盤（換月在夜盤 15:00 發生），故無換月缺口 / 換月成本。
  報酬以固定 100 萬元/口（不複利）計；MDD 盤中最壞情況 = 以每筆 MAE（最大不利價格）逐筆評估淨值最低點。
  IS 2017-06~2021；OOS 2022~；目標 IS 日損益 Sharpe（至少 150 筆）。walk-forward = 每年用前 3 年重選最佳參數。
不 commit 任何資料；從 repo 外的 workspace 根目錄執行：  .venv-pq/bin/python repo/examples/58_txf_overnight_volume_profile.py
"""
import sys, math, itertools, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
_sp = importlib.util.spec_from_file_location("sb55", HERE / "55_txf_ict_silver_bullet.py"); sb = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(sb)
load_bars = sb.load_bars
DATA = sb.DATA
OUT = Path("results"); CAP = 1_000_000; PV = 200.0; IS_END = pd.Timestamp("2022-01-01"); MIN_IS = 150
VAS = (0.6, 0.7, 0.8)
LVSETS = {"VAH": (0,), "VAL": (1,), "POC": (2,), "VA": (0, 1)}  # LV 欄 0=VAH 1=VAL 2=POC
STYLES = ("hold_any", "hold_rej", "brk_acc", "brk_retest")
STOPS = (0.1, 0.2, 0.4); TGTS = (1.0, 2.0, 0.0); TIMES = (60, 180, 0)  # tgt 0 = 不設；time 0 = 日盤收盤
ACC = 3


# ---------------------------------------------------------------- profile
@njit(cache=True)
def profile(h, l, v, i0, i1, vapct):
    mn = 1e18; mx = -1e18
    for i in range(i0, i1 + 1):
        mn = min(mn, l[i]); mx = max(mx, h[i])
    lo = int(round(mn)); n = int(round(mx)) - lo + 1
    vol = np.zeros(n)
    for i in range(i0, i1 + 1):
        a = int(round(l[i])) - lo; b = int(round(h[i])) - lo
        w = v[i] / (b - a + 1)
        for r in range(a, b + 1): vol[r] += w
    tot = vol.sum(); mid = (n - 1) / 2.0
    poc = 0; best = -1.0
    for r in range(n):
        if vol[r] > best + 1e-9 or (abs(vol[r] - best) <= 1e-9 and abs(r - mid) < abs(poc - mid)): best = vol[r]; poc = r
    a = poc; b = poc; acc = vol[poc]; tgt = vapct * tot
    while acc < tgt and (a > 0 or b < n - 1):
        up = -1.0; dn = -1.0
        if b < n - 1: up = vol[b + 1] + (vol[b + 2] if b + 2 < n else 0.0)
        if a > 0: dn = vol[a - 1] + (vol[a - 2] if a - 2 >= 0 else 0.0)
        if up >= dn:
            acc += vol[b + 1]
            if b + 2 < n: acc += vol[b + 2]
            b = min(b + 2, n - 1)
        else:
            acc += vol[a - 1]
            if a - 2 >= 0: acc += vol[a - 2]
            a = max(a - 2, 0)
    return lo + poc, lo + b, lo + a


def build(d):
    ts = d.ts.values
    sess = d.groupby(["skey", "night"]).agg(i0=("ts", lambda x: x.index[0]), i1=("ts", lambda x: x.index[-1]), n=("ts", "size"),
                                            hi=("high", "max"), lo=("low", "min")).reset_index()
    day = sess[~sess.night].set_index("skey"); ngt = sess[sess.night].set_index("skey")
    days = day.index.values
    h, l, v = d.high.values, d.low.values, d.volume.values
    rng = (day.hi - day.lo)
    D20 = rng.shift(1).rolling(20, min_periods=5).mean()
    rows = []
    for N0 in ngt.index:
        nr = ngt.loc[N0]; nd = days[days > N0]
        if not len(nd) or nr.n < 300: continue
        Dd = nd[0]; r = day.loc[Dd]
        if (pd.Timestamp(Dd) - pd.Timestamp(N0)).days > 5: continue
        if r.i0 != nr.i1 + 1: continue  # 夜盤須緊接該日盤
        t0 = pd.Timestamp(Dd) + pd.Timedelta("8h45min")
        if d.ts.iloc[r.i0] > t0 + pd.Timedelta(minutes=15) or r.n < 200: continue
        cut = np.searchsorted(ts[r.i0:r.i1 + 1], np.datetime64(pd.Timestamp(Dd) + pd.Timedelta("13h")))
        if np.isnan(D20.loc[Dd]) or cut < 60: continue
        row = dict(tdate=pd.Timestamp(Dd), s=r.i0, ce=r.i0 + cut, fe=r.i1 + 1, D=D20.loc[Dd], nclose=d.close.values[nr.i1], nrange=nr.hi - nr.lo)
        for va in VAS:
            poc, vah, val = profile(h, l, v, nr.i0, nr.i1, va)
            row[f"vah{int(va*100)}"] = vah; row[f"val{int(va*100)}"] = val; row["poc"] = poc
        rows.append(row)
    return pd.DataFrame(rows).sort_values("tdate").reset_index(drop=True)


# ---------------------------------------------------------------- simulation
@njit(cache=True)
def manage2(o, h, l, c, j0, fe, dr, ep, stop, tgt):
    """從 j0 下一根起管理（j0 成交 K 已檢查停損）。回傳 xp, kind(1 停損 2 停利 3 時間/收盤), amb, xj, mae。"""
    mae = (ep - l[j0]) if dr > 0 else (h[j0] - ep)
    for j in range(j0 + 1, fe):
        if dr > 0:
            hs = l[j] <= stop; ht = h[j] >= tgt
            mae = max(mae, ep - l[j])
            if hs: return min(o[j], stop) - 1.0, 1, 1 if ht else 0, j, mae
            if ht: return max(o[j], tgt), 2, 0, j, mae
        else:
            hs = h[j] >= stop; ht = l[j] <= tgt
            mae = max(mae, h[j] - ep)
            if hs: return max(o[j], stop) + 1.0, 1, 1 if ht else 0, j, mae
            if ht: return min(o[j], tgt), 2, 0, j, mae
    return c[fe - 1] - dr * 1.0, 3, 0, fe - 1, mae


@njit(cache=True)
def run(o, h, l, c, S0, CE, FE, LV, STP, active, style, tm, T):
    """out[k, slot, :] = dir, ep, stop, xp, kind, amb, risk, mae, ej, xj；每日最多 3 筆。"""
    n = len(S0); out = np.full((n, 3, 10), np.nan)
    for k in range(n):
        ph = np.zeros(3, np.int64); cnt = np.zeros(3, np.int64); ap = np.zeros(3, np.int64); used = np.zeros(3, np.bool_)
        j = S0[k]; ns = 0; S = STP[k]
        while j < CE[k] - 1 and ns < 3:
            sig = 0; sm = -1
            for m in range(3):
                if not active[m] or used[m]: continue
                L = LV[k, m]; ref = o[j] if j == S0[k] else c[j - 1]
                if ph[m] == 0:
                    if min(ref, l[j]) <= L and L <= max(ref, h[j]):
                        a = ref - L
                        if a == 0: used[m] = True; continue
                        ap[m] = 1 if a > 0 else -1
                        if style == 0: sig = ap[m]; sm = m
                        elif style == 1:
                            if (c[j] - L) * ap[m] > 0: sig = ap[m]; sm = m
                            else: used[m] = True
                        else:
                            ph[m] = 1; cnt[m] = 1 if (c[j] - L) * ap[m] < 0 else 0
                            if cnt[m] >= ACC:
                                if style == 2: sig = -ap[m]; sm = m
                                else: ph[m] = 2
                elif ph[m] == 1:
                    if (c[j] - L) * ap[m] < 0: cnt[m] += 1
                    else: cnt[m] = 0
                    if cnt[m] >= ACC:
                        if style == 2: sig = -ap[m]; sm = m
                        else: ph[m] = 2
                else:
                    dr_ = -ap[m]
                    if dr_ > 0:
                        if l[j] <= L:
                            if c[j] > L: sig = 1; sm = m
                            else: ph[m] = 1; cnt[m] = 0
                    else:
                        if h[j] >= L:
                            if c[j] < L: sig = -1; sm = m
                            else: ph[m] = 1; cnt[m] = 0
                if sig != 0: break
            if sig == 0:
                j += 1; continue
            used[sm] = True
            ej = j + 1; dr = float(sig)
            ep = o[ej] + dr; stp = ep - dr * S; tg = ep + dr * tm * S if tm > 0 else dr * 1e9
            fe = min(ej + T, FE[k]) if T > 0 else FE[k]
            if fe <= ej + 1: fe = ej + 1
            if (dr > 0 and l[ej] <= stp) or (dr < 0 and h[ej] >= stp):
                xp = stp - dr; kind = 1; amb = 0; xj = ej; mae = (ep - l[ej]) if dr > 0 else (h[ej] - ep)
            else:
                xp, kind, amb, xj, mae = manage2(o, h, l, c, ej, fe, dr, ep, stp, tg)
            mae = max(mae, dr * (ep - xp))
            out[k, ns, 0] = dr; out[k, ns, 1] = ep; out[k, ns, 2] = stp; out[k, ns, 3] = xp; out[k, ns, 4] = kind
            out[k, ns, 5] = amb; out[k, ns, 6] = S; out[k, ns, 7] = mae; out[k, ns, 8] = ej; out[k, ns, 9] = xj
            ns += 1
            lo_ = l[ej:xj + 1].min(); hi_ = h[ej:xj + 1].max()
            for m in range(3):
                if ph[m] == 0 and lo_ <= LV[k, m] and LV[k, m] <= hi_: used[m] = True
            j = xj + 1
    return out


@njit(cache=True)
def run_random(o, h, l, c, RB, FE, DR, STP, tm, T):
    n = len(RB); out = np.full((n, 10), np.nan)
    for k in range(n):
        ej = RB[k]
        if ej < 0: continue
        dr = DR[k]; S = STP[k]; ep = o[ej] + dr; stp = ep - dr * S; tg = ep + dr * tm * S if tm > 0 else dr * 1e9
        fe = min(ej + T, FE[k]) if T > 0 else FE[k]
        if fe <= ej + 1: fe = ej + 1
        if (dr > 0 and l[ej] <= stp) or (dr < 0 and h[ej] >= stp):
            xp = stp - dr; kind = 1; amb = 0; xj = ej; mae = (ep - l[ej]) if dr > 0 else (h[ej] - ep)
        else:
            xp, kind, amb, xj, mae = manage2(o, h, l, c, ej, fe, dr, ep, stp, tg)
        mae = max(mae, dr * (ep - xp))
        out[k, 0] = dr; out[k, 1] = ep; out[k, 2] = stp; out[k, 3] = xp; out[k, 4] = kind
        out[k, 5] = amb; out[k, 6] = S; out[k, 7] = mae; out[k, 8] = ej; out[k, 9] = xj
    return out


COLS = ["dir", "ep", "stop", "xp", "kind", "amb", "risk", "mae", "ej", "xj"]


def to_trades(out, I):
    k = out.shape[0]
    if out.ndim == 3:
        rep = np.repeat(np.arange(k), out.shape[1]); out = out.reshape(-1, 10)
    else: rep = np.arange(k)
    t = pd.DataFrame(out, columns=COLS); t["di"] = rep; t["tdate"] = I.tdate.values[rep]
    t = t.dropna(subset=["dir"]).copy()
    t["cost"] = 0.5 + 2e-5 * (t.ep + t.xp)
    t["net"] = t.dir * (t.xp - t.ep) - t.cost; t["R"] = t.net / t.risk
    return t.sort_values(["di", "ej"]).reset_index(drop=True)


# ---------------------------------------------------------------- metrics
def sharpe(x):
    s = x.std(); return x.mean() / s * math.sqrt(252) if s > 0 else np.nan


def dd_stats(t, alld, lo, hi):
    """closed-trade 日淨值 MDD、盤中最壞 MDD、最長水下天數（日曆日，峰值日→創高日/期末）。"""
    t = t[(t.tdate >= lo) & (t.tdate < hi)].sort_values(["di", "ej"]); ad = alld[(alld >= lo) & (alld < hi)]
    if not len(t): return dict(MDD_close=np.nan, MDD_intraday=np.nan, longest_dd_days=np.nan)
    dp = t.groupby("tdate").net.sum().reindex(ad, fill_value=0) * PV; eq = dp.cumsum(); pk = eq.cummax().clip(lower=0)
    # 最長水下
    under = eq < pk; best = 0; start = None; prevd = eq.index[0]
    for dt, u in zip(eq.index, under.values):
        if u and start is None: start = prevd
        if (not u) and start is not None: best = max(best, (dt - start).days); start = None
        prevd = dt
    if start is not None: best = max(best, (eq.index[-1] - start).days)
    # 盤中最壞：逐筆
    net = t.net.values * PV; cum_before = np.concatenate([[0], np.cumsum(net)[:-1]])
    entry_cost = (0.25 + 2e-5 * t.ep.values) * PV
    worst = cum_before - t.mae.values * PV - entry_cost
    peak_before = np.maximum.accumulate(np.concatenate([[0], np.cumsum(net)]))[:-1]
    mdd_i = min((worst - peak_before).min(), (eq - pk).min())
    return dict(MDD_close=(eq - pk).min(), MDD_intraday=mdd_i, longest_dd_days=best)


def metrics(t, alld, lo, hi, full=False):
    ad = alld[(alld >= lo) & (alld < hi)]; t = t[(t.tdate >= lo) & (t.tdate < hi)]
    n = len(t)
    if n == 0: return dict(trades=0)
    dp = t.groupby("tdate").net.sum().reindex(ad, fill_value=0) * PV
    w = t.net[t.net > 0].sum(); ls = -t.net[t.net < 0].sum()
    r = dict(trades=n, win=(t.net > 0).mean(), avgR=t.R.mean(), PF=w / ls if ls > 0 else np.nan, net_pts=t.net.sum(), net_twd=t.net.sum() * PV,
             ann_ret=dp.sum() / (len(ad) / 252) / CAP, Sharpe=sharpe(dp.values), avg_pts=t.net.mean())
    if full:
        top = t.net.nlargest(10).sum()
        r.update(dd_stats(t, alld, lo, hi)); r["top10_pts"] = top; r["top10_share"] = top / t.net.sum() if t.net.sum() > 0 else np.nan
        r["net_ex_top10_twd"] = (t.net.sum() - top) * PV
    return r


# ---------------------------------------------------------------- main
def cfg_run(arr, I, p, LVm):
    va, ls, st, sm, tm, T = p
    active = np.zeros(3, np.bool_)
    for m in LVSETS[ls]: active[m] = True
    LV = np.column_stack([I[f"vah{int(va*100)}"].values, I[f"val{int(va*100)}"].values, I["poc"].values]).astype(float) if LVm is None else LVm
    o, h, l, c = arr
    return run(o, h, l, c, I.s.values, I.ce.values, I.fe.values, LV, (I.D.values * sm).clip(min=5.0), active, STYLES.index(st), tm, T)


def label(p): return f"VA{int(p[0]*100)}%|{p[1]}|{p[2]}|stop{p[3]}D|tgt{'none' if p[4]==0 else str(p[4])+'R'}|time{'close' if p[5]==0 else str(p[5])+'m'}"


def main():
    OUT.mkdir(exist_ok=True)
    d = load_bars()
    vol = pd.read_parquet(DATA, columns=["date", "volume"]); vol["ts"] = vol.date.dt.tz_localize(None)
    d = d.merge(vol[["ts", "volume"]].drop_duplicates("ts"), on="ts", how="left"); d["volume"] = d.volume.fillna(1).astype(float)
    d = d.sort_values("ts").reset_index(drop=True)
    I = build(d); arr = tuple(d[k].values for k in ("open", "high", "low", "close"))
    alld = pd.DatetimeIndex(I.tdate); nd = len(alld); isd = alld < IS_END
    print("instances", nd, alld[0].date(), alld[-1].date(), "| IS days", isd.sum(), "OOS days", (~isd).sum())
    I["va70w"] = I.vah70 - I.val70
    print("profile sanity: VA70 width median %.0f pts (night range median %.0f); POC inside VA70 %.3f; D20 median %.0f" % (
        I.va70w.median(), I.nrange.median(), ((I.poc <= I.vah70) & (I.poc >= I.val70)).mean(), I.D.median()))
    print("night close inside VA70: %.2f, above VAH: %.2f, below VAL: %.2f" % (
        ((I.nclose <= I.vah70) & (I.nclose >= I.val70)).mean(), (I.nclose > I.vah70).mean(), (I.nclose < I.val70).mean()))
    params = list(itertools.product(VAS, LVSETS, STYLES, STOPS, TGTS, TIMES))
    rows = []; DP = np.zeros((len(params), nd), np.float32); CN = np.zeros((len(params), nd), np.int16); T = {}
    for i, p in enumerate(params):
        t = to_trades(cfg_run(arr, I, p, None), I); T[i] = t
        DP[i] = np.bincount(t.di.values, weights=t.net.values * PV, minlength=nd); CN[i] = np.bincount(t.di.values, minlength=nd)
        r = dict(id=i, va=p[0], lvl=p[1], style=p[2], stop=p[3], tgt=p[4], time=p[5])
        for nm, mk in (("IS", isd), ("OOS", ~isd)):
            tt = t[(t.tdate < IS_END) if nm == "IS" else (t.tdate >= IS_END)]; dv = DP[i][mk]; eq = dv.cumsum()
            w = tt.net[tt.net > 0].sum(); ls = -tt.net[tt.net < 0].sum()
            r.update({f"{nm}_trades": len(tt), f"{nm}_win": (tt.net > 0).mean() if len(tt) else np.nan, f"{nm}_avgR": tt.R.mean() if len(tt) else np.nan,
                      f"{nm}_PF": w / ls if ls > 0 else np.nan, f"{nm}_net_twd": tt.net.sum() * PV, f"{nm}_ann": dv.sum() / (len(dv) / 252) / CAP,
                      f"{nm}_Sharpe": sharpe(dv.astype(float)), f"{nm}_MDD": (eq - np.maximum.accumulate(eq)).min()})
        rows.append(r)
    G = pd.DataFrame(rows); G.to_csv(OUT / "ovp_grid.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    ok = G[G.IS_trades >= MIN_IS].copy(); ok["OOS_rank"] = ok.OOS_Sharpe.rank(ascending=False); top = ok.sort_values("IS_Sharpe", ascending=False)
    cols = ["va", "lvl", "style", "stop", "tgt", "time", "IS_trades", "IS_win", "IS_avgR", "IS_PF", "IS_ann", "IS_Sharpe", "OOS_trades", "OOS_win", "OOS_avgR", "OOS_PF", "OOS_ann", "OOS_Sharpe", "OOS_MDD", "OOS_rank"]
    print(f"\nconfigs {len(G)}, IS trades>={MIN_IS}: {len(ok)}"); print("== IS top10 =="); print(top.head(10)[cols].round(3).to_string(index=False))
    rho = ok.IS_Sharpe.rank().corr(ok.OOS_Sharpe.rank())
    print("\nSpearman IS/OOS %.3f | IS>0 %.3f OOS>0 %.3f both>0 %.3f | mean IS %.3f mean OOS %.3f" % (rho, (ok.IS_Sharpe > 0).mean(), (ok.OOS_Sharpe > 0).mean(), ((ok.IS_Sharpe > 0) & (ok.OOS_Sharpe > 0)).mean(), ok.IS_Sharpe.mean(), ok.OOS_Sharpe.mean()))
    for k in ("style", "lvl", "va", "stop", "tgt", "time"):
        g = ok.groupby(k).agg(N=("id", "size"), IS_Sh=("IS_Sharpe", "mean"), OOS_Sh=("OOS_Sharpe", "mean"), IS_avgR=("IS_avgR", "mean"), OOS_avgR=("OOS_avgR", "mean"),
                              OOSpos=("OOS_Sharpe", lambda x: (x > 0).mean()), sp=("IS_Sharpe", lambda x: np.nan))
        g["sp"] = [ok[ok[k] == v].IS_Sharpe.rank().corr(ok[ok[k] == v].OOS_Sharpe.rank()) for v in g.index]
        print(g.round(3).to_string()); print()
    b = top.iloc[0]; bi = int(b.id); bp = params[bi]; bt = T[bi]
    print("BEST (IS Sharpe)", label(bp))
    per = {"IS": (alld[0], IS_END), "OOS": (IS_END, alld[-1] + pd.Timedelta(days=1)), "FULL": (alld[0], alld[-1] + pd.Timedelta(days=1))}
    res = pd.DataFrame({k: metrics(bt, alld, lo, hi, full=True) for k, (lo, hi) in per.items()}).T
    print(res.round(3).to_string()); res.to_csv(OUT / "ovp_summary.csv")
    bt["year"] = bt.tdate.dt.year
    yr = bt.groupby("year").agg(trades=("net", "size"), win=("net", lambda x: (x > 0).mean()), avgR=("R", "mean"), net_pts=("net", "sum")); yr["net_twd"] = yr.net_pts * PV
    print("\nper year (best):"); print(yr.round(3).to_string()); yr.to_csv(OUT / "ovp_best_yearly.csv")
    print("ambiguous stop+tgt bars:", int(bt.amb.sum()), "of", len(bt), "| exit kinds:", bt.kind.value_counts().to_dict(), "| long/short:", bt.dir.value_counts().to_dict())
    # 單參數鄰居（改一個維度，其餘取 IS 最佳）
    print("\n== neighbours of best (IS/OOS Sharpe) ==")
    dims = dict(va=VAS, lvl=tuple(LVSETS), style=STYLES, stop=STOPS, tgt=TGTS, time=TIMES); nb = []
    for k, vals in dims.items():
        for v in vals:
            q = G[(G.id != -1) & ((G[[x for x in dims if x != k]] == b[[x for x in dims if x != k]].values).all(axis=1)) & (G[k] == v)].iloc[0]
            nb.append((k, v, int(q.IS_trades), round(q.IS_Sharpe, 3), round(q.OOS_Sharpe, 3), round(q.OOS_ann, 4)))
    print(pd.DataFrame(nb, columns=["dim", "value", "IS_trades", "IS_Sharpe", "OOS_Sharpe", "OOS_ann"]).to_string(index=False))
    # 隨機進場基準（同天數 / 方向隨機 / 同停損停利 / 同時間出場）
    rs = np.random.default_rng(7); o, h, l, c = arr; STPb = (I.D.values * bp[3]).clip(min=5.0); nb_ = len(bt); stats = []; eqr = []
    for rep in range(300):
        pick = rs.choice(nd, size=min(nb_, nd), replace=False)
        RB = np.full(nd, -1); RB[pick] = [rs.integers(I.s.values[k] + 1, I.ce.values[k]) for k in pick]
        DR = rs.choice([-1.0, 1.0], size=nd)
        tr = to_trades(run_random(o, h, l, c, RB, I.fe.values, DR, STPb, bp[4], bp[5]), I)
        stats.append({pn: metrics(tr, alld, lo, hi) for pn, (lo, hi) in per.items()})
        eqr.append(np.bincount(tr.di.values, weights=tr.net.values * PV, minlength=nd).cumsum())
    eqr = np.array(eqr); rbase = []
    print("\n== random-entry baseline (300 reps) ==")
    for pn, (lo, hi) in per.items():
        sh = np.array([s[pn]["Sharpe"] for s in stats]); real = res.loc[pn, "Sharpe"]; ar = np.mean([s[pn]["avgR"] for s in stats])
        rbase.append(dict(period=pn, rand_Sharpe_mean=sh.mean(), p5=np.percentile(sh, 5), p95=np.percentile(sh, 95), rand_avgR=ar, real_Sharpe=real, real_avgR=res.loc[pn, "avgR"], real_pctile=(sh < real).mean()))
        print(f"{pn}: random Sharpe mean {sh.mean():.3f} [5%,95%] [{np.percentile(sh,5):.3f},{np.percentile(sh,95):.3f}] avgR {ar:.3f} | real {real:.3f} avgR {res.loc[pn,'avgR']:.3f} pctile {(sh < real).mean():.2f}")
    pd.DataFrame(rbase).to_csv(OUT / "ovp_random_baseline.csv", index=False)
    # 安慰劑：保留各夜盤與收盤價的相對結構，但水平線換成「另一個隨機夜盤」的相對位置（資訊被打亂）
    pl = []
    rs2 = np.random.default_rng(11)
    for rep in range(200):
        perm = rs2.permutation(nd)
        off = {}
        for nm in ("vah", "val"): off[nm] = I[f"{nm}{int(bp[0]*100)}"].values - I.nclose.values
        off["poc"] = I.poc.values - I.nclose.values
        LVp = np.column_stack([I.nclose.values + off["vah"][perm], I.nclose.values + off["val"][perm], I.nclose.values + off["poc"][perm]])
        tr = to_trades(cfg_run(arr, I, bp, LVp), I); pl.append({pn: metrics(tr, alld, lo, hi) for pn, (lo, hi) in per.items()})
    print("\n== placebo: levels from a shuffled other night (shifted to this night's close), 200 reps ==")
    plrows = []
    for pn in per:
        sh = np.array([s[pn].get("Sharpe", np.nan) for s in pl]); real = res.loc[pn, "Sharpe"]
        plrows.append(dict(period=pn, placebo_Sharpe_mean=np.nanmean(sh), p5=np.nanpercentile(sh, 5), p95=np.nanpercentile(sh, 95), real_Sharpe=real, real_pctile=(sh < real).mean()))
        print(f"{pn}: placebo Sharpe mean {np.nanmean(sh):.3f} [5%,95%] [{np.nanpercentile(sh,5):.3f},{np.nanpercentile(sh,95):.3f}] real {real:.3f} pctile {(sh < real).mean():.2f}")
    pd.DataFrame(plrows).to_csv(OUT / "ovp_placebo.csv", index=False)
    # walk-forward：每年用前 3 年重選（訓練期至少 90 筆）
    picks = []; wf = []
    for Y in range(2020, alld.year.max() + 1):
        trm = (alld.year >= Y - 3) & (alld.year < Y); tsm = alld.year == Y
        X = DP[:, trm].astype(float); sh = X.mean(1) / np.where(X.std(1) > 0, X.std(1), np.nan) * math.sqrt(252)
        sh[CN[:, trm].sum(1) < 90] = np.nan
        bb = int(np.nanargmax(sh)); y = DP[bb, tsm].astype(float); wf.append(pd.Series(y, index=alld[tsm]))
        picks.append(dict(year=Y, cfg=label(params[bb]), train_Sharpe=round(sh[bb], 3), test_Sharpe=round(sharpe(y), 3), test_twd=round(y.sum()), trades=int(CN[bb, tsm].sum())))
    wfr = pd.concat(wf); P = pd.DataFrame(picks); print("\n== walk-forward (trailing 3y, reoptimise each year) =="); print(P.to_string(index=False)); P.to_csv(OUT / "ovp_wf_picks.csv", index=False)
    w22 = wfr[wfr.index >= IS_END]
    wf_stats = {}
    for nm, s in (("WF 2020+", wfr), ("WF 2022+ (OOS span)", w22)):
        eq = s.cumsum(); wf_stats[nm] = dict(net_twd=s.sum(), ann=s.sum() / (len(s) / 252) / CAP, Sharpe=sharpe(s.values), MDD_close=(eq - eq.cummax().clip(lower=0)).min())
        print(nm, {k: round(v, 3) for k, v in wf_stats[nm].items()})
    pd.DataFrame(wf_stats).T.to_csv(OUT / "ovp_wf_summary.csv")
    bt.drop(columns=["year"]).to_csv(OUT / "ovp_best_trades.csv", index=False)
    chart(ok, bt, alld, wfr, eqr, bp, res, yr)


def chart(ok, bt, alld, wfr, eqr, bp, res, yr):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK TC"; plt.rcParams["axes.unicode_minus"] = False
    gold = "#C9A227"; bg = "#111111"; gr = "#8a8a8a"; wh = "#e6e6e6"
    def sty(ax):
        ax.set_facecolor(bg); ax.tick_params(colors=wh); ax.grid(alpha=.15, color=gr)
        for s in ax.spines.values(): s.set_color(gr)
        ax.xaxis.label.set_color(wh); ax.yaxis.label.set_color(wh); ax.title.set_color(wh)
    dpv = bt.groupby("tdate").net.sum().reindex(alld, fill_value=0) * PV; eq = dpv.cumsum() / 1e4
    fig, ax = plt.subplots(figsize=(12, 6.2), facecolor=bg); sty(ax)
    q5, q50, q95 = np.percentile(eqr, [5, 50, 95], axis=0) / 1e4
    ax.fill_between(alld, q5, q95, color=gr, alpha=.35, label="隨機進場 5–95%（300 次）"); ax.plot(alld, q50, color=gr, lw=1, label="隨機進場 中位數")
    ax.plot(alld, eq, color=gold, lw=2, label="IS 最佳參數（2017-21 選定）")
    ax.plot(wfr.index, wfr.cumsum().values / 1e4 + 0, color="#5fb0ff", lw=1.5, label="walk-forward（前 3 年重選，2020 起）")
    ax.axvline(IS_END, color="white", ls="--", lw=1); ax.text(IS_END, ax.get_ylim()[1], " OOS →", color="white", va="top")
    ax.set_ylabel("累積淨損益（萬元，1 口 TXF）"); ax.legend(facecolor=bg, labelcolor=wh, edgecolor=gr, loc="upper left", fontsize=9)
    o_, i_ = res.loc["OOS"], res.loc["IS"]
    ax.set_title(f"夜盤 Volume Profile 首觸｜{label(bp)}\nIS Sharpe {i_.Sharpe:.2f} → OOS Sharpe {o_.Sharpe:.2f}（OOS {int(o_.trades)} 筆，勝率 {o_.win*100:.0f}%，淨 {o_.net_twd/1e4:.0f} 萬）", fontsize=11, color=gold)
    fig.text(0.98, 0.01, "夜盤貓頭鷹｜含成本與滑價｜回測不代表未來績效", color=gold, ha="right", fontsize=9); fig.tight_layout(rect=[0, .03, 1, 1])
    fig.savefig(OUT / "ovp_equity.png", dpi=110, facecolor=bg); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor=bg); cl = dict(zip(STYLES, (gold, "#ff9f43", "#5fb0ff", "#7bd88f")))
    for a in ax: sty(a)
    for s, g in ok.groupby("style"): ax[0].scatter(g.IS_Sharpe, g.OOS_Sharpe, s=6, color=cl[s], label=s, alpha=.7)
    ax[0].axhline(0, c=gr); ax[0].axvline(0, c=gr); ax[0].set_xlabel("IS Sharpe 2017-21"); ax[0].set_ylabel("OOS Sharpe 2022+")
    ax[0].set_title(f"各參數 IS vs OOS（Spearman {ok.IS_Sharpe.rank().corr(ok.OOS_Sharpe.rank()):.2f}，N={len(ok)}）"); ax[0].legend(facecolor=bg, labelcolor=wh, edgecolor=gr)
    ax[1].bar(yr.index.astype(str), yr.net_twd / 1e4, color=[gold if v > 0 else "#d9534f" for v in yr.net_twd]); ax[1].axhline(0, c=gr)
    ax[1].set_title("IS 最佳參數 各年度淨損益（萬元）"); ax[1].set_ylabel("萬元")
    fig.tight_layout(); fig.savefig(OUT / "ovp_is_vs_oos.png", dpi=110, facecolor=bg); plt.close(fig)
    print("charts results/ovp_equity.png results/ovp_is_vs_oos.png")


if __name__ == "__main__":
    main()
