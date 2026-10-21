"""08 用 07 存下來的台指期 1 分 K（CSV 或 Parquet），跑一個「開盤區間突破」當沖回測（對應回測系列第 2 篇）

執行：
  python examples/08_backtest_orb.py data/TXF_1c_1m_both_2017-06-01_2026-10-20.csv
  python examples/08_backtest_orb.py data/TXF_1c_1m_both_2017-06-01_2026-10-20.parquet
  python examples/08_backtest_orb.py <檔案> --range-min 30 --slippage 2 --commission 50

讀 CSV 只用標準函式庫，不需要 pandas；讀 .parquet 才會載入 pandas／pyarrow。
不需要登入，純讀檔。兩種格式同一份資料，回測結果完全相同。

策略（故意寫得很簡單，重點是「成本」跟「別偷看未來」）：
- 只看日盤（session = regular）。每天開盤前 N 分鐘（預設 15）的最高／最低 = 區間。
- 之後價格突破區間上緣做多、跌破下緣做空，一天最多一筆。
- 停損放在區間另一側；沒停損就在當天日盤最後一根收盤平倉，不留倉。
  → 當沖不跨日，也就不會吃到「連續月 1! 換月跳空」這個坑。

成本（每邊都算，進場一次、出場一次）：
- 期貨交易稅：成交金額 × 十萬分之二（股價類期貨）。大台每點 200 元。
- 手續費：每口每邊 --commission 元。各家券商不同，預設 0，請換成你自己的費率。
- 滑價：每邊 --slippage 點（預設 1 點）。突破單是停損單性質，實際成交通常比觸發價差。

刻意保守的假設：
- 突破那根如果開盤就跳過區間，用開盤價成交（不是用區間價）。
- 同一根 K 同時碰到進場價和停損價，一律當作停損（分 K 看不出先後）。

# TODO: 未驗證，請對照官方文件
# 假設分 K 的 date 是「該分鐘開始時間」（例如 08:45 那根涵蓋 08:45–08:46）。
# 若官方是以結束時間標示，--range-min 的區間要少算一根，請先拿一天資料對照看盤軟體。

※ 純技術範例，非投資建議。回測結果不代表未來績效，期貨有風險。
"""

import argparse
import csv
import datetime as dt
import math
import sys
from collections import OrderedDict

POINT_VALUE = 200          # 大台 TXF 每點 200 元（小台 MXF 為 50，請用 --point-value）
TAX_RATE = 0.00002         # 期貨交易稅，股價類期貨十萬分之二
DAY_OPEN = dt.time(8, 45)  # 日盤 08:45 開盤


def parse_args():
    p = argparse.ArgumentParser(description="台指期開盤區間突破當沖回測（含稅、手續費、滑價）")
    p.add_argument("path", help="07_download_minute_history.py 產生的 1 分 K 檔（.csv 或 .parquet）")
    p.add_argument("--range-min", type=int, default=15, help="開盤區間長度（分鐘）")
    p.add_argument("--slippage", type=float, default=1.0, help="每邊滑價（點）")
    p.add_argument("--commission", type=float, default=0.0, help="每口每邊手續費（元），換成你的費率")
    p.add_argument("--point-value", type=float, default=POINT_VALUE)
    p.add_argument("--start", default=None, help="yyyy-MM-dd，只回測這天之後")
    p.add_argument("--end", default=None, help="yyyy-MM-dd，只回測這天之前")
    p.add_argument("--trades-out", default=None, help="把逐筆交易存成 CSV")
    return p.parse_args()


def iter_csv(path):
    """(datetime, open, high, low, close)，純標準函式庫。"""
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("session", "regular") != "regular":
                continue
            yield (dt.datetime.fromisoformat(r["date"]), float(r["open"]), float(r["high"]),
                   float(r["low"]), float(r["close"]))


def iter_parquet(path):
    """同 iter_csv，但讀 Parquet；只有走這條才 import pandas。"""
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError:
        sys.exit("讀 .parquet 需要 pandas 與 pyarrow：pip install pandas pyarrow（或改用 CSV）")
    df = pd.read_parquet(path, engine="pyarrow")
    if "session" in df.columns:
        df = df[df["session"].astype(str) == "regular"]
    ts = df["date"]
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Asia/Taipei")  # 確保日期／時間以台北時間計
    cols = [df[c].astype("float64").tolist() for c in ("open", "high", "low", "close")]
    for t, o, h, l, c in zip(ts.dt.to_pydatetime(), *cols):
        yield t, o, h, l, c


def load_days(path, start, end):
    """回傳 {日期: [bar, ...]}，只留日盤，照時間排序。"""
    days = OrderedDict()
    rows = iter_parquet(path) if path.lower().endswith(".parquet") else iter_csv(path)
    for ts, o, h, l, c in rows:
        d = ts.date().isoformat()
        if (start and d < start) or (end and d > end):
            continue
        days.setdefault(d, []).append({"t": ts.time(), "o": o, "h": h, "l": l, "c": c})
    for bars in days.values():
        bars.sort(key=lambda b: b["t"])
    return days


def cost_per_side(price, args):
    """一口、一邊的成本（元）：稅＋手續費＋滑價。"""
    tax = round(price * args.point_value * TAX_RATE)  # 稅額四捨五入到元
    return tax + args.commission + args.slippage * args.point_value


def backtest_day(bars, args):
    cutoff = (dt.datetime.combine(dt.date.min, DAY_OPEN)
              + dt.timedelta(minutes=args.range_min)).time()
    rng = [b for b in bars if b["t"] < cutoff]
    rest = [b for b in bars if b["t"] >= cutoff]
    if len(rng) < args.range_min * 0.8 or not rest:
        return None  # 資料不完整的日子（半天、缺資料）跳過
    hi, lo = max(b["h"] for b in rng), min(b["l"] for b in rng)

    side = entry = stop = None
    for i, b in enumerate(rest):
        if side is None:
            if b["h"] > hi:
                side, entry, stop = 1, max(b["o"], hi), lo
            elif b["l"] < lo:
                side, entry, stop = -1, min(b["o"], lo), hi
            else:
                continue
            # 進場那根若也碰到停損 → 保守當作停損
            if (side == 1 and b["l"] <= stop) or (side == -1 and b["h"] >= stop):
                return side, entry, stop, "stop", b["t"]
            continue
        if side == 1 and b["l"] <= stop:
            return side, entry, min(b["o"], stop), "stop", b["t"]
        if side == -1 and b["h"] >= stop:
            return side, entry, max(b["o"], stop), "stop", b["t"]
    if side is None:
        return None  # 整天沒突破，不交易
    return side, entry, rest[-1]["c"], "close", rest[-1]["t"]


def max_drawdown(equity):
    peak, mdd = 0.0, 0.0
    for e in equity:
        peak = max(peak, e)
        mdd = max(mdd, peak - e)
    return mdd


def main():
    args = parse_args()
    days = load_days(args.path, args.start, args.end)
    if not days:
        sys.exit("檔案裡沒有日盤資料，請確認是用 07 抓的、且含 session=regular。")

    trades = []
    for d, bars in days.items():
        res = backtest_day(bars, args)
        if res is None:
            continue
        side, entry, exit_, reason, t = res
        gross = side * (exit_ - entry) * args.point_value
        cost = cost_per_side(entry, args) + cost_per_side(exit_, args)
        trades.append({"date": d, "side": "long" if side == 1 else "short",
                       "entry": entry, "exit": exit_, "reason": reason,
                       "exit_time": t.strftime("%H:%M"),
                       "gross": gross, "cost": cost, "net": gross - cost})

    if not trades:
        print("整段沒有任何交易。")
        return

    def summary(key):
        pnl = [t[key] for t in trades]
        eq, s = [], 0.0
        for x in pnl:
            s += x
            eq.append(s)
        wins = [x for x in pnl if x > 0]
        losses = [x for x in pnl if x <= 0]
        pf = sum(wins) / -sum(losses) if losses and sum(losses) < 0 else math.inf
        return s, len(wins) / len(pnl), pf, max_drawdown(eq)

    n = len(trades)
    print("期間：{} ~ {}，交易日 {} 天，交易 {} 筆".format(
        trades[0]["date"], trades[-1]["date"], len(days), n))
    print("假設：區間 {} 分、滑價每邊 {} 點、手續費每邊 {} 元、交易稅十萬分之二".format(
        args.range_min, args.slippage, args.commission))
    print("{:<10}{:>14}{:>8}{:>8}{:>14}".format("", "總損益(元)", "勝率", "獲利因子", "最大回撤(元)"))
    for label, key in (("不含成本", "gross"), ("含成本", "net")):
        s, wr, pf, mdd = summary(key)
        print("{:<10}{:>14,.0f}{:>7.1%}{:>8.2f}{:>14,.0f}".format(label, s, wr, pf, mdd))
    avg_cost = sum(t["cost"] for t in trades) / n
    print("平均每筆來回成本：{:,.0f} 元（約 {:.1f} 點）".format(avg_cost, avg_cost / args.point_value))

    by_year = OrderedDict()
    for t in trades:
        by_year.setdefault(t["date"][:4], []).append(t["net"])
    print("\n逐年（含成本）：")
    for y, pnl in by_year.items():
        print("  {}：{:>4} 筆 {:>12,.0f} 元".format(y, len(pnl), sum(pnl)))

    if args.trades_out:
        with open(args.trades_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(trades[0]))
            w.writeheader()
            w.writerows(trades)
        print("\n逐筆交易已存到", args.trades_out)


if __name__ == "__main__":
    main()
