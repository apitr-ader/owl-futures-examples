"""11 用 Backtrader 回測同一個台指期 Turtle 策略（回測系列 3，和 10 對照）

執行：
  pip install backtrader
  python examples/11_backtrader_turtle.py data/TXF_1c_D_regular_2010-01-01_2026-10-21.parquet
  python examples/11_backtrader_turtle.py data/TXF_1c_1m_both_....parquet --resample D

策略、成本與 10 完全相同（指標也是同一份 pandas 計算，見 _bt_data.py）。
Backtrader 原生支援期貨：CommInfo 設 mult=200（每點 200 元）、stocklike=False，
價格維持「點數」，損益自動 × 200；保證金用 automargin（契約價值 × 比例）。
滑價和 10 一樣併進手續費（_getcommission）扣，成交價不動，兩邊才好逐筆對帳。

兩個框架的成交模型差異（重要）：
- 市價單：兩者預設都是「訊號那根收盤後下單 → 下一根開盤成交」（backtesting.py
  trade_on_close=False；Backtrader 未開 cheat-on-open／coc）。本範例只用市價單，所以進出場價應一致。
- 停損：本範例的 2N 停損是「收盤確認後隔根開盤出場」，不是盤中觸價。若改成真的停損單：
  backtesting.py 的 sl= 在同一根內被觸及就以停損價（跳空則開盤價）成交，且同根同時碰到停損停利時
  偏向保守處理；Backtrader 的 Stop 單從下一根才開始檢查，跳空時以開盤價成交。兩者結果會開始分歧。
- 最後一根還沒平的部位：10 用 finalize_trades=True 在最後收盤強制結算並計入；
  這裡在最後一根手動平倉（下一根不存在，Backtrader 不會成交），所以會在摘要另列「未平倉」。
- 保證金不足：backtesting.py 直接取消訂單；Backtrader 的 broker 會 Margin 拒單（order.Margin）。
- 成本計算時點：backtesting.py 0.6 的 commission 在進場與出場各扣一次（以成交價）；
  Backtrader 同樣每次成交計一次。兩邊單邊成本公式相同。

Python 版本：backtrader 1.9.78.123 是 2023 年的最後一版，在 Python 3.13 實測可 import、可回測；
畫圖 cerebro.plot() 需 matplotlib 且新版 matplotlib 常出錯，這裡不畫圖（要圖請看 10 的 HTML）。
若你的環境 import 失敗，可改用 Python 3.10／3.11 的 venv，或 pip install backtrader2（社群維護分支）。

僅供教學，不構成投資建議；歷史回測不代表未來績效。
"""

import argparse
import sys

from _bt_data import (DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, POINT_VALUE, TAX_RATE,
                      add_turtle_indicators, load_ohlcv, roll_dates)

try:
    import backtrader as bt
except ImportError:
    sys.exit("請先安裝：pip install backtrader（Python 版本問題見檔頭說明）")


def parse_args():
    p = argparse.ArgumentParser(description="Backtrader 台指期 Turtle 回測")
    p.add_argument("path")
    p.add_argument("--resample", default=None)
    p.add_argument("--session", default="both", choices=("both", "regular", "afterhours"))
    p.add_argument("--roll-adjust", default="none", choices=("none", "back"))
    p.add_argument("--entry", type=int, default=20)
    p.add_argument("--exit", type=int, default=10)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    p.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    p.add_argument("--cash", type=float, default=2_000_000)
    p.add_argument("--margin-ratio", type=float, default=0.1)
    p.add_argument("--trades-csv", default=None)
    return p.parse_args()


class TXFComm(bt.CommInfoBase):
    """台指期：每點 200 元、保證金制；單邊成本 = 手續費 + 期交稅 + 滑價。"""
    params = (("stocklike", False), ("commtype", bt.CommInfoBase.COMM_FIXED),
              ("mult", POINT_VALUE), ("fee", DEFAULT_COMMISSION), ("slip", DEFAULT_SLIPPAGE))

    def _getcommission(self, size, price, pseudoexec):
        q = abs(size)
        return q * (self.p.fee + price * POINT_VALUE * TAX_RATE + self.p.slip * POINT_VALUE)


class TurtleData(bt.feeds.PandasData):
    lines = ("N", "hi_entry", "lo_entry", "hi_exit", "lo_exit")
    params = tuple((x, -1) for x in lines) + (("openinterest", None),)


class Turtle(bt.Strategy):
    def __init__(self):
        self.order = None
        self.stop_px = None
        self.trades = []
        self.equity = []

    def notify_order(self, order):
        if order.status in (order.Margin, order.Rejected, order.Canceled):
            print("訂單未成交：", order.getstatusname())
        if order.status == order.Completed and self.position and self.stop_px is None:
            n0 = self.data.N[-1]  # 訊號那根的 N（與 10 相同）
            px = order.executed.price
            self.stop_px = px - 2 * n0 if self.position.size > 0 else px + 2 * n0
        if not order.alive():
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                "EntryTime": bt.num2date(trade.dtopen), "ExitTime": bt.num2date(trade.dtclose),
                "Size": 1 if trade.long else -1, "EntryPrice": trade.price,
                "PnL": trade.pnlcomm, "Commission": trade.commission,
            })

    def next(self):
        self.equity.append(self.broker.getvalue())
        if self.order:
            return
        d, close = self.data, self.data.close[0]
        if len(d) == d.buflen():  # 最後一根：不再開新倉
            return
        if not self.position:
            self.stop_px = None
            if close > d.hi_entry[0]:
                self.order = self.buy(size=1)
            elif close < d.lo_entry[0]:
                self.order = self.sell(size=1)
        elif self.stop_px is not None:
            if self.position.size > 0 and (close < d.lo_exit[0] or close < self.stop_px):
                self.order = self.close()
            elif self.position.size < 0 and (close > d.hi_exit[0] or close > self.stop_px):
                self.order = self.close()


def max_drawdown_pct(equity):
    peak, mdd = float("-inf"), 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return mdd * 100


def main():
    args = parse_args()
    raw = load_ohlcv(args.path, session=args.session, resample=args.resample, roll_adjust=args.roll_adjust)
    rolls = roll_dates(raw)
    df = add_turtle_indicators(raw, args.entry, args.exit).drop(columns=["contractMonth"], errors="ignore")
    df.index = df.index.tz_localize(None)  # 台北時間，去 tz（Backtrader 只吃 naive datetime）

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(TurtleData(dataname=df))
    cerebro.addstrategy(Turtle)
    cerebro.broker.setcash(args.cash)
    cerebro.broker.addcommissioninfo(TXFComm(fee=args.commission, slip=args.slippage,
                                             automargin=args.margin_ratio * POINT_VALUE))
    strat = cerebro.run()[0]

    trades = strat.trades
    final = cerebro.broker.getvalue()
    pos = strat.position.size
    print("資料：{} ~ {}，{} 根，換月 {} 次（roll_adjust={}）".format(
        df.index[0], df.index[-1], len(df), len(rolls), args.roll_adjust))
    print("\n== 摘要（與 10 對照） ==")
    print("交易次數（已平倉）：{}{}".format(len(trades), "，另有未平倉 {} 口".format(pos) if pos else ""))
    if trades:
        pnl = sum(t["PnL"] for t in trades)
        print("勝率：{:.1f}%".format(sum(t["PnL"] > 0 for t in trades) / len(trades) * 100))
        print("淨損益（已平倉、含成本）：{:,.0f} 元，約 {:,.0f} 點".format(pnl, pnl / POINT_VALUE))
    print("最終權益（含未平倉市值）：{:,.0f} 元；最大回落：{:.2f}%".format(
        final, max_drawdown_pct(strat.equity + [final])))
    if args.trades_csv:
        import pandas as pd
        pd.DataFrame(trades).to_csv(args.trades_csv, index=False)


if __name__ == "__main__":
    main()
