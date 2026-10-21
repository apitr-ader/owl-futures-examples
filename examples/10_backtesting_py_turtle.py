"""10 用 backtesting.py 回測台指期 Turtle（海龜）策略（回測系列 3）

資料來源：09（日 K，2010 起）或 07（1 分 K，可合成日 K／60 分 K）下載的 Parquet／CSV。

執行：
  pip install backtesting
  python examples/10_backtesting_py_turtle.py data/TXF_1c_D_regular_2010-01-01_2026-10-21.parquet
  python examples/10_backtesting_py_turtle.py data/TXF_1c_1m_both_....parquet --resample D
  python examples/10_backtesting_py_turtle.py data/TXF_1c_1m_both_....parquet --resample 60min --roll-adjust back

策略（簡化版 Turtle System 1，多空都做，固定 1 口）：
- 收盤突破前 20 根最高 → 下一根開盤買進；跌破前 20 根最低 → 放空。
- 多單：收盤跌破前 10 根最低，或收盤低於「進場價 − 2N」→ 下一根開盤出場（空單對稱）。
- N = 20 根 Wilder ATR。不加碼（原版會加到 4 單位）、不做 System 1 的「上次獲利就跳過」過濾。

成本（每口、單邊）：手續費 --commission 元 ＋ 期交稅（價格 × 200 × 十萬分之二）＋ 滑價 --slippage 點 × 200 元。

backtesting.py 沒有「契約乘數」概念，這裡把價格 × 200 換成「元」，1 單位＝1 口，
損益、權益曲線就直接是新台幣；保證金用 margin 比例（--margin-ratio，原始保證金 ÷ 契約價值）。
滑價用 commission 函式一併扣（不改成交價），所以交易明細裡的成交價是「開盤價」，成本在 Commission 欄。

僅供教學，不構成投資建議；歷史回測不代表未來績效。
"""

import argparse
import sys
from pathlib import Path

from _bt_data import (DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, POINT_VALUE, add_turtle_indicators,
                      cost_per_side, load_ohlcv, roll_dates)

try:
    from backtesting import Backtest, Strategy
except ImportError:
    sys.exit("請先安裝：pip install backtesting")


def parse_args():
    p = argparse.ArgumentParser(description="backtesting.py 台指期 Turtle 回測")
    p.add_argument("path", help="07 或 09 輸出的 .parquet／.csv")
    p.add_argument("--resample", default=None, help="分 K 檔用：D 或 60min 等；日 K 檔不用帶")
    p.add_argument("--session", default="both", choices=("both", "regular", "afterhours"))
    p.add_argument("--roll-adjust", default="none", choices=("none", "back"))
    p.add_argument("--entry", type=int, default=20)
    p.add_argument("--exit", type=int, default=10)
    p.add_argument("--commission", type=float, default=DEFAULT_COMMISSION, help="每口單邊手續費（元）")
    p.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE, help="每口單邊滑價（點）")
    p.add_argument("--cash", type=float, default=2_000_000, help="初始資金（元）")
    p.add_argument("--margin-ratio", type=float, default=0.1,
                   help="原始保證金 ÷ 契約價值（預設 0.1；實際以期交所公告為準）")
    p.add_argument("--plot", default=None, help="HTML 圖檔路徑（預設 data/turtle_backtesting_py.html）")
    p.add_argument("--trades-csv", default=None, help="另存交易明細 CSV（方便和 11 比對）")
    return p.parse_args()


class Turtle(Strategy):
    def init(self):
        d = self.data
        # 指標已在 pandas 算好，self.I 只是讓它畫在圖上
        self.I(lambda: d.hi_entry, name="hi_entry", overlay=True)
        self.I(lambda: d.lo_entry, name="lo_entry", overlay=True)
        self.I(lambda: d.hi_exit, name="hi_exit", overlay=True)
        self.I(lambda: d.lo_exit, name="lo_exit", overlay=True)
        self.stop = None

    def next(self):
        d = self.data
        close = d.Close[-1]
        if not self.position:
            if close > d.hi_entry[-1]:
                self.buy(size=1)  # 下一根開盤成交；停損等成交後用實際進場價設定
            elif close < d.lo_entry[-1]:
                self.sell(size=1)
            return
        t = self.trades[-1]
        if self.stop is None or getattr(self, "_entry_bar", None) != t.entry_bar:
            # 用實際成交價重算 2N 停損（N 取進場前一根，與 11 一致）
            n0 = self._n_at(t.entry_bar - 1)
            self.stop = t.entry_price - 2 * n0 if t.is_long else t.entry_price + 2 * n0
            self._entry_bar = t.entry_bar
        if self.position.is_long and (close < d.lo_exit[-1] or close < self.stop):
            self.position.close()
        elif self.position.is_short and (close > d.hi_exit[-1] or close > self.stop):
            self.position.close()

    def _n_at(self, i):
        return self.data.df["N"].iloc[i]


def main():
    args = parse_args()
    raw = load_ohlcv(args.path, session=args.session, resample=args.resample, roll_adjust=args.roll_adjust)
    rolls = roll_dates(raw)
    df = add_turtle_indicators(raw, args.entry, args.exit)
    df = df.drop(columns=["contractMonth"], errors="ignore")
    df.index = df.index.tz_localize(None)  # 已是台北時間，去掉 tz 讓 bokeh 好畫
    # 價格 × 200 → 單位變成「元」，1 單位 = 1 口
    for c in ("Open", "High", "Low", "Close", "N", "hi_entry", "lo_entry", "hi_exit", "lo_exit"):
        df[c] = df[c] * POINT_VALUE

    def commission(size, price):  # price 已是「元」
        return cost_per_side(price / POINT_VALUE, size, args.commission, args.slippage)

    bt = Backtest(df, Turtle, cash=args.cash, commission=commission,
                  margin=args.margin_ratio, trade_on_close=False, exclusive_orders=False,
                  finalize_trades=True)
    stats = bt.run()

    print("資料：{} ~ {}，{} 根，換月 {} 次（roll_adjust={}）".format(
        df.index[0], df.index[-1], len(df), len(rolls), args.roll_adjust))
    print(stats.drop(["_strategy", "_equity_curve", "_trades"]).to_string())
    trades = stats["_trades"]
    print("\n== 摘要（與 11 對照） ==")
    print("交易次數：{}".format(len(trades)))
    if len(trades):
        print("勝率：{:.1f}%".format((trades["PnL"] > 0).mean() * 100))
        print("淨損益（含成本）：{:,.0f} 元，約 {:,.0f} 點".format(trades["PnL"].sum(),
                                                       trades["PnL"].sum() / POINT_VALUE))
    print("最終權益：{:,.0f} 元；最大回落：{:.2f}%".format(stats["Equity Final [$]"], stats["Max. Drawdown [%]"]))

    if args.trades_csv:
        trades.to_csv(args.trades_csv, index=False)
    plot = args.plot or str(Path(__file__).resolve().parent.parent / "data" / "turtle_backtesting_py.html")
    Path(plot).parent.mkdir(exist_ok=True)
    bt.plot(filename=plot, open_browser=False, resample=False)
    print("圖已存到 {}".format(plot))


if __name__ == "__main__":
    main()
