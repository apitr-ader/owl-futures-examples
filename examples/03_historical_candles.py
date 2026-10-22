"""03 查詢台指期歷史 K 線並存成 CSV（對應 Threads P8）

執行：
  python examples/03_historical_candles.py                     # 近月連續合約、近 30 天日 K
  python examples/03_historical_candles.py --timeframe 5 --days 5 --session afterhours

重點（官方文件，SDK v2.4.0 起）：
- product 用商品代號（TXF），契約月份用 contractMonth（1! / 2! / 3! 或 yyyyMM）。
  注意：契約代碼（例 TXFK6）不可當 product。
- 日／週／月 K 可回溯至 2010 年；分 K（1/5/10/15/30/60）自 2017-06 起。
- session：regular 日盤（預設）／afterhours 夜盤。
- 單次查詢區間須「小於 1 年」。
- 連續合約不做換月價差調整，換月那根會跳空，回測前要自己處理；
  每根 K 線的 data.contractMonth 標示實際所屬合約。
- 歷史行情速率限制 60 次／分，超過回 429。

官方文件：
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/http-api/historical/candles.txt
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/http-api/getting-started.txt
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/rate-limit.txt
"""

import argparse
import csv
import datetime as dt
import sys

from _common import REPO_ROOT, import_sdk, login

FubonSDK = import_sdk()
# 官方：2.2.4 及以後版本，行情 Web API 例外改用 Exception 回報
from fubon_neo.fugle_marketdata.rest.base_rest import FugleAPIError  # noqa: E402

TIMEFRAMES = ("1", "5", "10", "15", "30", "60", "D", "W", "M")
FIELDS = "open,high,low,close,volume"


def parse_args():
    p = argparse.ArgumentParser(description="查詢期貨歷史 K 線")
    p.add_argument("--product", default="TXF", help="商品代號，例如 TXF、MXF（預設 TXF）")
    p.add_argument("--contract-month", default="1!", help="1! / 2! / 3! 或 yyyyMM（預設 1!）")
    p.add_argument("--timeframe", default="D", choices=TIMEFRAMES)
    p.add_argument("--session", default="regular", choices=("regular", "afterhours"))
    p.add_argument("--days", type=int, default=30, help="往回查幾天（需小於 365）")
    p.add_argument("--out", default=None, help="輸出 CSV 路徑（預設 data/ 底下）")
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 < args.days < 365:
        sys.exit("--days 需介於 1 與 364 之間（官方：單次查詢區間須小於 1 年）")

    to_date = dt.date.today()
    from_date = to_date - dt.timedelta(days=args.days)

    sdk = FubonSDK()
    login(sdk)  # 需登入後才能取得行情權限
    sdk.init_realtime()  # 建立行情連線
    restfutopt = sdk.marketdata.rest_client.futopt

    query = {
        "product": args.product,
        "contractMonth": args.contract_month,
        "timeframe": args.timeframe,
        "session": args.session,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "fields": FIELDS,
        "sort": "asc",
    }
    try:
        result = restfutopt.historical.candles(**query)
    except FugleAPIError as e:
        # 404 = 區間內查無資料；429 = 超過速率限制（60 次／分），等 1 分鐘再試
        sys.exit("查詢失敗：status={} body={}".format(e.status_code, e.response_text))
    finally:
        sdk.logout()

    # 已實測（SDK 2.4.0）：Python SDK 回傳 dict（product、contractMonth、exchange、session、timeframe、sort、data[]）
    rows = result.get("data", []) if isinstance(result, dict) else []
    if not rows:
        print("沒有資料，原始回傳：{}".format(result))
        return

    out = args.out or str(REPO_ROOT / "data" / "{}_{}_{}_{}_{}_{}.csv".format(
        args.product, args.contract_month.replace("!", "c"), args.timeframe,
        args.session, from_date, to_date))
    (REPO_ROOT / "data").mkdir(exist_ok=True)
    columns = ["date", "contractMonth"] + FIELDS.split(",")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # 標出換月（contractMonth 改變）的那幾根，提醒回測要處理跳空
    rolls = [r["date"] for prev, r in zip(rows, rows[1:])
             if r.get("contractMonth") != prev.get("contractMonth")]
    print("共 {} 根 K 線，已存到 {}".format(len(rows), out))
    if rolls:
        print("注意：以下日期發生換月（連續合約未做價差調整，會有跳空）：{}".format(", ".join(rolls)))


if __name__ == "__main__":
    main()
