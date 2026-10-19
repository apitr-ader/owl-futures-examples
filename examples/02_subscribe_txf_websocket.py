"""02 用 WebSocket 訂閱台指期近月（TXF1!）成交與五檔（對應 Threads P5、P6）

執行：python examples/02_subscribe_txf_websocket.py
夜盤：在 .env 設 FUBON_AFTER_HOURS=1（訂閱參數加 "afterHours": True）

重點（官方文件）：
- TXF1! / TXF2! / TXF3! 是連續月別名（v2.2.9 起）。
- WebSocket 的別名只在「送出訂閱的當下」解析一次，跨結算日不會自動轉倉；
  長時間掛著的程式，換月後要 unsubscribe 再 subscribe。
- 頻道：trades（成交）、books（最佳五檔）、aggregates（聚合）。
- 速率限制：WebSocket 300 訂閱數、7 連線數。

官方文件：
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/websocket-api/getting-started.txt
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/websocket-api/market-data-channels/trades.txt
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/websocket-api/market-data-channels/books.txt
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/rate-limit.txt
"""

import json
import os
import time

from _common import env_flag, import_sdk, login

FubonSDK = import_sdk()
from fubon_neo.sdk import Mode  # noqa: E402  官方：Mode.Speed（預設）/ Mode.Normal

SYMBOL = os.environ.get("FUBON_WS_SYMBOL", "TXF1!")
AFTER_HOURS = env_flag("FUBON_AFTER_HOURS")


def handle_message(message):
    # 官方 Python 範例直接 print(message)；這裡嘗試當 JSON 解析，失敗就原樣印出。
    try:
        data = json.loads(message) if isinstance(message, (str, bytes)) else message
    except ValueError:
        print(message)
        return
    event = data.get("event") if isinstance(data, dict) else None
    if event == "heartbeat":
        return  # 伺服器每 30 秒送一次 heartbeat，不印
    print(data)


def handle_connect():
    print("market data connected")


def handle_disconnect(code, message, *reason):  # 健康檢查逾時會多帶一個參數
    print("market data disconnect: {}, {} {}".format(code, message, reason or ""))
    print("（SDK 不會自動重連，長時間執行請參考 05_reconnect_websocket.py）")


def handle_error(error):
    print("market data error: {}".format(error))


def main():
    sdk = FubonSDK()
    login(sdk)  # 需登入後才能取得行情權限

    mode = Mode.Normal if os.environ.get("FUBON_REALTIME_MODE", "").lower() == "normal" else Mode.Speed
    sdk.init_realtime(mode)  # 建立行情連線

    futopt = sdk.marketdata.websocket_client.futopt
    futopt.on("message", handle_message)
    futopt.on("connect", handle_connect)
    futopt.on("disconnect", handle_disconnect)
    futopt.on("error", handle_error)
    futopt.connect()

    for channel in ("trades", "books"):
        params = {"channel": channel, "symbol": SYMBOL}
        if AFTER_HOURS:
            params["afterHours"] = True  # 訂閱夜盤行情
        futopt.subscribe(params)
        print("已送出訂閱：{}".format(params))

    print("Ctrl+C 結束")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        futopt.disconnect()
        sdk.logout()
        print("已斷線並登出")


if __name__ == "__main__":
    main()
