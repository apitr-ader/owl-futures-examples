"""05 行情 WebSocket 斷線重連＋重新訂閱（對應 Threads P11、P13）

執行：python examples/05_reconnect_websocket.py

重點（官方文件「期貨行情 > WebSocket API > 開始使用 > 斷線重連」）：
- 連線中斷後 SDK 不會自動重連，也不保留原有訂閱。
- disconnect handler 只負責「通知」，重連交給背景執行緒。
- 重連前先 disconnect() 重設狀態，再 connect()。
- 重試間隔 2、4、8、16、30 秒，最多 5 次。
- 訂閱清單只存一份（每個頻道一筆、多檔用 symbols），重連後整份重新 subscribe。
- 約 60 秒沒收到伺服器任何訊息，SDK 會判定健康檢查逾時並觸發 disconnect，
  此時 handler 會多收到一個參數 {'reason': 'health-check-timeout'}。
- 連續月別名在每次 subscribe 當下解析；重連後重新訂閱 TXF1! 會解析到當下的近月。

本檔大致照官方 Python 範例改寫，只多加了：讀環境變數、夜盤開關、印訊息、Ctrl+C 結束。

官方文件：
https://www.fbs.com.tw/TradeAPI/docs/market-data-future/websocket-api/getting-started.txt
"""

import os
import threading
import time

from _common import env_flag, import_sdk, login

FubonSDK = import_sdk()

AFTER_HOURS = env_flag("FUBON_AFTER_HOURS")
MAX_RETRIES = 5
CONNECT_TIMEOUT = 10  # 秒

# 初次訂閱與重連後重新訂閱共用同一份清單；每個頻道一筆，多檔用 symbols
SYMBOLS = [s.strip() for s in os.environ.get("FUBON_WS_SYMBOLS", "TXF1!,MXF1!").split(",") if s.strip()]
SUBSCRIPTIONS = [
    {"channel": "trades", "symbols": SYMBOLS},
    {"channel": "books", "symbols": SYMBOLS},
]
if AFTER_HOURS:
    for _p in SUBSCRIPTIONS:
        _p["afterHours"] = True

reconnect_needed = threading.Event()
stop = threading.Event()


def main():
    sdk = FubonSDK()
    login(sdk)
    sdk.init_realtime()
    futopt = sdk.marketdata.websocket_client.futopt

    def handle_message(message):
        print(message)

    def handle_disconnect(code, message, *reason):  # 健康檢查逾時會多帶一個參數，用 *reason 接住
        print("market data disconnect: {}, {} {}".format(code, message, reason or ""))
        if not stop.is_set():
            reconnect_needed.set()  # 只通知，不在此直接重連

    def handle_error(error):
        print("market data error: {}".format(error))

    def subscribe_all():
        for params in SUBSCRIPTIONS:
            futopt.subscribe(params)

    def connect_with_timeout(seconds=CONNECT_TIMEOUT):
        # 官方說明：網路不通時 connect() 不會返回，另開執行緒加上逾時
        result = {}

        def run():
            try:
                futopt.connect()
            except Exception as error:  # noqa: BLE001
                result["error"] = error

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(seconds)
        if worker.is_alive():
            raise TimeoutError("connect timeout")
        if "error" in result:
            raise result["error"]

    def reconnect_loop():
        retries = 0
        while reconnect_needed.wait():
            if stop.is_set():
                return
            reconnect_needed.clear()
            retries += 1
            if retries > MAX_RETRIES:
                print("Reconnect failed {} times, please check the network".format(MAX_RETRIES))
                # TODO: 在這裡接你自己的通知（Telegram、LINE、Email…），半夜斷線才會知道
                stop.set()
                return
            time.sleep(min(2 ** retries, 30))  # 等 2、4、8、16、30 秒後再試
            try:
                futopt.disconnect()  # 先重設連線狀態，否則 connect() 不會重新連線
                connect_with_timeout()
                subscribe_all()
                retries = 0
                print("Reconnected and resubscribed")
            except Exception as error:  # noqa: BLE001
                print("Reconnect failed: {}".format(error))
                reconnect_needed.set()

    futopt.on("message", handle_message)
    futopt.on("disconnect", handle_disconnect)
    futopt.on("error", handle_error)
    threading.Thread(target=reconnect_loop, daemon=True).start()

    futopt.connect()
    subscribe_all()
    print("已訂閱 {}，Ctrl+C 結束".format(SUBSCRIPTIONS))

    try:
        while not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        reconnect_needed.set()  # 喚醒背景執行緒讓它結束
        futopt.disconnect()
        sdk.logout()
        print("已斷線並登出")


if __name__ == "__main__":
    main()
