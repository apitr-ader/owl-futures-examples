"""共用工具：讀取環境變數、登入富邦新一代 API、挑出期貨帳號。

所有範例都 import 這個檔案，帳密一律從環境變數（或專案根目錄的 .env）讀取，
程式碼裡不寫任何帳號、密碼或金鑰。

官方文件依據：
- 登入（帳密＋憑證）：https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/login/loginPassword.txt
- 登入（API Key，SDK >= v2.2.7）：https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/login/loginAPIKey.txt
- 快速開始（預設密碼憑證的三參數寫法）：https://www.fbs.com.tw/TradeAPI/docs/trading-future/quickstart.txt
- 登出：https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/logout.txt
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """若有安裝 python-dotenv，就讀取專案根目錄的 .env；沒有也不影響，直接用系統環境變數。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def require_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit("缺少環境變數 {}，請參考 .env.example 設定。".format(name))
    return value


def env_flag(name):
    """把 '1' / 'true' / 'yes' 視為 True，其餘（含未設定）一律 False。"""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def import_sdk():
    try:
        from fubon_neo.sdk import FubonSDK  # 官方套件名稱：fubon_neo
    except ImportError:
        sys.exit(
            "找不到 fubon_neo 套件。請到官方 SDK 下載頁下載對應平台的 .whl 後用 pip 安裝：\n"
            "https://www.fbs.com.tw/TradeAPI/docs/download/download-sdk.txt"
        )
    return FubonSDK


def login(sdk):
    """依 FUBON_LOGIN_MODE（password / apikey）登入，回傳 SDK 的 Result 物件。

    Result 欄位（官方文件）：is_success、data（帳號清單）、message。
    """
    load_env()
    mode = os.environ.get("FUBON_LOGIN_MODE", "password").strip().lower()
    personal_id = require_env("FUBON_ID")
    cert_path = require_env("FUBON_CERT_PATH")
    cert_pass = os.environ.get("FUBON_CERT_PASSWORD", "") or os.environ.get("FUBON_CERT_PASS", "")

    cert_file = Path(cert_path).expanduser()
    if not cert_file.is_absolute():
        cert_file = REPO_ROOT / cert_file  # 相對路徑以 repo 根目錄為準
    if not cert_file.is_file():
        sys.exit("找不到憑證檔：{}（請確認 FUBON_CERT_PATH）".format(cert_file))
    cert_path = str(cert_file)

    if mode == "apikey":
        api_key = require_env("FUBON_API_KEY")
        if not cert_pass:
            # 已實測（SDK 2.4.0）：apikey_login(ID, Key, 憑證路徑, 憑證密碼) 四參數可登入。
            # 官方 apikey_login 範例只示範四個參數（ID、Key、憑證路徑、憑證密碼），
            # 沒有示範「預設密碼憑證」省略第四個參數的寫法，因此這裡要求必填。
            sys.exit("API Key 登入請設定 FUBON_CERT_PASSWORD（預設密碼憑證請填你的登入 ID）。")
        accounts = sdk.apikey_login(personal_id, api_key, cert_path, cert_pass)
    elif mode == "password":
        password = require_env("FUBON_PASSWORD")
        if cert_pass:
            accounts = sdk.login(personal_id, password, cert_path, cert_pass)
        else:
            # 官方快速開始：若憑證選用「預設密碼」，SDK v1.3.2 與較新版本可省略憑證密碼
            accounts = sdk.login(personal_id, password, cert_path)
    else:
        sys.exit("FUBON_LOGIN_MODE 只能是 password 或 apikey，目前是：{}".format(mode))

    if not accounts.is_success:
        sys.exit("登入失敗：{}".format(accounts.message))
    return accounts


def pick_futopt_account(accounts):
    """從登入結果挑出期貨帳號（account_type == 'futopt'）。

    可用 FUBON_ACCOUNT 指定帳號（多帳號歸戶時）。
    """
    wanted = os.environ.get("FUBON_ACCOUNT", "").strip()
    futopt = [a for a in accounts.data if getattr(a, "account_type", "") == "futopt"]
    if wanted:
        futopt = [a for a in futopt if a.account == wanted]
    if not futopt:
        sys.exit("登入成功，但找不到期貨帳號（account_type == 'futopt'）。")
    return futopt[0]


def mask(text, keep=2):
    """印帳號時遮罩，避免截圖外流。"""
    text = str(text)
    if len(text) <= keep:
        return "*" * len(text)
    return text[:keep] + "*" * (len(text) - keep)
