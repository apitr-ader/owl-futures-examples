"""01 登入並列出期貨帳號（對應 Threads P5、P10）

執行：python examples/01_login.py

- 帳密＋憑證登入：sdk.login(id, password, cert_path, cert_pass)
- API Key 登入（SDK >= v2.2.7，選用）：sdk.apikey_login(id, api_key, cert_path, cert_pass)
  在 .env 設 FUBON_LOGIN_MODE=apikey 即可切換。

官方文件：
https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/login/loginPassword.txt
https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/login/loginAPIKey.txt
https://www.fbs.com.tw/TradeAPI/docs/trading-future/library/python/logout.txt
"""

from _common import import_sdk, login, mask

FubonSDK = import_sdk()


def main():
    sdk = FubonSDK()
    accounts = login(sdk)
    try:
        print("登入成功，共 {} 個帳號：".format(len(accounts.data)))
        for acc in accounts.data:
            # Account 欄位（官方文件）：name、account、branch_no、account_type
            print("  - 類型={} 分公司={} 帳號={}".format(
                acc.account_type, acc.branch_no, mask(acc.account)))
    finally:
        sdk.logout()
        print("已登出")


if __name__ == "__main__":
    main()
