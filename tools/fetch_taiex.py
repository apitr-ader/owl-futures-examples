"""下載 TWSE 加權指數日資料（官方 rwd 端點，逐月、禮貌間隔、可續傳）。
MI_5MINS_HIST：開高低收（1999-01 起）；MFI94U：發行量加權股價報酬指數（2003-01 起）；FMTQIK：成交股數/金額/筆數。
輸出 data/index/taiex_daily.parquet（date, open, high, low, close, tr_close, volume, turnover, trades）；月檔快取於 data/index/raw/。
用法：python tools/fetch_taiex.py [--root /workspace/fubon-threads]
"""
import argparse, json, random, time, urllib.request
from pathlib import Path
import pandas as pd

EP = {"MI_5MINS_HIST": ("indicesReport", 1999)  # rwd 路徑 302 → 用舊路徑
, "MFI94U": ("TAIEX", 2003), "FMTQIK": ("afterTrading", 1999),
      "EFTRI_HIST": ("indicesReport", 2000)}  # 電子類/金融保險類（僅收盤，2000-01-04 起）


def get(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
            if j.get("stat") == "OK" or "沒有符合" in str(j.get("stat")):
                return j
        except Exception as e:  # noqa
            print("retry", k, url, e)
        time.sleep(10 * 2 ** k)
    raise RuntimeError(url)


def months(y0):
    return pd.period_range(f"{y0}-01", pd.Timestamp.today().strftime("%Y-%m"), freq="M")


def num(s):
    s = str(s).replace(",", "").strip()
    return float(s) if s not in ("", "--") else float("nan")


def roc(s):
    y, m, d = s.strip().split("/"); return pd.Timestamp(int(y) + 1911, int(m), int(d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/workspace/fubon-threads"); a = ap.parse_args()
    out = Path(a.root) / "data/index"; raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    this = pd.Timestamp.today().strftime("%Y%m")
    for ep, (grp, y0) in EP.items():
        for p in months(y0):
            f = raw / f"{ep}_{p.strftime('%Y%m')}.json"
            if f.exists() and p.strftime("%Y%m") != this:
                continue
            base = "https://www.twse.com.tw/" if ep in ("MI_5MINS_HIST", "EFTRI_HIST") else "https://www.twse.com.tw/rwd/zh/"
            j = get(f"{base}{grp}/{ep}?response=json&date={p.strftime('%Y%m')}01")
            f.write_text(json.dumps(j, ensure_ascii=False)); print(ep, p, len(j.get("data", [])), flush=True)
            time.sleep(3 + 2 * random.random())
    rows = {}
    for ep in ("MI_5MINS_HIST", "MFI94U", "FMTQIK"):
        recs = []
        for f in sorted(raw.glob(f"{ep}_*.json")):
            for r in json.loads(f.read_text()).get("data", []):
                recs.append([roc(r[0])] + [num(x) for x in r[1:]])
        rows[ep] = recs
    px = pd.DataFrame(rows["MI_5MINS_HIST"], columns=["date", "open", "high", "low", "close"]).set_index("date")
    tr = pd.DataFrame(rows["MFI94U"], columns=["date", "tr_close"]).set_index("date")
    vq = pd.DataFrame([r[:4] for r in rows["FMTQIK"]], columns=["date", "volume", "turnover", "trades"]).set_index("date")
    df = px.join(tr, how="outer").join(vq, how="left").sort_index()
    df = df[~df.index.duplicated(keep="last")].reset_index()
    df.to_parquet(out / "taiex_daily.parquet", index=False)
    print(df.shape, df.date.min().date(), df.date.max().date())
    sec = {"electronics": "電子", "finance": "金融保險"}; recs = {k: [] for k in sec}
    for f in sorted(raw.glob("EFTRI_HIST_*.json")):
        j = json.loads(f.read_text()); fl = [x.replace("\u3000", "").replace(" ", "") for x in j.get("fields", [])]
        for k, nm in sec.items():
            c = next((i for i, x in enumerate(fl) if x.startswith(nm)), None)
            if c is not None:
                recs[k] += [(roc(r[0]), num(r[c])) for r in j.get("data", [])]
    for k, v in recs.items():
        d = pd.DataFrame(v, columns=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")
        d.to_parquet(out / f"sector_{k}_daily.parquet", index=False); print(k, d.shape, d.date.min().date(), d.date.max().date())


if __name__ == "__main__":
    main()
