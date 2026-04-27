import requests
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data/binance_aggtrades")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
START = datetime(2022, 10, 15)
END = datetime(2022, 11, 25)

base = "https://data.binance.vision/data/spot/daily/aggTrades"

current = START
total = 0
fails = 0
while current <= END:
    date_str = current.strftime("%Y-%m-%d")
    fn = SYMBOL + "-aggTrades-" + date_str + ".zip"
    url = base + "/" + SYMBOL + "/" + fn
    out = DATA_DIR / fn
    if not out.exists():
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            with open(out, "wb") as f:
                f.write(r.content)
            print("OK ", date_str, len(r.content), "bytes")
            total += 1
        else:
            print("FAIL", date_str, r.status_code)
            fails += 1
    else:
        print("skip", date_str)
    current += timedelta(days=1)

print()
print("Downloaded:", total, "Failed:", fails)