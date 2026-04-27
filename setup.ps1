$projectDir = "C:\research\ftx-liquidity"
New-Item -ItemType Directory -Force -Path $projectDir | Out-Null
Set-Location $projectDir

@'
import os
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path

DATA_DIR = Path("data/binance")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
START_DATE = datetime(2022, 10, 15)
END_DATE = datetime(2022, 11, 25)

BASE_URL = "https://data.binance.vision/data/spot/daily/bookDepth"

def download_day(symbol, date):
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-bookDepth-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    out_path = DATA_DIR / filename
    if out_path.exists():
        return out_path
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"  Failed: {url} (status {r.status_code})")
        return None
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

def main():
    current = START_DATE
    days = []
    while current <= END_DATE:
        days.append(current)
        current += timedelta(days=1)
    print(f"Downloading {len(days)} days of {SYMBOL} bookDepth data...")
    for date in tqdm(days):
        download_day(SYMBOL, date)
    print(f"\nDone. Files in {DATA_DIR.resolve()}")

if __name__ == "__main__":
    main()
'@ | Out-File -FilePath "download_binance.py" -Encoding UTF8

@'
import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

ETHERSCAN_API_KEY = "HWDDGHQNNICPICVP7EATJF4ZSJ2C78FWPI"

DATA_DIR = Path("data/onchain")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TOKENS = {
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
}

FTX_WALLETS = {
    "ftx_cold": "0xC098B2a3Aa256D2140208C3de6543aAEf5cd3A94",
    "ftx_hot":  "0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2",
    "ftx_main": "0x83a127952d266A6eA306c40Ac62A4a70668FE3BD",
    "alameda":  "0x3cD751E6b0078Be393132286c442345e5DC49699",
}

START_BLOCK = 15770000
END_BLOCK   = 16060000

BASE_URL = "https://api.etherscan.io/api"

def fetch_token_transfers(wallet, token_contract, start_block, end_block):
    all_txs = []
    page = 1
    while True:
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": token_contract,
            "address": wallet,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": 10000,
            "sort": "asc",
            "apikey": ETHERSCAN_API_KEY,
        }
        r = requests.get(BASE_URL, params=params, timeout=30)
        data = r.json()
        if data.get("status") != "1":
            if data.get("message") == "No transactions found":
                break
            print(f"  Error: {data.get('message')}")
            break
        result = data.get("result", [])
        all_txs.extend(result)
        if len(result) < 10000:
            break
        page += 1
        time.sleep(0.25)
    return all_txs

def main():
    all_records = []
    for wallet_name, wal