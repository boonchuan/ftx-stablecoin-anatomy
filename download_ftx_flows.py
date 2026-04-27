import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

ETHERSCAN_API_KEY = "Y23MRPSGEJNGPCSRTJBIHHF21FDR3Q8WYI"

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
END_BLOCK = 16060000
BASE_URL = "https://api.etherscan.io/v2/api"

def fetch_chunk(wallet, token, start_block, end_block):
    """Fetch up to 10000 results in a single block range chunk."""
    params = {
        "chainid": 1,
        "module": "account",
        "action": "tokentx",
        "contractaddress": token,
        "address": wallet,
        "startblock": start_block,
        "endblock": end_block,
        "page": 1,
        "offset": 10000,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    d = r.json()
    if d.get("status") != "1":
        msg = d.get("message", "")
        if msg == "No transactions found":
            return []
        return None  # signal error
    return d.get("result", [])

def fetch(wallet, token, start_block, end_block):
    """Walk block range, splitting if a chunk hits the 10000 cap."""
    out = []
    pending = [(start_block, end_block)]
    while pending:
        lo, hi = pending.pop(0)
        result = fetch_chunk(wallet, token, lo, hi)
        time.sleep(0.25)
        if result is None:
            print("  Error in range", lo, "-", hi, "-- skipping")
            continue
        if len(result) < 10000:
            out.extend(result)
        else:
            # Hit the cap -- split the block range in half and retry both halves
            if hi - lo <= 1:
                # Can't split further -- take what we have
                out.extend(result)
                print("  Warning: > 10000 txs in single block range", lo, "-", hi)
            else:
                mid = (lo + hi) // 2
                pending.insert(0, (mid + 1, hi))
                pending.insert(0, (lo, mid))
                print("  Splitting range", lo, "-", hi, "into halves")
    return out
records = []
for wname, waddr in FTX_WALLETS.items():
    for tname, taddr in TOKENS.items():
        print("Fetching", tname, "for", wname)
        txs = fetch(waddr, taddr, START_BLOCK, END_BLOCK)
        print("  Got", len(txs), "transactions")
        for tx in txs:
            value = int(tx["value"]) / (10 ** int(tx["tokenDecimal"]))
            direction = "in" if tx["to"].lower() == waddr.lower() else "out"
            records.append({
                "timestamp": int(tx["timeStamp"]),
                "datetime": datetime.fromtimestamp(int(tx["timeStamp"])),
                "wallet": wname,
                "token": tname,
                "direction": direction,
                "amount": value,
            })
        time.sleep(0.25)

df = pd.DataFrame(records)
out_path = DATA_DIR / "ftx_flows.parquet"
df.to_parquet(out_path)
print()
print("Saved", len(df), "records to", out_path)
print()
print("Date range:", df["datetime"].min(), "to", df["datetime"].max())
print()
print("Summary by wallet/direction:")
print(df.groupby(["wallet", "direction"])["amount"].agg(["count", "sum"]))