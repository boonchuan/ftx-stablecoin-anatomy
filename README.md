# On-Chain Anatomy of the FTX Run

Replication code and data documentation for an empirical descriptive paper on stablecoin flows from the FTX Hot Wallet during the November 2022 collapse.

## Author

Boon Chuan Lim
ORCID: [0009-0005-8477-9393](https://orcid.org/0009-0005-8477-9393)

## Paper

The paper itself lives at `paper/ftx_paper_draft.docx`. SSRN posting forthcoming.

## What this repo contains

| Path | Contents |
|---|---|
| `paper/` | Draft manuscript and figures |
| `scripts/` | Numbered pipeline scripts (run in order, 01 → 08) |
| `outputs/` | Pre-computed numerical outputs (Appendix A.3) |
| `data/` | Schema documentation; data files are gitignored due to size |

## Replication

### Prerequisites

- Python 3.8 or newer
- Free Etherscan API V2 key from [etherscan.io/myapikey](https://etherscan.io/myapikey)
- ~5 GB free disk for Binance aggTrades data

### Setup

```powershell
# Clone and enter
git clone https://github.com/boonchuan/ftx-stablecoin-anatomy.git
cd ftx-stablecoin-anatomy

# Virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Configure your Etherscan key
copy .env.example .env
# Edit .env, paste your key after ETHERSCAN_API_KEY=
```

### Run the pipeline

Scripts are numbered in dependency order:

```powershell
python scripts\01_download_ftx_flows.py     # ~2 min — fetches ERC-20 transfers
python scripts\02_download_binance.py       # ~30 min — Binance aggTrades zips
python scripts\03_merge_and_figures.py      # ~1 min — produces hourly_merged_v2.parquet + figures
python scripts\04_compute_stats.py          # ~5 sec — Tables 1, 2, 3 numbers
python scripts\05_descriptive.py            # ~10 sec — supplementary descriptive stats
python scripts\06_regression.py             # ~5 sec — Appendix A.1 OLS-HAC
python scripts\07_placebo.py                # ~30 sec — Appendix A.1 circular-shift test
python scripts\08_robustness.py             # ~10 sec — Appendix A.2 reverse-direction test
python scripts\09_subhourly_analysis.py     # ~10 sec — Appendix A.3 IAT distributions
```

## Data sources

All data is publicly available and free:

- **Etherscan API V2** — ERC-20 transfers for four publicly-tagged FTX-affiliated wallets (USDT and USDC), Oct 15 to Nov 25, 2022. See `data/README.md` for full address list.
- **Binance Vision** ([data.binance.vision](https://data.binance.vision/)) — aggTrades for BTC/USDT spot, daily files for the same window.

## License

MIT. See `LICENSE`.

## Citation

If you use this code or data documentation, please cite:
Lim, B. C. (2026). On-Chain Anatomy of the FTX Run: Pre-Disclosure Stablecoin
Outflows and the Operational Halt. Working paper.

A machine-readable `CITATION.cff` will be added once the SSRN identifier is assigned.