"""
fix_and_plot.py

(1) Rebuild hourly on-chain transaction count from data/onchain/ftx_flows.parquet
(2) Merge into hourly_merged.parquet (preserved alongside under hourly_merged_v2.parquet)
(3) Print the true halt-window numbers so we can verify the paper claims
(4) Generate both paper figures with the correct data
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ONCHAIN_PATH = Path("data/onchain/ftx_flows.parquet")
MERGED_PATH  = Path("data/hourly_merged.parquet")
OUT_PARQUET  = Path("data/hourly_merged_v2.parquet")
DPI = 300

# Time windows (UTC)
FIG1_START = pd.Timestamp("2022-10-17", tz="UTC")
FIG1_END   = pd.Timestamp("2022-11-13 23:59", tz="UTC")
FIG2_START = pd.Timestamp("2022-11-07", tz="UTC")
FIG2_END   = pd.Timestamp("2022-11-12 23:59", tz="UTC")

EVENT_COINDESK   = pd.Timestamp("2022-11-02 14:44", tz="UTC")
EVENT_CZ_TWEET   = pd.Timestamp("2022-11-06 15:47", tz="UTC")
EVENT_HALT       = pd.Timestamp("2022-11-08 19:00", tz="UTC")
EVENT_BANKRUPTCY = pd.Timestamp("2022-11-11 14:30", tz="UTC")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 9,
})


def rebuild_onchain_hourly():
    print("Step 1: Rebuilding hourly on-chain stats from raw transactions...")
    raw = pd.read_parquet(ONCHAIN_PATH)
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)

    # Restrict to FTX Hot Wallet only (consistent with paper)
    hot = raw[raw["wallet"] == "ftx_hot"].copy()
    print(f"  Total transactions across all wallets: {len(raw):,}")
    print(f"  FTX Hot Wallet transactions:           {len(hot):,}")

    hot["hour"] = hot["datetime"].dt.floor("h")

    hourly_tx = (
        hot.groupby("hour")
           .agg(onchain_tx_count=("amount", "size"),
                onchain_total_amount=("amount", "sum"))
           .reset_index()
    )
    hourly_tx = hourly_tx.set_index("hour")

    print(f"  Hourly bins covered: {len(hourly_tx):,}")
    print(f"  Mean tx/hour: {hourly_tx['onchain_tx_count'].mean():.1f}")
    print(f"  Max tx/hour:  {hourly_tx['onchain_tx_count'].max():,}")
    return hourly_tx


def merge_and_save(hourly_tx):
    print("\nStep 2: Merging into existing hourly_merged.parquet...")
    merged = pd.read_parquet(MERGED_PATH)
    if merged.index.tz is None:
        merged.index = merged.index.tz_localize("UTC")

    # Drop any old onchain columns to avoid double-merge issues
    drop_cols = [c for c in merged.columns if c.startswith("onchain_")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    out = merged.join(hourly_tx, how="left")
    out["onchain_tx_count"] = out["onchain_tx_count"].fillna(0).astype(int)
    out["onchain_total_amount"] = out["onchain_total_amount"].fillna(0.0)

    out.to_parquet(OUT_PARQUET)
    print(f"  Wrote {OUT_PARQUET}")
    print(f"  Columns now: {list(out.columns)}")
    return out


def verify_halt_numbers(df):
    print("\nStep 3: Verifying paper claims against actual data...")
    halt_window = pd.Timestamp("2022-11-08 12:00", tz="UTC")
    end_window  = pd.Timestamp("2022-11-09 06:00", tz="UTC")

    sl = df.loc[halt_window:end_window, "onchain_tx_count"]
    print("\n  Hour-by-hour around the halt (Nov 8 12:00 -> Nov 9 06:00 UTC):")
    for ts, v in sl.items():
        marker = "  <-- HALT?" if v < 50 and ts >= EVENT_HALT - pd.Timedelta(hours=2) else ""
        print(f"    {ts.strftime('%Y-%m-%d %H:%M UTC')}  tx={int(v):>5}{marker}")

    # Find the largest hour-over-hour drop in tx count
    tx = df["onchain_tx_count"].loc[FIG1_START:FIG1_END]
    deltas = tx.diff()
    biggest_drop_idx = deltas.idxmin()
    bd_after = int(tx.loc[biggest_drop_idx])
    bd_before = int(tx.loc[biggest_drop_idx - pd.Timedelta(hours=1)])
    print(f"\n  Largest hour-over-hour drop in study window:")
    print(f"    {biggest_drop_idx.strftime('%Y-%m-%d %H:%M UTC')}: "
          f"{bd_before} -> {bd_after}  (delta = {bd_after - bd_before})")

    # Count zero-tx hours in the post-halt window
    post = df.loc[EVENT_HALT:FIG1_END, "onchain_tx_count"]
    n_zero = int((post == 0).sum())
    print(f"\n  Hours with exactly zero on-chain tx after Nov 8 19:00 UTC: {n_zero}")

    # Pre/post mean comparison (paper claims 129 -> 191 mean tx/hr)
    pre  = df.loc[pd.Timestamp("2022-10-15", tz="UTC"):
                  pd.Timestamp("2022-11-01 23:59", tz="UTC"), "onchain_tx_count"]
    post = df.loc[pd.Timestamp("2022-11-02", tz="UTC"):
                  pd.Timestamp("2022-11-09 23:59", tz="UTC"), "onchain_tx_count"]
    print(f"\n  Pre-news mean tx/hour  (Oct 15 - Nov 1):  {pre.mean():.1f}")
    print(f"  Post-news mean tx/hour (Nov 2 - Nov 9):  {post.mean():.1f}")
    print(f"  Ratio: {post.mean() / pre.mean():.2f}x")


def add_event_lines(ax, events):
    for ts, _, color in events:
        ax.axvline(ts, color=color, linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)


def label_events(ax, events):
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    for ts, label, color in events:
        ax.text(ts, ymax - 0.02 * yspan, " " + label,
                rotation=90, va="top", ha="left",
                fontsize=8, color=color, alpha=0.9)


def make_figure_1(df):
    print("\nStep 4a: Generating Figure 1 (3-panel overview)...")
    sub = df.loc[FIG1_START:FIG1_END].copy()

    sub["cum_net_outflow"] = -sub["net_inflow_usd"].cumsum()
    sub["cum_net_outflow"] -= sub["cum_net_outflow"].iloc[0]
    sub["net_outflow"] = -sub["net_inflow_usd"]

    events = [
        (EVENT_COINDESK,   "CoinDesk Nov 2",     "#444"),
        (EVENT_CZ_TWEET,   "CZ tweet Nov 6",     "#444"),
        (EVENT_HALT,       "Halt 19:00 Nov 8",   "#c0392b"),
        (EVENT_BANKRUPTCY, "Chapter 11 Nov 11",  "#444"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1, 1], "hspace": 0.18})

    ax = axes[0]
    ax.fill_between(sub.index, 0, sub["cum_net_outflow"] / 1e6,
                    color="#2E75B6", alpha=0.25, linewidth=0)
    ax.plot(sub.index, sub["cum_net_outflow"] / 1e6,
            color="#1a4f7a", linewidth=1.6)
    ax.set_ylabel("Cumulative net outflow ($M)")
    ax.set_title("A. Cumulative net stablecoin outflow from FTX Hot Wallet")
    add_event_lines(ax, events)
    label_events(ax, events)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)

    ax = axes[1]
    pos = sub["net_outflow"].clip(lower=0) / 1e6
    neg = sub["net_outflow"].clip(upper=0) / 1e6
    ax.bar(sub.index, pos, width=0.04, color="#c0392b", alpha=0.85,
           linewidth=0, label="Net outflow")
    ax.bar(sub.index, neg, width=0.04, color="#2980b9", alpha=0.85,
           linewidth=0, label="Net inflow")
    ax.set_ylabel("Hourly net flow ($M)")
    ax.set_title("B. Hourly net flow (positive = withdrawal)")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    add_event_lines(ax, events)
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.bar(sub.index, sub["onchain_tx_count"], width=0.04,
           color="#555", alpha=0.85, linewidth=0)
    ax.set_ylabel("On-chain transactions per hour")
    ax.set_title("C. Hourly on-chain transaction count from FTX Hot Wallet")
    add_event_lines(ax, events)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(FIG1_START, FIG1_END)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_xlabel("Date (UTC), 2022")

    fig.text(0.5, 0.005,
             "Source: Etherscan API V2; ERC-20 transfers (USDT, USDC) at "
             "FTX Hot Wallet 0x2FAF...6AD2.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig("paper_figure_1.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Wrote paper_figure_1.png")


def make_figure_2(df):
    print("\nStep 4b: Generating Figure 2 (zoomed halt cliff)...")
    sub = df.loc[FIG2_START:FIG2_END].copy()
    cliff_hour = EVENT_HALT

    fig, ax = plt.subplots(figsize=(9, 4.2))

    colors = np.where(sub.index < cliff_hour, "#555", "#c0392b")
    ax.bar(sub.index, sub["onchain_tx_count"], width=0.035,
           color=colors, alpha=0.9, linewidth=0)

    cliff_value = int(sub.loc[cliff_hour, "onchain_tx_count"]) if cliff_hour in sub.index else 0
    prev_hour = cliff_hour - pd.Timedelta(hours=1)
    prev_value = int(sub.loc[prev_hour, "onchain_tx_count"]) if prev_hour in sub.index else 0

    ymax = sub["onchain_tx_count"].max()
    ax.annotate(
        f"19:00 UTC Nov 8:\n{prev_value} -> {cliff_value} tx/hr",
        xy=(cliff_hour, cliff_value),
        xytext=(cliff_hour + pd.Timedelta(hours=18), ymax * 0.6),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        fontsize=10, color="#c0392b", fontweight="bold", ha="left",
    )

    ax.set_ylabel("On-chain transactions per hour")
    ax.set_xlabel("Date (UTC), 2022")
    ax.set_title(
        "FTX Hot Wallet hourly on-chain transaction count, November 7-12, 2022\n"
        "Operational halt visible at hourly resolution"
    )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.set_xlim(FIG2_START, FIG2_END)
    ax.set_ylim(bottom=0)
    ax.axvline(cliff_hour, color="#c0392b", linestyle="--",
               linewidth=0.9, alpha=0.6, zorder=1)

    fig.text(0.5, 0.005,
             "Source: Etherscan API V2; ERC-20 transfers (USDT, USDC) at "
             "FTX Hot Wallet 0x2FAF...6AD2.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig("paper_figure_2.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Wrote paper_figure_2.png")


def main():
    hourly_tx = rebuild_onchain_hourly()
    df = merge_and_save(hourly_tx)
    verify_halt_numbers(df)
    make_figure_1(df)
    make_figure_2(df)
    print("\nAll done.")


if __name__ == "__main__":
    main()
