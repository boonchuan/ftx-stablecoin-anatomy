import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_parquet("data/hourly_merged.parquet")
print(f"Loaded {len(df)} hourly observations\n")

# Setup
df["log_volume"] = np.log(df["volume"].replace(0, np.nan))
df["log_volume_lag1"] = df["log_volume"].shift(1)
df["log_abs_flow"] = np.log(df["abs_flow_usd"] + 1)
df["log_abs_flow_lag1"] = df["log_abs_flow"].shift(1)

def hac_ols(y, X, label):
    work = pd.concat([y, X], axis=1).dropna()
    y_c = work[y.name]
    X_c = sm.add_constant(work[X.columns.tolist()])
    m = sm.OLS(y_c, X_c).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    print(f"\n--- {label} ---")
    print(f"N={int(m.nobs)}, R²={m.rsquared:.4f}")
    print(f"{'variable':<28} {'coef':>10} {'se':>10} {'t':>8} {'p':>8}")
    for v in m.params.index:
        c, s, t, p = m.params[v], m.bse[v], m.tvalues[v], m.pvalues[v]
        sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
        print(f"{v:<28} {c:>+10.4f} {s:>10.4f} {t:>+8.2f} {p:>8.4f} {sig}")
    return m

# ---- Pre-news vs post-news split ----
print("="*70)
print("PRE-NEWS vs POST-NEWS regression")
print("Split point: 2022-11-02 (CoinDesk article)")
print("="*70)

pre_news = df.loc[:"2022-11-01 23:59:59"]
post_news = df.loc["2022-11-02":]
print(f"Pre-news: N={len(pre_news)}, Post-news: N={len(post_news)}")

hac_ols(pre_news["log_volume"], pre_news[["log_abs_flow_lag1", "log_volume_lag1"]],
        "Pre-news (Oct 15 - Nov 1)")
hac_ols(post_news["log_volume"], post_news[["log_abs_flow_lag1", "log_volume_lag1"]],
        "Post-news (Nov 2 - Nov 25)")

# ---- Event study around Nov 8 12:00 UTC ----
print("\n" + "="*70)
print("EVENT STUDY: ±24h around Nov 8 12:00 UTC")
print("="*70)

event_time = pd.Timestamp("2022-11-08 12:00:00")
window = df.loc[event_time - pd.Timedelta(hours=24): event_time + pd.Timedelta(hours=24)].copy()
window["hours_from_event"] = (window.index - event_time).total_seconds() / 3600
window["pct_change_price"] = window["mean_price"].pct_change() * 100

print(f"Event window: {len(window)} hourly bins")
print(f"\nBaseline (24h before event):")
pre_event = window[window["hours_from_event"] < 0]
print(f"  Mean log volume: {pre_event['log_volume'].mean():.3f}")
print(f"  Mean realized vol: {pre_event['rv'].mean():.5f}")
print(f"  Mean abs price change: {pre_event['pct_change_price'].abs().mean():.3f}%")

print(f"\nResponse (24h after event):")
post_event = window[window["hours_from_event"] >= 0]
print(f"  Mean log volume: {post_event['log_volume'].mean():.3f}")
print(f"  Mean realized vol: {post_event['rv'].mean():.5f}")
print(f"  Mean abs price change: {post_event['pct_change_price'].abs().mean():.3f}%")

print(f"\nDifference (post - pre):")
print(f"  Volume: {post_event['log_volume'].mean() - pre_event['log_volume'].mean():+.3f}")
print(f"  Vol: {post_event['rv'].mean() - pre_event['rv'].mean():+.5f}")
print(f"  |Price change|: {post_event['pct_change_price'].abs().mean() - pre_event['pct_change_price'].abs().mean():+.3f}%")

# ---- Plot the event study ----
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(window["hours_from_event"], window["mean_price"], color="black", linewidth=1.5, marker="o", markersize=3)
axes[0].set_ylabel("BTC/USDT")
axes[0].set_title("Event Study: ±24h around FTX withdrawal slowdown (Nov 8 12:00 UTC)")
axes[0].axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)

axes[1].bar(window["hours_from_event"], window["abs_flow_usd"]/1e6, width=0.8, color="steelblue")
axes[1].set_ylabel("Abs On-Chain Flow ($M/h)")
axes[1].axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)

axes[2].plot(window["hours_from_event"], window["volume"]/1e6, color="darkgreen", linewidth=1.5, marker="o", markersize=3)
axes[2].set_ylabel("Binance Volume ($M/h)")
axes[2].axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)

axes[3].plot(window["hours_from_event"], window["rv"], color="purple", linewidth=1.5, marker="o", markersize=3)
axes[3].set_ylabel("Realized Vol (1h)")
axes[3].set_xlabel("Hours from event (Nov 8 12:00 UTC)")
axes[3].axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig("event_study.png", dpi=120)
print("\nSaved: event_study.png")

# ---- Summary table ----
print("\n" + "="*70)
print("PAPER-READY SUMMARY: pre-news vs post-news comparison")
print("="*70)
print(f"\n{'period':<25} {'N':>6} {'beta_flow':>12} {'t-stat':>10} {'p-value':>10}")
for label, sub in [("Pre-news (Oct 15-Nov 1)", pre_news), ("Post-news (Nov 2-Nov 25)", post_news), ("Full window", df)]:
    work = sub[["log_volume", "log_abs_flow_lag1", "log_volume_lag1"]].dropna()
    X = sm.add_constant(work[["log_abs_flow_lag1", "log_volume_lag1"]])
    m = sm.OLS(work["log_volume"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    b = m.params["log_abs_flow_lag1"]
    t = m.tvalues["log_abs_flow_lag1"]
    p = m.pvalues["log_abs_flow_lag1"]
    print(f"{label:<25} {int(m.nobs):>6} {b:>+12.4f} {t:>+10.2f} {p:>10.4f}")

print("\nDone.")