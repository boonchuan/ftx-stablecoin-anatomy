import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_parquet("data/hourly_merged.parquet")
print(f"Loaded {len(df)} hourly observations")

# Setup base variables
df["log_volume"] = np.log(df["volume"].replace(0, np.nan))
df["log_volume_lag1"] = df["log_volume"].shift(1)
df["log_abs_flow"] = np.log(df["abs_flow_usd"] + 1)
df["log_abs_flow_lag1"] = df["log_abs_flow"].shift(1)

# Per-stablecoin lagged flow
flows = pd.read_parquet("data/onchain/ftx_flows.parquet")
flows = flows[flows["wallet"] == "ftx_hot"].copy()
flows["datetime"] = pd.to_datetime(flows["datetime"])
flows["signed"] = np.where(flows["direction"]=="out", -flows["amount"], flows["amount"])

usdt = (flows[flows["token"]=="USDT"].set_index("datetime")
        .resample("1h")["signed"].sum().abs())
usdc = (flows[flows["token"]=="USDC"].set_index("datetime")
        .resample("1h")["signed"].sum().abs())
df["log_usdt_lag1"] = np.log(usdt + 1).shift(1)
df["log_usdc_lag1"] = np.log(usdc + 1).shift(1)

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

# ===========================================================================
# CHECK 1: USDT vs USDC split, but ONLY on post-news subsample
# ===========================================================================
print("\n" + "="*70)
print("CHECK 1: USDT vs USDC split, post-news subsample only")
print("If both stablecoins predict, the combined-series result is robust.")
print("If neither does, the combined effect is an aggregation artifact.")
print("="*70)

post = df.loc["2022-11-02":]
hac_ols(post["log_volume"], post[["log_usdt_lag1", "log_volume_lag1"]],
        "Post-news, USDT only")
hac_ols(post["log_volume"], post[["log_usdc_lag1", "log_volume_lag1"]],
        "Post-news, USDC only")
hac_ols(post["log_volume"], post[["log_usdt_lag1", "log_usdc_lag1", "log_volume_lag1"]],
        "Post-news, both stablecoins together")

# ===========================================================================
# CHECK 2: Sliding window coefficient
# ===========================================================================
print("\n" + "="*70)
print("CHECK 2: Rolling 100-hour coefficient on log_abs_flow_lag1")
print("Shows whether relationship strengthens around Nov 2 visibly")
print("="*70)

work = df.dropna(subset=["log_volume", "log_abs_flow_lag1", "log_volume_lag1"]).copy()
W = 100
betas, ses, tstats, midpoints = [], [], [], []

for i in range(W, len(work)):
    sub = work.iloc[i-W:i]
    X = sm.add_constant(sub[["log_abs_flow_lag1", "log_volume_lag1"]])
    try:
        m = sm.OLS(sub["log_volume"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        betas.append(m.params["log_abs_flow_lag1"])
        ses.append(m.bse["log_abs_flow_lag1"])
        tstats.append(m.tvalues["log_abs_flow_lag1"])
        midpoints.append(sub.index[W//2])
    except Exception:
        continue

rolling = pd.DataFrame({
    "midpoint": midpoints,
    "beta": betas,
    "se": ses,
    "tstat": tstats,
}).set_index("midpoint")
print(f"\nComputed {len(rolling)} rolling windows")
print(f"\nMean beta pre-Nov2: {rolling.loc[:'2022-11-02', 'beta'].mean():+.4f}")
print(f"Mean beta post-Nov2: {rolling.loc['2022-11-02':, 'beta'].mean():+.4f}")
print(f"Max beta: {rolling['beta'].max():+.4f} at {rolling['beta'].idxmax()}")
print(f"Min beta: {rolling['beta'].min():+.4f} at {rolling['beta'].idxmin()}")

# Plot rolling coefficient
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(rolling.index, rolling["beta"], color="darkblue", linewidth=1.5)
axes[0].fill_between(rolling.index, rolling["beta"] - 1.96*rolling["se"],
                     rolling["beta"] + 1.96*rolling["se"], alpha=0.2, color="blue")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_ylabel("β (log_abs_flow_lag1)")
axes[0].set_title("Rolling 100-hour Regression Coefficient (with 95% CI)")

axes[1].plot(rolling.index, rolling["tstat"], color="darkred", linewidth=1.5)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].axhline(1.96, color="red", linestyle=":", alpha=0.5, label="t=1.96")
axes[1].axhline(-1.96, color="red", linestyle=":", alpha=0.5)
axes[1].set_ylabel("t-statistic")
axes[1].set_xlabel("Window midpoint")

events = [("2022-11-02","CoinDesk"),("2022-11-06","CZ"),("2022-11-08","LOI"),
          ("2022-11-10","Halt"),("2022-11-11","Bankruptcy")]
for d, lbl in events:
    for ax in axes:
        ax.axvline(pd.Timestamp(d), color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    axes[0].text(pd.Timestamp(d), rolling["beta"].max()*0.95, lbl,
                 rotation=90, fontsize=8, va="top")

plt.tight_layout()
plt.savefig("rolling_coef.png", dpi=120)
print("Saved: rolling_coef.png")

# ===========================================================================
# CHECK 3: Drop ±24h around Nov 8 event window, re-run
# ===========================================================================
print("\n" + "="*70)
print("CHECK 3: Exclude ±24h around Nov 8 12:00 UTC")
print("If the result is event-driven, coefficient should drop sharply.")
print("="*70)

event = pd.Timestamp("2022-11-08 12:00:00")
mask = (df.index < event - pd.Timedelta(hours=24)) | (df.index > event + pd.Timedelta(hours=24))
no_event = df[mask].copy()
print(f"Excluded {len(df) - len(no_event)} hours around event")
print(f"Remaining: {len(no_event)} hours")

hac_ols(no_event["log_volume"], no_event[["log_abs_flow_lag1", "log_volume_lag1"]],
        "Full window minus Nov 8 event ±24h")

# Same on post-news subsample
post_no_event = no_event.loc["2022-11-02":]
hac_ols(post_no_event["log_volume"], post_no_event[["log_abs_flow_lag1", "log_volume_lag1"]],
        "Post-news minus Nov 8 event ±24h")

print("\nDone.")