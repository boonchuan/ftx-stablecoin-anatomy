import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet("data/hourly_merged.parquet")
print(f"Loaded {len(df)} hourly observations\n")

# Setup
df["log_volume"] = np.log(df["volume"].replace(0, np.nan))
df["log_volume_lag1"] = df["log_volume"].shift(1)
df["log_abs_flow"] = np.log(df["abs_flow_usd"] + 1)
df["log_abs_flow_lag1"] = df["log_abs_flow"].shift(1)
df["log_rv"] = np.log(df["rv"].replace(0, np.nan))
df["log_rv_lag1"] = df["log_rv"].shift(1)
df["log_hl"] = np.log(df["hl_range"].replace(0, np.nan))
df["log_hl_lag1"] = df["log_hl"].shift(1)

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

# ---- Test 1: Multiple stress outcomes ----
print("="*70)
print("TEST 1: Same flow predictor, different stress outcomes")
print("="*70)
hac_ols(df["log_volume"], df[["log_abs_flow_lag1", "log_volume_lag1"]], "DV: log volume")
hac_ols(df["log_rv"], df[["log_abs_flow_lag1", "log_rv_lag1"]], "DV: log realized vol")
hac_ols(df["log_hl"], df[["log_abs_flow_lag1", "log_hl_lag1"]], "DV: log high-low range")

# ---- Test 2: USDT vs USDC separately ----
print("\n" + "="*70)
print("TEST 2: USDT vs USDC flows")
print("="*70)

flows = pd.read_parquet("data/onchain/ftx_flows.parquet")
flows = flows[flows["wallet"] == "ftx_hot"].copy()
flows["datetime"] = pd.to_datetime(flows["datetime"])
flows["signed"] = np.where(flows["direction"]=="out", -flows["amount"], flows["amount"])

usdt_flow = (flows[flows["token"]=="USDT"].set_index("datetime")
             .resample("1h")["signed"].sum().abs().rename("usdt_abs"))
usdc_flow = (flows[flows["token"]=="USDC"].set_index("datetime")
             .resample("1h")["signed"].sum().abs().rename("usdc_abs"))

df["log_usdt_lag1"] = np.log(usdt_flow + 1).shift(1)
df["log_usdc_lag1"] = np.log(usdc_flow + 1).shift(1)

hac_ols(df["log_volume"], df[["log_usdt_lag1", "log_volume_lag1"]], "DV: log volume, predictor: USDT flow")
hac_ols(df["log_volume"], df[["log_usdc_lag1", "log_volume_lag1"]], "DV: log volume, predictor: USDC flow")
hac_ols(df["log_volume"], df[["log_usdt_lag1", "log_usdc_lag1", "log_volume_lag1"]], "DV: log volume, both")

# ---- Test 3: Reverse direction (volume -> flow) ----
print("\n" + "="*70)
print("TEST 3: Reverse causality test")
print("Does Binance volume predict next-hour on-chain flow?")
print("="*70)

df["log_abs_flow_t"] = df["log_abs_flow"]
hac_ols(df["log_abs_flow_t"], df[["log_volume_lag1", "log_abs_flow_lag1"]],
        "DV: log abs flow, predictor: lagged volume")

# Compare strength of forward vs reverse
print("\nKey comparison:")
print("  Forward: flow_lag1 -> volume   (Test 1, top)")
print("  Reverse: volume_lag1 -> flow   (Test 3)")
print("If reverse > forward, the apparent 'predictive' relationship is reactive.")