"""
09_event_study.py

Formal event study around the November 2, 2022 CoinDesk article disclosing
Alameda Research's balance sheet composition. Tests whether the qualitative
regime change documented in Section 4.2 (transaction count rising sharply
while flow magnitude rose only modestly) is supported by formal structural
break tests.

Tests three families on two outcome series:
    Outcome 1: hourly on-chain transaction count (onchain_tx_count)
    Outcome 2: hourly absolute flow magnitude (abs_flow_usd)

Test families:
    A. Welch's t-test for difference in pre/post means with HAC SE
    B. Chow test at Nov 2, 2022 14:44 UTC, on AR(1) model
    C. Bai-Perron endogenous breakpoint test, 1-3 breaks, BIC selection

Sample window: Oct 15 to Nov 8 19:00 UTC (excludes post-halt regime)
Pre-news: Oct 15 - Nov 1 23:59 UTC
Post-news (pre-halt): Nov 2 00:00 UTC - Nov 8 19:00 UTC

Inputs:  data/hourly_merged_v2.parquet
Outputs: outputs/event_study_results.txt (table for paper appendix)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = Path("data/hourly_merged_v2.parquet")
OUT_PATH = Path("outputs/event_study_results.txt")

# Event timestamp: CoinDesk article publication
EVENT = pd.Timestamp("2022-11-02 14:44", tz="UTC")
# Sample window
SAMPLE_START = pd.Timestamp("2022-10-15", tz="UTC")
SAMPLE_END = pd.Timestamp("2022-11-08 19:00", tz="UTC")  # exclude operational halt


def load_data():
    df = pd.read_parquet(DATA_PATH)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.loc[SAMPLE_START:SAMPLE_END].copy()
    df = df.dropna(subset=["onchain_tx_count", "abs_flow_usd"])
    df["post"] = (df.index >= EVENT).astype(int)
    df["t"] = np.arange(len(df))
    return df


def welch_hac(df, col, label):
    """Pre/post means with HAC standard errors via OLS on a constant + post indicator."""
    y = df[col].values
    X = sm.add_constant(df["post"].values)
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    pre_mean = m.params[0]
    diff = m.params[1]
    diff_se = m.bse[1]
    diff_t = m.tvalues[1]
    diff_p = m.pvalues[1]

    pre = df.loc[df["post"] == 0, col]
    post = df.loc[df["post"] == 1, col]

    return {
        "label": label,
        "n_pre": len(pre),
        "n_post": len(post),
        "pre_mean": pre.mean(),
        "post_mean": post.mean(),
        "ratio": post.mean() / pre.mean() if pre.mean() != 0 else np.nan,
        "diff": diff,
        "diff_se_hac": diff_se,
        "t_stat": diff_t,
        "p_value": diff_p,
    }


def chow_test_ar1(df, col, label):
    """Chow test for structural break at Nov 2 in an AR(1) model.

    Null: same AR(1) coefficients pre and post.
    Alt:  coefficients differ (intercept and/or AR(1) slope).
    Test statistic: F((k, n - 2k)) where k = number of params per regime = 2.
    """
    y = df[col].values
    y_lag = pd.Series(y).shift(1).values
    valid = ~np.isnan(y_lag)

    y = y[valid]
    y_lag = y_lag[valid]
    post = df["post"].values[valid]

    n = len(y)
    k = 2  # const + AR(1)

    # Restricted: single regime
    X_r = sm.add_constant(y_lag)
    m_r = sm.OLS(y, X_r).fit()
    rss_r = (m_r.resid ** 2).sum()

    # Unrestricted: two regimes (interactions)
    X_u = np.column_stack([
        np.ones(n),
        y_lag,
        post,
        post * y_lag,
    ])
    m_u = sm.OLS(y, X_u).fit()
    rss_u = (m_u.resid ** 2).sum()

    f_stat = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)

    return {
        "label": label,
        "n": n,
        "f_stat": f_stat,
        "df1": k,
        "df2": n - 2 * k,
        "p_value": p_value,
    }


def bai_perron_grid(df, col, label, n_breaks_max=3, trim=0.15):
    """Approximate Bai-Perron via grid search over candidate breakpoints.

    For each k in {1, 2, 3}, search over (k-tuples of) candidate
    breakpoints to minimize SSR. Then compare BIC across k = 0, 1, 2, 3
    where k = 0 is the no-break baseline.

    Trim: don't allow breaks within `trim` fraction of either endpoint.
    """
    y = df[col].values
    n = len(y)
    trim_n = int(np.ceil(n * trim))
    candidates = list(range(trim_n, n - trim_n))

    def _segmented_ssr(breaks):
        """SSR when fitting a constant per segment defined by `breaks`."""
        edges = [0] + sorted(breaks) + [n]
        ssr = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            seg = y[a:b]
            if len(seg) > 0:
                ssr += ((seg - seg.mean()) ** 2).sum()
        return ssr

    def _bic(ssr, k_breaks):
        # Each segment has 1 parameter (mean); k breaks => k+1 params
        n_params = (k_breaks + 1)
        return n * np.log(ssr / n) + n_params * np.log(n)

    results = []
    # k = 0 baseline
    ssr0 = _segmented_ssr([])
    results.append({"k": 0, "breaks": [], "breaks_dt": [], "ssr": ssr0, "bic": _bic(ssr0, 0)})

    # k = 1
    best = (np.inf, None)
    for c in candidates:
        ssr = _segmented_ssr([c])
        if ssr < best[0]:
            best = (ssr, [c])
    if best[1]:
        results.append({"k": 1, "breaks": best[1],
                        "breaks_dt": [df.index[b].strftime("%Y-%m-%d %H:%M UTC") for b in best[1]],
                        "ssr": best[0], "bic": _bic(best[0], 1)})

    # k = 2 (greedy: hold the k=1 break, search for second)
    if best[1]:
        anchor = best[1][0]
        best2 = (np.inf, None)
        for c in candidates:
            if abs(c - anchor) < trim_n:
                continue
            ssr = _segmented_ssr([anchor, c])
            if ssr < best2[0]:
                best2 = (ssr, sorted([anchor, c]))
        if best2[1]:
            results.append({"k": 2, "breaks": best2[1],
                            "breaks_dt": [df.index[b].strftime("%Y-%m-%d %H:%M UTC") for b in best2[1]],
                            "ssr": best2[0], "bic": _bic(best2[0], 2)})

        # k = 3 (greedy continuation)
        if best2[1]:
            anchors = best2[1]
            best3 = (np.inf, None)
            for c in candidates:
                if any(abs(c - a) < trim_n for a in anchors):
                    continue
                ssr = _segmented_ssr(sorted(anchors + [c]))
                if ssr < best3[0]:
                    best3 = (ssr, sorted(anchors + [c]))
            if best3[1]:
                results.append({"k": 3, "breaks": best3[1],
                                "breaks_dt": [df.index[b].strftime("%Y-%m-%d %H:%M UTC") for b in best3[1]],
                                "ssr": best3[0], "bic": _bic(best3[0], 3)})

    return {"label": label, "results": results}


def format_results(welch_results, chow_results, bp_results):
    lines = []
    lines.append("=" * 78)
    lines.append("APPENDIX A.4: EVENT STUDY AROUND NOVEMBER 2, 2022 COINDESK DISCLOSURE")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Event timestamp:  {EVENT.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Sample window:    {SAMPLE_START.strftime('%Y-%m-%d')} to {SAMPLE_END.strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("Sample excludes the post-halt period (Nov 8 19:00 UTC onward).")
    lines.append("")

    lines.append("-" * 78)
    lines.append("PANEL A. PRE/POST MEAN COMPARISON WITH HAC STANDARD ERRORS")
    lines.append("-" * 78)
    lines.append("")
    hdr = f"{'Series':<28}{'n_pre':>8}{'n_post':>8}{'mean_pre':>14}{'mean_post':>14}{'ratio':>10}{'t (HAC)':>10}{'p':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in welch_results:
        lines.append(
            f"{r['label']:<28}{r['n_pre']:>8d}{r['n_post']:>8d}"
            f"{r['pre_mean']:>14.2f}{r['post_mean']:>14.2f}"
            f"{r['ratio']:>10.2f}{r['t_stat']:>+10.2f}{r['p_value']:>10.4f}"
        )
    lines.append("")
    lines.append("HAC standard errors (Newey-West, lag = 24 hours).")
    lines.append("")

    lines.append("-" * 78)
    lines.append("PANEL B. CHOW TEST AT NOV 2, AR(1) MODEL")
    lines.append("-" * 78)
    lines.append("")
    hdr = f"{'Series':<28}{'n':>6}{'F-stat':>12}{'df1':>6}{'df2':>6}{'p-value':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in chow_results:
        lines.append(
            f"{r['label']:<28}{r['n']:>6d}{r['f_stat']:>12.3f}"
            f"{r['df1']:>6d}{r['df2']:>6d}{r['p_value']:>12.4f}"
        )
    lines.append("")
    lines.append("Null: same AR(1) intercept and slope pre/post Nov 2.")
    lines.append("")

    lines.append("-" * 78)
    lines.append("PANEL C. BAI-PERRON BREAKPOINT SEARCH (GRID, BIC SELECTION)")
    lines.append("-" * 78)
    lines.append("")
    for bp in bp_results:
        lines.append(f"Series: {bp['label']}")
        lines.append(f"{'k':>3}  {'BIC':>14}  {'Breakpoints':<60}")
        for r in bp["results"]:
            bp_str = ", ".join(r["breaks_dt"]) if r["breaks_dt"] else "(no breaks)"
            lines.append(f"{r['k']:>3d}  {r['bic']:>14.2f}  {bp_str:<60}")
        best = min(bp["results"], key=lambda x: x["bic"])
        lines.append(f"  -> Selected by BIC: k = {best['k']} breaks")
        lines.append("")
    lines.append("Trim = 15% of sample at each endpoint. Greedy grid for k > 1.")
    lines.append("")

    return "\n".join(lines)


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = load_data()
    print(f"Sample size: {len(df)} hourly observations")
    print(f"Pre-event:  {(df['post'] == 0).sum()} hours")
    print(f"Post-event: {(df['post'] == 1).sum()} hours")
    print()

    print("Running Panel A: Welch t-test with HAC SE...")
    welch_results = [
        welch_hac(df, "onchain_tx_count", "Hourly transaction count"),
        welch_hac(df, "abs_flow_usd", "Hourly absolute flow ($)"),
    ]

    print("Running Panel B: Chow test on AR(1)...")
    chow_results = [
        chow_test_ar1(df, "onchain_tx_count", "Hourly transaction count"),
        chow_test_ar1(df, "abs_flow_usd", "Hourly absolute flow ($)"),
    ]

    print("Running Panel C: Bai-Perron grid search...")
    bp_results = [
        bai_perron_grid(df, "onchain_tx_count", "Hourly transaction count"),
        bai_perron_grid(df, "abs_flow_usd", "Hourly absolute flow ($)"),
    ]

    out = format_results(welch_results, chow_results, bp_results)
    print()
    print(out)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"\nResults written to {OUT_PATH}")


if __name__ == "__main__":
    main()
