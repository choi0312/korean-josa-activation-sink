from typing import List
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def fdr_bh(pvals: List[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out

def cohen_dz_from_diff(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) <= 1:
        return np.nan
    sd = diff.std(ddof=1)
    if sd <= 1e-12:
        return np.nan
    return float(diff.mean() / sd)

def sentence_layer_pair_summary(df: pd.DataFrame, metric: str, use_strict_josa: bool = False, exclude_special: bool = True) -> pd.DataFrame:
    work = df.copy()
    if exclude_special:
        work = work[~work["is_special"]].copy()
    if use_strict_josa:
        work["group_is_josa"] = work["is_josa_strict"].astype(bool)
        work = work[(work["group_is_josa"]) | (~work["is_josa"])].copy()
    else:
        work["group_is_josa"] = work["is_josa"].astype(bool)
    rows = []
    for (sid, layer), g in work.groupby(["sentence_id", "layer"]):
        gj = g[g["group_is_josa"]]
        gn = g[~g["group_is_josa"]]
        if len(gj) == 0 or len(gn) == 0:
            continue
        rows.append({"sentence_id": sid, "layer": layer, "n_josa_tokens": len(gj), "n_nonjosa_tokens": len(gn), "mean_josa": gj[metric].mean(), "mean_nonjosa": gn[metric].mean(), "median_josa": gj[metric].median(), "median_nonjosa": gn[metric].median()})
    out = pd.DataFrame(rows)
    if len(out):
        out["diff_josa_minus_nonjosa"] = out["mean_josa"] - out["mean_nonjosa"]
    return out

def run_layerwise_paired_test(df: pd.DataFrame, metric: str, use_strict_josa: bool = False, alternative: str = "greater", min_pairs: int = 8) -> pd.DataFrame:
    summary = sentence_layer_pair_summary(df, metric, use_strict_josa=use_strict_josa)
    rows = []
    if len(summary) == 0:
        return pd.DataFrame()
    for layer, g in summary.groupby("layer"):
        g = g.dropna(subset=["mean_josa", "mean_nonjosa", "diff_josa_minus_nonjosa"]).copy()
        if len(g) < min_pairs:
            rows.append({"layer": layer, "n_sentence_pairs": len(g), "mean_diff_josa_minus_nonjosa": np.nan, "median_diff_josa_minus_nonjosa": np.nan, "cohen_dz": np.nan, "p_value": np.nan})
            continue
        diff = g["diff_josa_minus_nonjosa"].values
        try:
            stat = wilcoxon(g["mean_josa"].values, g["mean_nonjosa"].values, zero_method="wilcox", alternative=alternative)
            p_value = float(stat.pvalue)
        except Exception:
            p_value = np.nan
        rows.append({"layer": int(layer), "n_sentence_pairs": int(len(g)), "mean_diff_josa_minus_nonjosa": float(np.nanmean(diff)), "median_diff_josa_minus_nonjosa": float(np.nanmedian(diff)), "cohen_dz": cohen_dz_from_diff(diff), "p_value": p_value})
    out = pd.DataFrame(rows).sort_values("layer")
    valid = out["p_value"].notna()
    out.loc[valid, "p_value_fdr_bh"] = fdr_bh(out.loc[valid, "p_value"].tolist())
    out["significant_fdr_0_05"] = out["p_value_fdr_bh"] < 0.05
    out["metric"] = metric
    out["use_strict_josa"] = use_strict_josa
    out["alternative"] = alternative
    return out
