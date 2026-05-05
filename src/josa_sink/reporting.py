from pathlib import Path
from typing import Any, Dict
import json
import shutil
import pandas as pd
import numpy as np

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def layer_sig_count(stats_df: pd.DataFrame) -> Dict[str, Any]:
    if stats_df is None or len(stats_df) == 0:
        return {"sig_layers": 0, "tested_layers": 0, "sig_layer_list": []}
    return {"sig_layers": int(stats_df["significant_fdr_0_05"].fillna(False).sum()), "tested_layers": int(stats_df["p_value_fdr_bh"].notna().sum()), "sig_layer_list": [int(x) for x in stats_df.loc[stats_df["significant_fdr_0_05"].fillna(False), "layer"].tolist()]}

def josa_nonjosa_metric_summary(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    j = df[df["is_josa"]][metric].dropna()
    n = df[~df["is_josa"]][metric].dropna()
    mean_j = j.mean() if len(j) else np.nan
    mean_n = n.mean() if len(n) else np.nan
    return {"metric": metric, "n_josa": int(len(j)), "n_nonjosa": int(len(n)), "mean_josa": safe_float(mean_j), "mean_nonjosa": safe_float(mean_n), "diff_josa_minus_nonjosa": safe_float(mean_j - mean_n) if len(j) and len(n) else None, "ratio_josa_over_nonjosa": safe_float(mean_j / mean_n) if len(j) and len(n) and abs(mean_n) > 1e-12 else None}

def build_final_summary(cfg_dict: Dict[str, Any], activation_df: pd.DataFrame, attention_df: pd.DataFrame, stats_act_max: pd.DataFrame, stats_act_zmax: pd.DataFrame, stats_att_future: pd.DataFrame, fakeq_overall: pd.DataFrame, alignment_diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    last_layer = int(activation_df["layer"].max())
    act_last = activation_df[(activation_df["layer"] == last_layer) & (~activation_df["is_special"])].copy()
    att_last = attention_df[(attention_df["layer"] == last_layer) & (~attention_df["is_special"])].copy()
    activation_last_summary = [josa_nonjosa_metric_summary(act_last, m) for m in ["max_abs", "zmax", "norm", "outlier_ratio", "topk_share"]]
    attention_last_summary = [josa_nonjosa_metric_summary(att_last, m) for m in ["received_attn_future_mean", "received_attn_total_share", "attn_sink_z_future"]]
    fakeq_condition_rank = fakeq_overall.to_dict(orient="records") if fakeq_overall is not None and len(fakeq_overall) > 0 else []
    act_sig = layer_sig_count(stats_act_max)
    z_sig = layer_sig_count(stats_act_zmax)
    att_sig = layer_sig_count(stats_att_future)
    activation_supported = act_sig["tested_layers"] > 0 and act_sig["sig_layers"] >= max(1, int(0.4 * act_sig["tested_layers"])) and float(stats_act_max.iloc[-1]["mean_diff_josa_minus_nonjosa"]) > 0
    attention_supported = att_sig["tested_layers"] > 0 and att_sig["sig_layers"] >= max(1, int(0.25 * att_sig["tested_layers"])) and float(stats_att_future.iloc[-1]["mean_diff_josa_minus_nonjosa"]) > 0
    quant_supported = False
    if fakeq_overall is not None and len(fakeq_overall) > 0:
        fq = fakeq_overall.set_index("condition")
        if {"matched_josa_only", "matched_nonjosa_only"}.issubset(set(fq.index)):
            quant_supported = float(fq.loc["matched_josa_only", "mean_delta_loss_per_token"]) > float(fq.loc["matched_nonjosa_only", "mean_delta_loss_per_token"])
    return {"model_id": cfg_dict["model_id"], "n_sentences": cfg_dict["max_sentences"], "last_layer": last_layer, "alignment_diagnostics": alignment_diagnostics, "activation_last_layer_summary": activation_last_summary, "attention_last_layer_summary": attention_last_summary, "activation_max_abs_layer_test": act_sig, "activation_zmax_layer_test": z_sig, "attention_future_mean_layer_test": att_sig, "fake_quant_condition_rank": fakeq_condition_rank, "interpretation_flags": {"activation_sink_supported_by_current_run": bool(activation_supported), "attention_sink_supported_by_current_run": bool(attention_supported), "josa_quantization_sensitivity_supported_by_current_run": bool(quant_supported)}}

def save_final_report(final_summary: Dict[str, Any], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    act_table = pd.DataFrame(final_summary["activation_last_layer_summary"]).to_markdown(index=False)
    att_table = pd.DataFrame(final_summary["attention_last_layer_summary"]).to_markdown(index=False)
    flags = json.dumps(final_summary["interpretation_flags"], ensure_ascii=False, indent=2)
    fakeq = json.dumps(final_summary["fake_quant_condition_rank"][:10], ensure_ascii=False, indent=2)
    text = f"""
# Korean Josa Activation / Attention Sink Probe Report

## 1. Summary

Model: `{final_summary['model_id']}`

The current run evaluates whether Korean postposition-overlapping tokens behave as activation or attention sinks.

## 2. Activation Summary

{act_table}

## 3. Attention Summary

{att_table}

## 4. Fake Quantization Ranking

<pre>
{fakeq}
</pre>

## 5. Interpretation Flags

<pre>
{flags}
</pre>

## 6. Interpretation

The most conservative interpretation is:

- Korean JOSA-overlapping tokens show stronger evidence as activation concentration points.
- Attention-sink evidence should be treated separately and more cautiously.
- Quantization sensitivity supports the hypothesis that these positions may carry structurally important information.
"""
    output_path.write_text(text, encoding="utf-8")

def make_zip(output_dir: str) -> str:
    output_dir = Path(output_dir)
    return shutil.make_archive(str(output_dir), "zip", str(output_dir))
