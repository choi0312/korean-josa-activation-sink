
# Korean Josa Activation / Attention Sink Probe Report

## 1. Experimental Setup

- Model: `Bllossom/llama-3.2-Korean-Bllossom-3B`
- Morphological analyzer: KoNLPy Okt
- Number of sentences: 180
- Max sequence length: 128
- Decoder layers: 28
- Last analyzed layer: 27
- Primary JOSA definition: an HF subword token whose character span overlaps a KoNLPy `Josa` morpheme.
- Strict JOSA sensitivity definition: JOSA overlap ratio >= 0.5.

## 2. Research Question

This experiment tests whether Korean postposition-overlapping tokens behave as activation or attention sinks inside a Korean LLM.

The operational questions are:

1. Do JOSA tokens show larger hidden-state activation peaks than NON-JOSA tokens?
2. Is the effect repeated across many decoder layers?
3. Do JOSA tokens receive disproportionately large attention mass as key positions?
4. Are JOSA token hidden states more sensitive to selective fake activation quantization?

## 3. Activation-Sink Test

Primary activation metrics:

- `max_abs`: maximum absolute hidden-state value per token.
- `zmax`: z-score of token `max_abs` using layer-wise scalar activation calibration.
- `norm`: hidden-state L2 norm.
- `outlier_ratio`: fraction of hidden channels exceeding the layer calibration threshold.
- `topk_share`: fraction of absolute activation mass concentrated in the top-k channels.

Layer-wise paired Wilcoxon result:

- `max_abs` significant layers after FDR correction: 14 / 28
- `zmax` significant layers after FDR correction: 14 / 28

Last-layer activation summary:

| metric        |   n_josa |   n_nonjosa |   mean_josa |   mean_nonjosa |   diff_josa_minus_nonjosa |   ratio_josa_over_nonjosa |
|:--------------|---------:|------------:|------------:|---------------:|--------------------------:|--------------------------:|
| max_abs       |      304 |        1652 | 22.8158     |    20.211      |                2.60481    |                  1.12888  |
| zmax          |      304 |        1652 | 17.4452     |    15.3589     |                2.08631    |                  1.13584  |
| norm          |      304 |        1652 | 88.5749     |    89.7427     |               -1.16773    |                  0.986988 |
| outlier_ratio |      304 |        1652 |  0.00587008 |     0.00459177 |                0.00127831 |                  1.27839  |
| topk_share    |      304 |        1652 |  0.11262    |     0.0887332  |                0.0238865  |                  1.26919  |

## 4. Attention-Sink Test

Primary attention metric:

- `received_attn_future_mean`: for each token as a key position, the mean attention mass received from future query tokens.
- This excludes self-attention and normalizes by the number of eligible future query positions.
- This correction is important because causal attention structurally favors earlier key positions.

Layer-wise paired Wilcoxon result:

- `received_attn_future_mean` significant layers after FDR correction: 9 / 28

Last-layer attention summary:

| metric                    |   n_josa |   n_nonjosa |   mean_josa |   mean_nonjosa |   diff_josa_minus_nonjosa |   ratio_josa_over_nonjosa |
|:--------------------------|---------:|------------:|------------:|---------------:|--------------------------:|--------------------------:|
| received_attn_future_mean |      304 |        1472 |   0.0121006 |      0.0174    |              -0.00529939  |                  0.695437 |
| received_attn_total_share |      304 |        1652 |   0.020122  |      0.0195045 |               0.000617534 |                  1.03166  |
| attn_sink_z_future        |      304 |        1472 |  -0.328123  |     -0.302077  |              -0.0260465   |                  1.08622  |

## 5. Matched Fake Activation Quantization

Perturbation design:

- A forward hook is inserted at selected decoder layers.
- Only selected token hidden states are fake-quantized.
- Fake quantization uses 4-bit symmetric per-token absmax quantization.
- JOSA tokens and NON-JOSA tokens are matched inside the same batch.
- The main comparison is `matched_josa_only` vs `matched_nonjosa_only`.

Overall fake-quantization condition ranking:

<pre>
[
  {
    "condition": "matched_josa_only",
    "n": 1480,
    "mean_delta_loss": 0.1161623669637216,
    "mean_delta_loss_per_token": 0.039362111979162985,
    "median_delta_loss_per_token": 0.017415651253291538,
    "mean_n_quantized_tokens": 4.108108108108108
  },
  {
    "condition": "matched_nonjosa_only",
    "n": 1480,
    "mean_delta_loss": 0.09931659408517786,
    "mean_delta_loss_per_token": 0.029387833663242054,
    "median_delta_loss_per_token": 0.022597031933920722,
    "mean_n_quantized_tokens": 4.108108108108108
  },
  {
    "condition": "matched_random_non_special",
    "n": 1480,
    "mean_delta_loss": 0.08744196924003395,
    "mean_delta_loss_per_token": 0.025110611390947325,
    "median_delta_loss_per_token": 0.012996298926217216,
    "mean_n_quantized_tokens": 4.108108108108108
  },
  {
    "condition": "all_non_special",
    "n": 1800,
    "mean_delta_loss": 0.4189565141995748,
    "mean_delta_loss_per_token": 0.02421337093691645,
    "median_delta_loss_per_token": 0.01595351957913601,
    "mean_n_quantized_tokens": 21.733333333333334
  }
]
</pre>

## 6. Current-Run Interpretation Flags

<pre>
{
  "activation_sink_supported_by_current_run": true,
  "attention_sink_supported_by_current_run": false,
  "josa_quantization_sensitivity_supported_by_current_run": true
}
</pre>

## 7. Output Files

The experiment saves the following result groups:

- `tables/*.csv`: token-level metrics, statistical tests, control summaries, fake-quantization results.
- `figs/*.png`: layer curves, POS boxplots, heatmaps, fake-quantization plots.
- `json/*.json`: configuration, environment, model information, final summary.
- `reports/final_report.md`: this report.
- `korean_josa_sink_results.zip`: complete compressed result package.

## 8. Caveats

1. KoNLPy Okt does not provide native character offsets, so morpheme spans are recovered by sequential string matching.
2. HF subword tokens may merge noun stems and JOSA morphemes. Therefore, the primary label means JOSA-overlapping token, not always a pure JOSA-only token.
3. Attention-sink evidence should be interpreted using the future-query-normalized metric, not raw total received attention alone.
4. Fake activation quantization is a controlled perturbation probe. It is not identical to full deployment quantization.
5. Stronger evidence would require replication across additional Korean LLMs, larger corpora, and alternative morphological analyzers.
