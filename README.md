# Korean Josa Activation Sink Analysis

> [Research] Quantitative analysis of Korean postposition-focused activation sink and quantization sensitivity in Korean LLMs

This repository contains a modular research pipeline for testing whether Korean postposition-overlapping tokens behave as activation sinks, attention sinks, or quantization-sensitive structural carrier positions in Korean large language models.

## Target Model

- `Bllossom/llama-3.2-Korean-Bllossom-3B`

The pipeline was designed for Google Colab A100, but the package structure can also be used in a local CUDA environment.

## Research Question

Do Korean postposition-overlapping tokens, detected using KoNLPy-based morphological analysis, act as internal structural carrier positions inside a Korean LLM?

The project tests three hypotheses:

1. Korean postposition-overlapping tokens show larger hidden-state activation peaks than non-postposition tokens.
2. This effect appears repeatedly across many transformer layers.
3. Postposition-overlapping tokens are more sensitive to selective fake activation quantization.

## Main Metrics

### Activation sink metrics

- `max_abs`: maximum absolute hidden-state value per token.
- `zmax`: layer-normalized outlier score.
- `norm`: L2 norm of hidden state.
- `outlier_ratio`: fraction of hidden channels exceeding a calibrated threshold.
- `topk_share`: concentration of activation mass in top-k channels.

### Attention sink metrics

- `received_attn_future_mean`: attention mass received by a token as a key position from future query tokens.
- `attn_sink_z_future`: sentence-layer normalized attention sink score.

### Quantization sensitivity

- Selective fake activation quantization using forward hooks.
- Matched comparison between `matched_josa_only` and `matched_nonjosa_only`.
- Primary output: `delta_loss_per_quantized_token`.

## Repository Structure

- `configs/default.yaml`: default experiment configuration.
- `scripts/colab_setup.sh`: Colab dependency setup.
- `scripts/run_full_experiment.py`: full pipeline entry point.
- `src/josa_sink/`: modular Python package.
- `assets/figures/`: selected result figures.
- `assets/tables/`: selected summary tables.
- `assets/reports/`: selected reports and summaries.

## Colab Usage

Run the following commands in Colab:

    bash scripts/colab_setup.sh
    pip install -e .
    python scripts/run_full_experiment.py --config configs/default.yaml

Alternatively:

    josa-sink-run --config configs/default.yaml

## Expected Outputs

The pipeline saves outputs under `output_dir` defined in `configs/default.yaml`:

- `tables/*.csv`
- `figs/*.png`
- `json/*.json`
- `reports/final_report.md`
- final zip archive

## Current Interpretation Summary

The current experimental pattern supports the interpretation that Korean postposition-overlapping tokens are more plausibly sparse activation sinks or quantization-sensitive structural carrier positions than ordinary attention sinks.

## License

MIT License


---

## Reproduced Colab Result Snapshot

This section was automatically generated from the latest Colab run.

### Interpretation Flags

| Hypothesis | Result |
|---|---:|
| Activation sink supported | `True` |
| Attention sink supported | `False` |
| JOSA quantization sensitivity supported | `True` |

### Layer-wise Significance

| Metric | Significant layers | Tested layers |
|---|---:|---:|
| Activation `max_abs` | 14 | 28 |
| Activation `zmax` | 14 | 28 |
| Attention `received_attn_future_mean` | 9 | 28 |

### Conservative Interpretation

The current run supports the claim that Korean postposition-overlapping tokens behave more like activation sinks or quantization-sensitive structural carrier positions than ordinary attention sinks.
