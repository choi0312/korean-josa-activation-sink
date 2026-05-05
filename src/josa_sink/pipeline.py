import platform
import pandas as pd
import torch
from .activation import collect_activation_metrics
from .attention import collect_attention_sink_metrics
from .calibration import calibrate_layer_stats
from .config import ExperimentConfig, prepare_output_dirs
from .corpus import load_or_build_corpus
from .modeling import load_tokenizer_and_model, get_decoder_layers, get_model_device
from .morph_align import align_corpus
from .quant_probe import run_matched_fake_quant_probe
from .reporting import build_final_summary, save_final_report, make_zip
from .stats import run_layerwise_paired_test
from .utils import save_json, seed_everything
from .visualization import plot_layer_diff_curve, plot_last_layer_box, plot_fakeq_summary

def run_pipeline(cfg: ExperimentConfig):
    seed_everything(cfg.seed)
    dirs = prepare_output_dirs(cfg)
    env_info = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "config": cfg.to_dict()}
    save_json(env_info, dirs["json"] / "environment_and_config.json")
    tokenizer, model = load_tokenizer_and_model(cfg)
    decoder_layers = get_decoder_layers(model)
    model_info = {"model_id": cfg.model_id, "tokenizer_fast": tokenizer.is_fast, "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id, "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id, "num_decoder_layers": len(decoder_layers), "hidden_size": getattr(model.config, "hidden_size", None), "num_attention_heads": getattr(model.config, "num_attention_heads", None), "device": str(get_model_device(model))}
    save_json(model_info, dirs["json"] / "model_info.json")
    corpus = load_or_build_corpus(cfg.optional_corpus_txt, cfg.max_sentences)
    pd.DataFrame({"sentence_id": range(len(corpus)), "text": corpus}).to_csv(dirs["tables"] / "corpus.csv", index=False, encoding="utf-8-sig")
    encoded_samples, morph_df, align_df = align_corpus(corpus=corpus, tokenizer=tokenizer, max_length=cfg.max_length, strict_josa_ratio=cfg.strict_josa_ratio)
    morph_df.to_csv(dirs["tables"] / "morph_df_konlpy_okt.csv", index=False, encoding="utf-8-sig")
    align_df.to_csv(dirs["tables"] / "token_morph_alignment.csv", index=False, encoding="utf-8-sig")
    alignment_diagnostics = {"n_sentences": len(encoded_samples), "n_tokens_total": int(len(align_df)), "n_non_special_tokens": int((~align_df["is_special"]).sum()), "n_josa_tokens_any_overlap": int(align_df["is_josa"].sum()), "n_josa_tokens_strict": int(align_df["is_josa_strict"].sum()), "n_alignment_no_overlap": int((align_df["alignment_confidence"] == "NO_MORPH_OVERLAP").sum())}
    save_json(alignment_diagnostics, dirs["json"] / "alignment_diagnostics.json")
    layer_stats = calibrate_layer_stats(model=model, encoded_samples=encoded_samples, batch_size=cfg.batch_size_activation, pad_token_id=tokenizer.pad_token_id, sample_per_layer_per_batch=cfg.calibration_sample_per_layer_per_batch, quantiles=cfg.activation_quantiles)
    pd.DataFrame(layer_stats).T.reset_index().rename(columns={"index": "layer"}).to_csv(dirs["tables"] / "layer_activation_calibration_stats.csv", index=False, encoding="utf-8-sig")
    save_json(layer_stats, dirs["json"] / "layer_activation_calibration_stats.json")
    activation_df = collect_activation_metrics(model=model, encoded_samples=encoded_samples, batch_size=cfg.batch_size_activation, pad_token_id=tokenizer.pad_token_id, layer_stats=layer_stats, topk_channels=cfg.topk_channels, primary_quantile=cfg.primary_activation_quantile)
    activation_df.to_csv(dirs["tables"] / "activation_token_metrics.csv", index=False, encoding="utf-8-sig")
    attention_df = collect_attention_sink_metrics(model=model, encoded_samples=encoded_samples, batch_size=cfg.batch_size_attention, pad_token_id=tokenizer.pad_token_id)
    attention_df.to_csv(dirs["tables"] / "attention_token_metrics.csv", index=False, encoding="utf-8-sig")
    stats_act_max = run_layerwise_paired_test(activation_df, "max_abs", min_pairs=cfg.min_sentence_pairs_per_layer)
    stats_act_zmax = run_layerwise_paired_test(activation_df, "zmax", min_pairs=cfg.min_sentence_pairs_per_layer)
    stats_att_future = run_layerwise_paired_test(attention_df, "received_attn_future_mean", min_pairs=cfg.min_sentence_pairs_per_layer)
    stats_act_max.to_csv(dirs["tables"] / "stats_activation_max_abs.csv", index=False, encoding="utf-8-sig")
    stats_act_zmax.to_csv(dirs["tables"] / "stats_activation_zmax.csv", index=False, encoding="utf-8-sig")
    stats_att_future.to_csv(dirs["tables"] / "stats_attention_future_mean.csv", index=False, encoding="utf-8-sig")
    plot_layer_diff_curve(stats_act_max, "Layer-wise paired difference: activation max_abs", dirs["figs"] / "layerwise_josa_minus_nonjosa_activation_max_abs.png", dpi=cfg.save_dpi)
    plot_layer_diff_curve(stats_act_zmax, "Layer-wise paired difference: activation zmax", dirs["figs"] / "layerwise_josa_minus_nonjosa_activation_zmax.png", dpi=cfg.save_dpi)
    plot_layer_diff_curve(stats_att_future, "Layer-wise paired difference: future-normalized received attention", dirs["figs"] / "layerwise_josa_minus_nonjosa_attention_future_mean.png", dpi=cfg.save_dpi)
    plot_last_layer_box(activation_df, "max_abs", dirs["figs"] / "last_layer_activation_max_abs_boxplot.png", "Activation max_abs distribution by POS", dpi=cfg.save_dpi)
    plot_last_layer_box(attention_df, "received_attn_future_mean", dirs["figs"] / "last_layer_attention_future_mean_boxplot.png", "Future-normalized received attention by POS", dpi=cfg.save_dpi)
    fakeq_df = run_matched_fake_quant_probe(model=model, encoded_samples=encoded_samples, pad_token_id=tokenizer.pad_token_id, batch_size=cfg.batch_size_quant, num_bits=cfg.fake_quant_bits, probe_layers=cfg.fake_quant_probe_layers, match_trials=cfg.fake_quant_match_trials, base_seed=cfg.seed, matched_by_position_and_length=cfg.matched_by_position_and_length)
    fakeq_df.to_csv(dirs["tables"] / "matched_fake_activation_quant_probe_raw.csv", index=False, encoding="utf-8-sig")
    fakeq_summary = fakeq_df.dropna(subset=["delta_loss", "delta_loss_per_quantized_token"]).groupby(["layer", "condition"]).agg(n=("delta_loss", "size"), mean_delta_loss=("delta_loss", "mean"), sem_delta_loss=("delta_loss", lambda x: x.std(ddof=1) / max(len(x), 1) ** 0.5), mean_delta_loss_per_token=("delta_loss_per_quantized_token", "mean"), sem_delta_loss_per_token=("delta_loss_per_quantized_token", lambda x: x.std(ddof=1) / max(len(x), 1) ** 0.5), mean_n_quantized_tokens=("n_quantized_tokens", "mean")).reset_index()
    fakeq_summary.to_csv(dirs["tables"] / "matched_fake_activation_quant_probe_summary.csv", index=False, encoding="utf-8-sig")
    fakeq_overall = fakeq_df.dropna(subset=["delta_loss", "delta_loss_per_quantized_token"]).groupby("condition").agg(n=("delta_loss", "size"), mean_delta_loss=("delta_loss", "mean"), mean_delta_loss_per_token=("delta_loss_per_quantized_token", "mean"), median_delta_loss_per_token=("delta_loss_per_quantized_token", "median"), mean_n_quantized_tokens=("n_quantized_tokens", "mean")).sort_values("mean_delta_loss_per_token", ascending=False).reset_index()
    fakeq_overall.to_csv(dirs["tables"] / "matched_fake_activation_quant_probe_overall.csv", index=False, encoding="utf-8-sig")
    plot_fakeq_summary(fakeq_summary, "mean_delta_loss", dirs["figs"] / "fake_quant_delta_loss_by_layer.png", "Matched fake activation quantization: delta loss", dpi=cfg.save_dpi)
    plot_fakeq_summary(fakeq_summary, "mean_delta_loss_per_token", dirs["figs"] / "fake_quant_delta_loss_per_token_by_layer.png", "Matched fake activation quantization: delta loss per quantized token", dpi=cfg.save_dpi)
    final_summary = build_final_summary(cfg_dict=cfg.to_dict(), activation_df=activation_df, attention_df=attention_df, stats_act_max=stats_act_max, stats_act_zmax=stats_act_zmax, stats_att_future=stats_att_future, fakeq_overall=fakeq_overall, alignment_diagnostics=alignment_diagnostics)
    save_json(final_summary, dirs["json"] / "final_summary.json")
    save_final_report(final_summary, dirs["reports"] / "final_report.md")
    zip_path = make_zip(cfg.output_dir)
    print("[DONE] ZIP:", zip_path)
    return final_summary
