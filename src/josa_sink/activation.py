from typing import Any, Dict, List
import pandas as pd
import torch
from tqdm.auto import tqdm
from .collate import collate_encoded_samples
from .modeling import get_model_device
from .utils import batched, clean_mem

@torch.no_grad()
def collect_activation_metrics(model, encoded_samples: List[Dict[str, Any]], batch_size: int, pad_token_id: int, layer_stats: Dict[int, Dict[str, float]], topk_channels: int = 32, primary_quantile: float = 99.5) -> pd.DataFrame:
    model.eval()
    device = get_model_device(model)
    rows = []
    q_key = f"scalar_abs_q{str(primary_quantile).replace('.', '_')}"
    for batch_samples in tqdm(list(batched(encoded_samples, batch_size)), desc="collect activation metrics"):
        batch = collate_encoded_samples(batch_samples, pad_token_id)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False, return_dict=True, use_cache=False)
        hidden_states = outputs.hidden_states[1:]
        per_layer = []
        for layer_idx, hs in enumerate(hidden_states):
            hs = hs.detach().float()
            abs_hs = hs.abs()
            sum_abs = abs_hs.sum(dim=-1) + 1e-12
            norm = torch.linalg.vector_norm(hs, dim=-1)
            max_abs = abs_hs.amax(dim=-1)
            mean_abs = abs_hs.mean(dim=-1)
            top1_share = max_abs / sum_abs
            k = min(topk_channels, abs_hs.shape[-1])
            topk_share = torch.topk(abs_hs, k=k, dim=-1).values.sum(dim=-1) / sum_abs
            threshold = layer_stats[layer_idx][q_key]
            outlier_ratio = (abs_hs > threshold).float().mean(dim=-1)
            max_over_threshold = max_abs / max(threshold, 1e-8)
            zmax = (max_abs - layer_stats[layer_idx]["scalar_abs_mean"]) / max(layer_stats[layer_idx]["scalar_abs_std"], 1e-8)
            per_layer.append({"norm": norm.cpu().numpy(), "max_abs": max_abs.cpu().numpy(), "mean_abs": mean_abs.cpu().numpy(), "top1_share": top1_share.cpu().numpy(), "topk_share": topk_share.cpu().numpy(), "outlier_ratio": outlier_ratio.cpu().numpy(), "max_over_threshold": max_over_threshold.cpu().numpy(), "zmax": zmax.cpu().numpy()})
        for b_idx, sample in enumerate(batch_samples):
            valid_len = len(sample["input_ids"])
            for t_idx, meta in enumerate(sample["token_meta"]):
                for layer_idx, mats in enumerate(per_layer):
                    rows.append({"sentence_id": sample["sentence_id"], "text": sample["text"], "seq_len": valid_len, "token_idx": t_idx, "token_text": meta["token_text"], "token_display": meta["token_display"], "char_start": meta["char_start"], "char_end": meta["char_end"], "char_len": meta["char_len"], "rel_pos": meta["rel_pos"], "is_special": bool(meta["is_special"]), "best_morph_form": meta["best_morph_form"], "best_pos_tag": meta["best_pos_tag"], "major_coarse_pos": meta["major_coarse_pos"], "coarse_pos": meta["coarse_pos"], "josa_overlap_ratio": meta["josa_overlap_ratio"], "is_josa": bool(meta["is_josa"]), "is_josa_strict": bool(meta["is_josa_strict"]), "alignment_confidence": meta["alignment_confidence"], "layer": layer_idx, "norm": float(mats["norm"][b_idx, t_idx]), "max_abs": float(mats["max_abs"][b_idx, t_idx]), "mean_abs": float(mats["mean_abs"][b_idx, t_idx]), "top1_share": float(mats["top1_share"][b_idx, t_idx]), "topk_share": float(mats["topk_share"][b_idx, t_idx]), "outlier_ratio": float(mats["outlier_ratio"][b_idx, t_idx]), "max_over_threshold": float(mats["max_over_threshold"][b_idx, t_idx]), "zmax": float(mats["zmax"][b_idx, t_idx])})
        del outputs, hidden_states, input_ids, attention_mask
        clean_mem()
    df = pd.DataFrame(rows)
    for metric in ["norm", "max_abs", "zmax", "outlier_ratio", "topk_share"]:
        df[f"{metric}_rank_pct_in_sentence_layer"] = df.groupby(["sentence_id", "layer"])[metric].rank(method="average", pct=True)
    return df
