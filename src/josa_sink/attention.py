from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from .collate import collate_encoded_samples
from .modeling import get_model_device
from .utils import batched, clean_mem

@torch.no_grad()
def collect_attention_sink_metrics(model, encoded_samples: List[Dict[str, Any]], batch_size: int, pad_token_id: int) -> pd.DataFrame:
    model.eval()
    device = get_model_device(model)
    rows = []
    for batch_samples in tqdm(list(batched(encoded_samples, batch_size)), desc="collect attention metrics"):
        batch = collate_encoded_samples(batch_samples, pad_token_id)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=True, return_dict=True, use_cache=False)
        if outputs.attentions is None:
            raise RuntimeError("outputs.attentions is None. Reload model with attn_implementation='eager'.")
        attentions = outputs.attentions
        for b_idx, sample in enumerate(batch_samples):
            valid_len = len(sample["input_ids"])
            q_idx = torch.arange(valid_len, device=device).view(valid_len, 1)
            k_idx = torch.arange(valid_len, device=device).view(1, valid_len)
            future_mask = (q_idx > k_idx).float()
            causal_mask = (q_idx >= k_idx).float()
            future_query_count = future_mask.sum(dim=0).cpu().numpy()
            causal_query_count = causal_mask.sum(dim=0).cpu().numpy()
            for layer_idx in range(len(attentions)):
                attn = attentions[layer_idx][b_idx, :, :valid_len, :valid_len].detach().float()
                n_heads = attn.shape[0]
                future_mass_per_key = (attn * future_mask.unsqueeze(0)).sum(dim=(0, 1))
                causal_mass_per_key = (attn * causal_mask.unsqueeze(0)).sum(dim=(0, 1))
                total_mass_per_key = attn.sum(dim=(0, 1))
                denom_future = torch.clamp(future_mask.sum(dim=0) * n_heads, min=1.0)
                denom_causal = torch.clamp(causal_mask.sum(dim=0) * n_heads, min=1.0)
                denom_total = max(float(valid_len * n_heads), 1.0)
                future_mean = future_mass_per_key / denom_future
                causal_mean = causal_mass_per_key / denom_causal
                total_share = total_mass_per_key / denom_total
                future_mass_per_head_key = (attn * future_mask.unsqueeze(0)).sum(dim=1)
                denom_future_head = torch.clamp(future_mask.sum(dim=0), min=1.0)
                future_mean_per_head_key = future_mass_per_head_key / denom_future_head.unsqueeze(0)
                max_head_future_mean = future_mean_per_head_key.max(dim=0).values
                for t_idx, meta in enumerate(sample["token_meta"]):
                    fq = int(future_query_count[t_idx])
                    rows.append({"sentence_id": sample["sentence_id"], "text": sample["text"], "seq_len": valid_len, "token_idx": t_idx, "token_text": meta["token_text"], "token_display": meta["token_display"], "char_start": meta["char_start"], "char_end": meta["char_end"], "char_len": meta["char_len"], "rel_pos": meta["rel_pos"], "is_special": bool(meta["is_special"]), "best_morph_form": meta["best_morph_form"], "best_pos_tag": meta["best_pos_tag"], "major_coarse_pos": meta["major_coarse_pos"], "coarse_pos": meta["coarse_pos"], "josa_overlap_ratio": meta["josa_overlap_ratio"], "is_josa": bool(meta["is_josa"]), "is_josa_strict": bool(meta["is_josa_strict"]), "alignment_confidence": meta["alignment_confidence"], "layer": layer_idx, "future_query_count": fq, "causal_query_count": int(causal_query_count[t_idx]), "received_attn_future_mean": float(future_mean[t_idx].item()) if fq > 0 else np.nan, "received_attn_causal_mean": float(causal_mean[t_idx].item()), "received_attn_total_share": float(total_share[t_idx].item()), "max_head_received_attn_future_mean": float(max_head_future_mean[t_idx].item()) if fq > 0 else np.nan})
        del outputs, attentions, input_ids, attention_mask
        clean_mem()
    df = pd.DataFrame(rows)
    for metric in ["received_attn_future_mean", "received_attn_causal_mean", "received_attn_total_share", "max_head_received_attn_future_mean"]:
        df[f"{metric}_rank_pct_in_sentence_layer"] = df.groupby(["sentence_id", "layer"])[metric].rank(method="average", pct=True)
    def zscore_series(x):
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)
    df["attn_sink_z_future"] = df.groupby(["sentence_id", "layer"])["received_attn_future_mean"].transform(zscore_series)
    return df
