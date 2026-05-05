from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from .collate import collate_encoded_samples
from .modeling import get_decoder_layers, get_model_device
from .utils import batched, clean_mem

def fake_quantize_per_token_absmax(x: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    qmax = (2 ** (num_bits - 1)) - 1
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    xq = torch.round(xf / scale).clamp(-qmax, qmax) * scale
    return xq.to(orig_dtype)

def make_selective_quant_hook(token_mask: torch.Tensor, num_bits: int = 4):
    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = ()
        local_mask = token_mask.to(hs.device).unsqueeze(-1)
        qhs = fake_quantize_per_token_absmax(hs, num_bits=num_bits)
        mixed = torch.where(local_mask, qhs, hs)
        if isinstance(output, tuple):
            return (mixed, *rest)
        return mixed
    return _hook

@torch.no_grad()
def evaluate_batch_loss(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if labels.shape[1] > 0:
        labels[:, 0] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=False, output_attentions=False, return_dict=True, use_cache=False)
    return float(outputs.loss.item())

def _pos_bin_from_rel(rel: float) -> int:
    if rel < 0.25: return 0
    if rel < 0.50: return 1
    if rel < 0.75: return 2
    return 3

def _len_bin_from_char_len(n: int) -> int:
    if n <= 1: return 0
    if n <= 2: return 1
    if n <= 4: return 2
    return 3

def build_matched_masks_from_batch(batch: Dict[str, Any], seed: int = 42, matched_by_position_and_length: bool = True):
    rng = np.random.default_rng(seed)
    josa_mask = batch["josa_mask"].clone()
    special_mask = batch["special_mask"].clone()
    attention_mask = batch["attention_mask"].bool()
    non_special = (~special_mask) & attention_mask
    only_josa = josa_mask & non_special
    non_josa = (~josa_mask) & non_special
    matched_josa = torch.zeros_like(josa_mask)
    matched_nonjosa = torch.zeros_like(josa_mask)
    bsz, _ = josa_mask.shape
    for b in range(bsz):
        j_idx_all = torch.where(only_josa[b])[0].cpu().numpy().tolist()
        n_idx_all = torch.where(non_josa[b])[0].cpu().numpy().tolist()
        if len(j_idx_all) == 0 or len(n_idx_all) == 0:
            continue
        rng.shuffle(j_idx_all)
        used_nonjosa, selected_j, selected_n = set(), [], []
        meta = batch["meta_batch"][b]
        for jidx in j_idx_all:
            candidates = [n for n in n_idx_all if n not in used_nonjosa]
            if len(candidates) == 0:
                break
            if matched_by_position_and_length:
                j_pbin = _pos_bin_from_rel(meta[jidx]["rel_pos"])
                j_lbin = _len_bin_from_char_len(meta[jidx]["char_len"])
                same_both = [n for n in candidates if _pos_bin_from_rel(meta[n]["rel_pos"]) == j_pbin and _len_bin_from_char_len(meta[n]["char_len"]) == j_lbin]
                same_pos = [n for n in candidates if _pos_bin_from_rel(meta[n]["rel_pos"]) == j_pbin]
                pool = same_both if same_both else same_pos if same_pos else candidates
            else:
                pool = candidates
            chosen_n = int(rng.choice(pool))
            used_nonjosa.add(chosen_n)
            selected_j.append(jidx)
            selected_n.append(chosen_n)
        for j in selected_j:
            matched_josa[b, j] = True
        for n in selected_n:
            matched_nonjosa[b, n] = True
    return matched_josa, matched_nonjosa

@torch.no_grad()
def run_matched_fake_quant_probe(model, encoded_samples: List[Dict[str, Any]], pad_token_id: int, batch_size: int = 2, num_bits: int = 4, probe_layers: Any = "auto", match_trials: int = 5, base_seed: int = 42, matched_by_position_and_length: bool = True) -> pd.DataFrame:
    model.eval()
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    device = get_model_device(model)
    if probe_layers == "auto":
        probe_layers = sorted({n_layers // 2, (3 * n_layers) // 4, n_layers - 1})
    else:
        probe_layers = sorted([int(x) for x in probe_layers if 0 <= int(x) < n_layers])
    results = []
    for batch_idx, batch_samples in enumerate(tqdm(list(batched(encoded_samples, batch_size)), desc="fake quant probe")):
        batch = collate_encoded_samples(batch_samples, pad_token_id)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        special_mask = batch["special_mask"].to(device)
        baseline_loss = evaluate_batch_loss(model, input_ids, attention_mask)
        all_non_special = ((~special_mask) & attention_mask.bool()).to(device)
        for trial in range(match_trials):
            matched_josa, matched_nonjosa = build_matched_masks_from_batch(batch=batch, seed=base_seed + batch_idx * 1000 + trial, matched_by_position_and_length=matched_by_position_and_length)
            matched_josa = matched_josa.to(device)
            matched_nonjosa = matched_nonjosa.to(device)
            rng = np.random.default_rng(base_seed + 777 + batch_idx * 1000 + trial)
            random_mask = torch.zeros_like(matched_josa)
            for b in range(random_mask.shape[0]):
                k = int(matched_josa[b].sum().item())
                candidates = torch.where(all_non_special[b].detach().cpu())[0].numpy()
                if k > 0 and len(candidates) >= k:
                    chosen = rng.choice(candidates, size=k, replace=False)
                    random_mask[b, chosen] = True
            random_mask = random_mask.to(device)
            condition_masks = {"matched_josa_only": matched_josa, "matched_nonjosa_only": matched_nonjosa, "matched_random_non_special": random_mask, "all_non_special": all_non_special}
            for layer_idx in probe_layers:
                for cond_name, token_mask in condition_masks.items():
                    n_quant = int(token_mask.sum().item())
                    if n_quant == 0:
                        results.append({"batch_idx": batch_idx, "trial": trial, "layer": layer_idx, "condition": cond_name, "baseline_loss": baseline_loss, "quantized_loss": np.nan, "delta_loss": np.nan, "n_quantized_tokens": 0, "delta_loss_per_quantized_token": np.nan, "n_sentences_in_batch": len(batch_samples)})
                        continue
                    handle = layers[layer_idx].register_forward_hook(make_selective_quant_hook(token_mask=token_mask, num_bits=num_bits))
                    try:
                        q_loss = evaluate_batch_loss(model, input_ids, attention_mask)
                    finally:
                        handle.remove()
                    delta = q_loss - baseline_loss
                    results.append({"batch_idx": batch_idx, "trial": trial, "layer": layer_idx, "condition": cond_name, "baseline_loss": baseline_loss, "quantized_loss": q_loss, "delta_loss": delta, "n_quantized_tokens": n_quant, "delta_loss_per_quantized_token": delta / max(n_quant, 1), "n_sentences_in_batch": len(batch_samples)})
        del input_ids, attention_mask, special_mask
        clean_mem()
    return pd.DataFrame(results)
