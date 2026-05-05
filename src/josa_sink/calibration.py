from collections import defaultdict
from typing import Any, Dict, List
import numpy as np
import torch
from tqdm.auto import tqdm
from .collate import collate_encoded_samples
from .modeling import get_model_device
from .utils import batched, clean_mem

@torch.no_grad()
def calibrate_layer_stats(model, encoded_samples: List[Dict[str, Any]], batch_size: int, pad_token_id: int, sample_per_layer_per_batch: int = 8192, quantiles: List[float] = None) -> Dict[int, Dict[str, float]]:
    if quantiles is None:
        quantiles = [99.0, 99.5, 99.9]
    model.eval()
    device = get_model_device(model)
    layer_buffers = defaultdict(list)
    for batch_samples in tqdm(list(batched(encoded_samples, batch_size)), desc="activation calibration"):
        batch = collate_encoded_samples(batch_samples, pad_token_id)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        special_mask = batch["special_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False, return_dict=True, use_cache=False)
        hidden_states = outputs.hidden_states[1:]
        valid_token_mask = attention_mask.bool() & (~special_mask)
        valid_3d = valid_token_mask.unsqueeze(-1)
        for layer_idx, hs in enumerate(hidden_states):
            abs_vals = hs.detach().float().abs()
            flat = abs_vals[valid_3d.expand_as(abs_vals)]
            if flat.numel() == 0:
                continue
            if flat.numel() > sample_per_layer_per_batch:
                perm = torch.randperm(flat.numel(), device=flat.device)[:sample_per_layer_per_batch]
                flat = flat[perm]
            layer_buffers[layer_idx].append(flat.cpu())
        del outputs, hidden_states, input_ids, attention_mask, special_mask
        clean_mem()
    layer_stats = {}
    for layer_idx in sorted(layer_buffers.keys()):
        cat = torch.cat(layer_buffers[layer_idx], dim=0).numpy()
        entry = {"scalar_abs_mean": float(np.mean(cat)), "scalar_abs_std": float(np.std(cat) + 1e-8), "scalar_abs_median": float(np.median(cat))}
        for q in quantiles:
            entry[f"scalar_abs_q{str(q).replace('.', '_')}"] = float(np.percentile(cat, q))
        layer_stats[layer_idx] = entry
    return layer_stats
