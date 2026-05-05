from typing import Any, Dict, List
import torch

def collate_encoded_samples(samples: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in samples)
    bsz = len(samples)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    josa_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    josa_strict_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    special_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    meta_batch, sentence_ids, texts = [], [], []
    for i, sample in enumerate(samples):
        ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attn = torch.tensor(sample["attention_mask"], dtype=torch.long)
        input_ids[i, :len(ids)] = ids
        attention_mask[i, :len(attn)] = attn
        meta_list = sample["token_meta"]
        meta_batch.append(meta_list)
        sentence_ids.append(sample["sentence_id"])
        texts.append(sample["text"])
        for t, meta in enumerate(meta_list):
            josa_mask[i, t] = bool(meta["is_josa"])
            josa_strict_mask[i, t] = bool(meta["is_josa_strict"])
            special_mask[i, t] = bool(meta["is_special"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "josa_mask": josa_mask, "josa_strict_mask": josa_strict_mask, "special_mask": special_mask, "meta_batch": meta_batch, "sentence_ids": sentence_ids, "texts": texts}
