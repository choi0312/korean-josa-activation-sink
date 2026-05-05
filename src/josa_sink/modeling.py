from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import ExperimentConfig

def get_torch_dtype(cfg: ExperimentConfig):
    if not torch.cuda.is_available():
        return torch.float32
    if cfg.device_dtype.lower() == "bf16":
        return torch.bfloat16
    if cfg.device_dtype.lower() == "fp16":
        return torch.float16
    return torch.float32

def get_model_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device

def get_decoder_layers(model) -> nn.ModuleList:
    candidate_paths = [["model", "layers"], ["transformer", "h"], ["gpt_neox", "layers"], ["model", "decoder", "layers"]]
    for path in candidate_paths:
        cur = model
        ok = True
        for name in path:
            if hasattr(cur, name):
                cur = getattr(cur, name)
            else:
                ok = False
                break
        if ok and isinstance(cur, (nn.ModuleList, list)):
            return cur
    raise ValueError("Decoder layer path was not found. Inspect model architecture manually.")

def load_tokenizer_and_model(cfg: ExperimentConfig) -> Tuple[object, object]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, trust_remote_code=cfg.trust_remote_code)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Fast tokenizer is required because return_offsets_mapping=True is used.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(torch_dtype=get_torch_dtype(cfg), device_map="auto", low_cpu_mem_usage=True, trust_remote_code=cfg.trust_remote_code)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, attn_implementation=cfg.attn_implementation, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return tokenizer, model
