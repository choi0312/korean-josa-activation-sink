from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Union
import json
import yaml

@dataclass
class ExperimentConfig:
    model_id: str = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    seed: int = 42
    device_dtype: str = "bf16"
    attn_implementation: str = "eager"
    trust_remote_code: bool = False
    max_length: int = 128
    max_sentences: int = 180
    optional_corpus_txt: str = "/content/korean_josa_corpus.txt"
    strict_josa_ratio: float = 0.5
    batch_size_activation: int = 2
    batch_size_attention: int = 1
    batch_size_quant: int = 2
    calibration_sample_per_layer_per_batch: int = 8192
    activation_quantiles: List[float] = None
    primary_activation_quantile: float = 99.5
    topk_channels: int = 32
    min_sentence_pairs_per_layer: int = 8
    num_permutations: int = 300
    alpha: float = 0.05
    fake_quant_bits: int = 4
    fake_quant_match_trials: int = 5
    fake_quant_probe_layers: Union[str, List[int]] = "auto"
    matched_by_position_and_length: bool = True
    num_heatmap_sentences: int = 6
    local_window: int = 3
    save_dpi: int = 160
    output_dir: str = "/content/korean_josa_sink_results"

    def __post_init__(self):
        if self.activation_quantiles is None:
            self.activation_quantiles = [99.0, 99.5, 99.9]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

def load_config(path: str = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return ExperimentConfig(**data)

def prepare_output_dirs(cfg: ExperimentConfig) -> Dict[str, Path]:
    root = Path(cfg.output_dir)
    dirs = {
        "root": root,
        "figs": root / "figs",
        "tables": root / "tables",
        "json": root / "json",
        "reports": root / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
