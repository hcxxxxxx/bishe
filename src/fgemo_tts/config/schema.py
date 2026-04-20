from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptCondition:
    emotion: str
    intensity: float
    arousal: float
    valence: float
    style: str


@dataclass
class TrainConfig:
    train_manifest: str
    val_manifest: str
    batch_size: int = 12
    num_workers: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_steps: int = 20000
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 1000
    output_dir: str = "exp/fgemo"
    ablation: str = "full"  # none | rule_only | full
    seed: int = 42
    pretrained_backbone_ckpt: Optional[str] = None
