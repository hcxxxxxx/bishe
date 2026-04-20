from typing import List

import torch
import torch.nn as nn

from fgemo_tts.config.schema import PromptCondition


class RuleConditionEncoder(nn.Module):
    """Deterministic non-trainable condition vector for ablation (rule_only)."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, conds: List[PromptCondition]) -> torch.Tensor:
        arr = []
        for c in conds:
            # [intensity, arousal, valence] repeated to out_dim
            base = torch.tensor([c.intensity, c.arousal, c.valence], dtype=torch.float32)
            rep = self.out_dim // 3
            rem = self.out_dim % 3
            vec = base.repeat(rep)
            if rem:
                vec = torch.cat([vec, base[:rem]], dim=0)
            arr.append(vec)
        return torch.stack(arr, dim=0)
