from typing import List

import torch
import torch.nn as nn

from fgemo_tts.config.schema import PromptCondition


EMOTION_ID = {
    "中性": 0,
    "高兴": 1,
    "开心": 1,
    "悲伤": 2,
    "难过": 2,
    "温柔": 3,
    "平静": 4,
    "愤怒": 5,
    "严肃": 6,
    "紧张": 7,
    "兴奋": 8,
}

STYLE_ID = {
    "自然": 0,
    "温柔": 1,
    "克制": 2,
    "坚定": 3,
    "轻快": 4,
    "低沉": 5,
    "柔和": 6,
}


class PromptEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256, out_dim: int = 256):
        super().__init__()
        self.emotion_emb = nn.Embedding(9, hidden_size // 2)
        self.style_emb = nn.Embedding(7, hidden_size // 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + 3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, conds: List[PromptCondition]) -> torch.Tensor:
        device = self.emotion_emb.weight.device
        eids = torch.tensor([EMOTION_ID.get(c.emotion, 0) for c in conds], dtype=torch.long, device=device)
        sids = torch.tensor([STYLE_ID.get(c.style, 0) for c in conds], dtype=torch.long, device=device)
        scalars = torch.tensor(
            [[c.intensity, c.arousal, c.valence] for c in conds],
            dtype=torch.float32,
            device=device,
        )

        e = self.emotion_emb(eids)
        s = self.style_emb(sids)
        x = torch.cat([e, s, scalars], dim=-1)
        return self.mlp(x)
