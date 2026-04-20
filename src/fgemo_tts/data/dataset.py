import json
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset

from fgemo_tts.config.schema import PromptCondition


@dataclass
class Item:
    text: str
    prompt: str
    emotion: str
    intensity: float
    arousal: float
    valence: float
    style: str
    wav_path: str


class JsonlPromptTTSDataset(Dataset):
    """
    Expected jsonl line format:
    {
      "text": "...",
      "prompt": "用略带悲伤但温柔的语气说...",
      "emotion": "悲伤",
      "intensity": 0.35,
      "arousal": -0.35,
      "valence": -0.75,
      "style": "温柔",
      "wav_path": "/abs/path.wav"
    }
    """

    def __init__(self, manifest_path: str, max_text_len: int = 256, mel_len: int = 200, mel_dim: int = 80):
        self.items: List[Item] = []
        self.max_text_len = max_text_len
        self.mel_len = mel_len
        self.mel_dim = mel_dim

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(Item(**obj))

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _simple_tokenize(text: str, max_len: int) -> torch.Tensor:
        ids = [min(ord(c), 4095) for c in text][:max_len]
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        text_tokens = self._simple_tokenize(item.text, self.max_text_len)

        # Placeholder acoustic target. Replace by real mel extraction in your backbone pipeline.
        acoustic = torch.randn(self.mel_len, self.mel_dim, dtype=torch.float32)

        cond = PromptCondition(
            emotion=item.emotion,
            intensity=float(item.intensity),
            arousal=float(item.arousal),
            valence=float(item.valence),
            style=item.style,
        )
        return text_tokens, acoustic, cond, item.prompt


def collate_fn(batch):
    text_tokens, acoustics, conds, prompts = zip(*batch)

    max_t = max(x.size(0) for x in text_tokens)
    padded_tokens = []
    for x in text_tokens:
        if x.size(0) < max_t:
            pad = torch.zeros(max_t - x.size(0), dtype=torch.long)
            x = torch.cat([x, pad], dim=0)
        padded_tokens.append(x)

    padded_tokens = torch.stack(padded_tokens, dim=0)
    acoustics = torch.stack(acoustics, dim=0)
    return padded_tokens, acoustics, list(conds), list(prompts)
