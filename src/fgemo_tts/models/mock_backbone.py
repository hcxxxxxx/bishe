from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgemo_tts.models.backbone_interface import TTSBackboneBase


class MockTTSBackbone(TTSBackboneBase):
    """
    Minimal backbone for quick sanity tests.
    It predicts mel-like frames to verify prompt control training loop.
    """

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 256, mel_dim: int = 80):
        super().__init__()
        self.txt_emb = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, mel_dim)

    def forward(self, text_tokens: torch.Tensor, acoustic: torch.Tensor, cond: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.txt_emb(text_tokens)
        x, _ = self.encoder(x)

        # cond["gamma"], cond["beta"]: [B, H*2], project-free FiLM on sequence mean
        gamma = cond["gamma"].unsqueeze(1)
        beta = cond["beta"].unsqueeze(1)
        x = x * (1.0 + gamma) + beta

        y = self.proj(x)
        T = min(y.size(1), acoustic.size(1))
        loss = F.l1_loss(y[:, :T], acoustic[:, :T])
        return {"loss": loss, "mel_pred": y}

    def infer(self, text: str, cond: Dict[str, torch.Tensor], speaker_wav: str = "") -> torch.Tensor:
        # Dummy waveform-like tensor for interface completeness.
        T = 16000 * 2
        wav = torch.randn(T, device=cond["gamma"].device) * 0.01
        wav += cond["gamma"].mean() * 0.001
        return wav
