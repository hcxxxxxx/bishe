from typing import Dict, List

import torch
import torch.nn as nn

from fgemo_tts.config.schema import PromptCondition
from fgemo_tts.models.backbone_interface import TTSBackboneBase
from fgemo_tts.models.prompt_encoder import PromptEncoder


class PromptControlAdaptor(nn.Module):
    def __init__(self, cond_dim: int = 256, backbone_hidden_dim: int = 512):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, backbone_hidden_dim)
        self.to_beta = nn.Linear(cond_dim, backbone_hidden_dim)

    def forward(self, cond_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "gamma": self.to_gamma(cond_vec),
            "beta": self.to_beta(cond_vec),
        }


class PromptControlledTTS(nn.Module):
    def __init__(self, backbone: TTSBackboneBase, cond_dim: int = 256, backbone_hidden_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.prompt_encoder = PromptEncoder(out_dim=cond_dim)
        self.adaptor = PromptControlAdaptor(cond_dim=cond_dim, backbone_hidden_dim=backbone_hidden_dim)

    def forward(self, text_tokens: torch.Tensor, acoustic: torch.Tensor, conds: List[PromptCondition]) -> Dict[str, torch.Tensor]:
        cond_vec = self.prompt_encoder(conds)
        cond = self.adaptor(cond_vec)
        return self.backbone(text_tokens=text_tokens, acoustic=acoustic, cond=cond)

    @torch.no_grad()
    def infer(self, text: str, cond: PromptCondition, speaker_wav: str = "") -> torch.Tensor:
        cond_vec = self.prompt_encoder([cond])
        cond_dict = self.adaptor(cond_vec)
        return self.backbone.infer(text=text, cond=cond_dict, speaker_wav=speaker_wav)
