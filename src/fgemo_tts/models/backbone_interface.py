from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class TTSBackboneBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, text_tokens: torch.Tensor, acoustic: torch.Tensor, cond: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def infer(self, text: str, cond: Dict[str, torch.Tensor], speaker_wav: str = "") -> torch.Tensor:
        raise NotImplementedError
