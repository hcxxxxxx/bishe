"""Model download and loading utilities for CosyVoice2-0.5B.

This module is intentionally defensive because CosyVoice APIs may vary slightly
across versions. It tries several import/inference paths and reports clear errors.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from huggingface_hub import snapshot_download


DEFAULT_HF_REPO = "FunAudioLLM/CosyVoice2-0.5B"


@dataclass
class ModelConfig:
    """Runtime config for loading CosyVoice/CosyVoice2."""

    hf_repo_id: str = DEFAULT_HF_REPO
    local_dir: str = "./models/CosyVoice2-0.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    load_jit: bool = False
    load_trt: bool = False


class CosyVoiceModelLoader:
    """Download, load and run inference on CosyVoice2 model."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self.model = None
        self.model_dir = None

    def download_model(self, force_download: bool = False) -> str:
        """Download model files from Hugging Face and return local path."""
        local_dir = Path(self.config.local_dir).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)

        # If already populated and not forcing, keep local cache.
        has_any_file = any(local_dir.iterdir())
        if has_any_file and not force_download:
            self.model_dir = str(local_dir)
            return self.model_dir

        self.model_dir = snapshot_download(
            repo_id=self.config.hf_repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=force_download,
        )
        return self.model_dir

    def load_model(self, force_download: bool = False) -> Any:
        """Load CosyVoice2 model with best-effort API compatibility."""
        if self.model is not None:
            return self.model

        model_dir = self.download_model(force_download=force_download)

        import_errors = []

        # Try canonical import path first.
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2

            self.model = CosyVoice2(
                model_dir,
                load_jit=self.config.load_jit,
                load_trt=self.config.load_trt,
                fp16=self.config.fp16,
            )
            return self.model
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"cosyvoice.cli.cosyvoice.CosyVoice2 failed: {exc}")

        # Fallback 1: older class name.
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice

            self.model = CosyVoice(model_dir)
            return self.model
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"cosyvoice.cli.cosyvoice.CosyVoice failed: {exc}")

        # Fallback 2: top-level export if package changed.
        try:
            from cosyvoice import CosyVoice2  # type: ignore

            self.model = CosyVoice2(model_dir)
            return self.model
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"cosyvoice.CosyVoice2 failed: {exc}")

        msg = "\n".join(import_errors)
        raise RuntimeError(
            "Unable to load CosyVoice model. Checked multiple API paths.\n"
            f"Model dir: {model_dir}\n"
            f"Errors:\n{msg}"
        )

    @staticmethod
    def _materialize_output(output: Any) -> List[Dict[str, Any]]:
        """Normalize different CosyVoice output shapes into list[dict]."""
        if output is None:
            return []
        if isinstance(output, dict):
            return [output]
        if isinstance(output, list):
            if output and isinstance(output[0], dict):
                return output
            return [{"tts_speech": item} for item in output]
        if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
            items = list(output)
            if items and isinstance(items[0], dict):
                return items
            return [{"tts_speech": item} for item in items]
        return [{"tts_speech": output}]

    @staticmethod
    def _pick_wave_tensor(sample: Dict[str, Any]) -> torch.Tensor:
        """Extract waveform tensor from one inference sample."""
        keys_priority = ["tts_speech", "speech", "audio", "wav", "waveform"]
        for key in keys_priority:
            if key in sample:
                wave = sample[key]
                if isinstance(wave, torch.Tensor):
                    return wave.detach().cpu()
                if hasattr(wave, "numpy"):
                    arr = torch.from_numpy(wave)  # type: ignore[arg-type]
                    return arr.detach().cpu()
        raise ValueError(f"Cannot find waveform key in output keys={list(sample.keys())}")

    def _call_if_supported(self, fn_name: str, kwargs: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Call a model method only with supported arguments."""
        fn = getattr(self.model, fn_name, None)
        if fn is None:
            return None

        sig = inspect.signature(fn)
        accepted = {}
        for k, v in kwargs.items():
            if k in sig.parameters:
                accepted[k] = v

        output = fn(**accepted)
        return self._materialize_output(output)

    def synthesize(
        self,
        text: str,
        instruct_text: str,
        prompt_speech_16k: Optional[torch.Tensor] = None,
        sample_rate: int = 22050,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[torch.Tensor]:
        """Synthesize one text under an instruction prompt.

        Returns:
            list of waveform tensors, each shape [T] or [C, T]
        """
        if self.model is None:
            self.load_model()

        kwargs: Dict[str, Any] = {
            "text": text,
            "tts_text": text,
            "instruct_text": instruct_text,
            "instruction": instruct_text,
            "prompt_text": instruct_text,
            "prompt_speech_16k": prompt_speech_16k,
            "sample_rate": sample_rate,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        # Priority: instruction-aware inference.
        for method in ["inference_instruct2", "inference_instruct", "inference"]:
            try:
                outputs = self._call_if_supported(method, kwargs)
                if outputs:
                    return [self._pick_wave_tensor(sample) for sample in outputs]
            except Exception:
                continue

        # Last fallback: try call model directly as callable.
        if callable(self.model):
            out = self.model(text=text, instruct_text=instruct_text)
            samples = self._materialize_output(out)
            return [self._pick_wave_tensor(sample) for sample in samples]

        raise RuntimeError(
            "No valid inference method found on loaded CosyVoice model. "
            "Please check installed cosyvoice version."
        )


def get_default_loader(
    model_dir: str = "./models/CosyVoice2-0.5B",
    device: Optional[str] = None,
    fp16: bool = True,
) -> CosyVoiceModelLoader:
    """Convenience builder used by other modules."""
    cfg = ModelConfig(
        local_dir=model_dir,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        fp16=fp16,
    )
    return CosyVoiceModelLoader(cfg)


if __name__ == "__main__":
    loader = get_default_loader()
    model = loader.load_model(force_download=False)
    print(f"Loaded model type: {type(model)}")
