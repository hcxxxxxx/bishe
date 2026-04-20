"""Model download and loading utilities for CosyVoice2-0.5B.

This module is intentionally defensive because CosyVoice APIs may vary slightly
across versions. It tries several import/inference paths and reports clear errors.
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    END_OF_PROMPT_TOKEN = "<|endofprompt|>"

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
            force_download=force_download,
        )
        return self.model_dir

    @staticmethod
    def _maybe_prepare_cosyvoice_source_import() -> Optional[str]:
        """Try adding an official CosyVoice source directory into sys.path.

        This helps when users clone FunAudioLLM/CosyVoice locally rather than
        using a pip package that exposes `cosyvoice.cli`.
        """
        candidates = [
            os.getenv("COSYVOICE_REPO", ""),
            "./third_party/CosyVoice",
            "./CosyVoice",
            "../CosyVoice",
            "../../CosyVoice",
        ]
        for c in candidates:
            if not c:
                continue
            p = Path(c).resolve()
            marker = p / "cosyvoice" / "cli" / "cosyvoice.py"
            if marker.exists():
                p_str = str(p)
                if p_str not in sys.path:
                    sys.path.insert(0, p_str)
                return p_str
        return None

    def load_model(self, force_download: bool = False) -> Any:
        """Load CosyVoice2 model with best-effort API compatibility."""
        if self.model is not None:
            return self.model

        model_dir = self.download_model(force_download=force_download)

        import_errors = []
        source_path = self._maybe_prepare_cosyvoice_source_import()

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
        fix_hint = (
            "Detected no usable `cosyvoice.cli` runtime. "
            "Please install the official FunAudioLLM/CosyVoice runtime.\n"
            "Recommended steps:\n"
            "1) git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
            "2) cd CosyVoice && git submodule update --init --recursive\n"
            "3) pip install -r requirements.txt\n"
            "4) export COSYVOICE_REPO=/absolute/path/to/CosyVoice\n"
            "Then rerun this script.\n"
        )
        if source_path:
            fix_hint += f"Already added source path into sys.path: {source_path}\n"

        raise RuntimeError(
            "Unable to load CosyVoice model. Checked multiple API paths.\n"
            f"Model dir: {model_dir}\n"
            f"Errors:\n{msg}\n\n{fix_hint}"
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
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        for k, v in kwargs.items():
            if has_var_keyword or k in sig.parameters:
                accepted[k] = v

        output = fn(**accepted)
        return self._materialize_output(output)

    def _call_with_patterns(
        self,
        fn_name: str,
        kwargs_patterns: Sequence[Dict[str, Any]],
        args_patterns: Sequence[Tuple[Any, ...]],
    ) -> List[Dict[str, Any]]:
        """Try one model method with multiple kwargs/args patterns."""
        fn = getattr(self.model, fn_name, None)
        if fn is None:
            raise AttributeError(f"Method `{fn_name}` not found.")

        errors: List[str] = []
        for kwargs in kwargs_patterns:
            try:
                outputs = self._call_if_supported(fn_name, kwargs) or []
                if outputs:
                    return outputs
            except Exception as exc:  # noqa: BLE001
                errors.append(f"kwargs={list(kwargs.keys())}: {type(exc).__name__}: {exc}")

        for args in args_patterns:
            try:
                outputs = self._materialize_output(fn(*args))
                if outputs:
                    return outputs
            except Exception as exc:  # noqa: BLE001
                errors.append(f"args_len={len(args)}: {type(exc).__name__}: {exc}")

        raise RuntimeError("; ".join(errors) if errors else "No valid output returned.")

    @classmethod
    def _normalize_instruct_text(cls, instruct_text: str) -> str:
        """Normalize instruction text for CosyVoice2 instruct-style APIs.

        In several CosyVoice2 builds, instruct_text is treated as prompt text and
        may be spoken out. Appending <|endofprompt|> helps stop leakage.
        """
        text = (instruct_text or "").strip()
        if not text:
            return cls.END_OF_PROMPT_TOKEN
        if text.endswith(cls.END_OF_PROMPT_TOKEN):
            return text
        return f"{text}{cls.END_OF_PROMPT_TOKEN}"

    def synthesize(
        self,
        text: str,
        instruct_text: str,
        prompt_speech_16k: Optional[Any] = None,
        sample_rate: int = 22050,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[torch.Tensor]:
        """Synthesize one text under an instruction prompt.

        Returns:
            list of waveform tensors, each shape [T] or [C, T]
        """
        if self.model is None:
            self.load_model()
        normalized_instruct_text = self._normalize_instruct_text(instruct_text)

        kwargs: Dict[str, Any] = {
            "text": text,
            "tts_text": text,
            "instruct_text": normalized_instruct_text,
            "instruction": normalized_instruct_text,
            "prompt_text": normalized_instruct_text,
            "prompt_speech_16k": prompt_speech_16k,
            # Some CosyVoice2 versions use `prompt_wav` instead.
            "prompt_wav": prompt_speech_16k,
            "sample_rate": sample_rate,
            "stream": False,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        spk_id = kwargs.get("spk_id")
        method_errors: List[str] = []
        available_methods = sorted([name for name in dir(self.model) if name.startswith("inference")])

        # Priority: instruction-aware inference. For CosyVoice2 this commonly
        # needs prompt_speech_16k, so we try both with/without prompt patterns.
        planned_calls: List[Tuple[str, Sequence[Dict[str, Any]], Sequence[Tuple[Any, ...]]]] = [
            (
                "inference_instruct2",
                [
                    {
                        "tts_text": text,
                        "instruct_text": normalized_instruct_text,
                        "prompt_wav": prompt_speech_16k,
                        "prompt_speech_16k": prompt_speech_16k,
                        "stream": False,
                        "text_frontend": False,
                    },
                    {
                        "text": text,
                        "instruct_text": normalized_instruct_text,
                        "prompt_wav": prompt_speech_16k,
                        "prompt_speech_16k": prompt_speech_16k,
                        "stream": False,
                        "text_frontend": False,
                    },
                    {
                        "tts_text": text,
                        "instruct_text": normalized_instruct_text,
                        "stream": False,
                        "text_frontend": False,
                    },
                    {
                        "text": text,
                        "instruct_text": normalized_instruct_text,
                        "stream": False,
                        "text_frontend": False,
                    },
                ],
                [
                    (text, normalized_instruct_text, prompt_speech_16k),
                    (text, normalized_instruct_text),
                ],
            ),
            (
                "inference_zero_shot",
                [
                    {
                        "tts_text": text,
                        "prompt_text": normalized_instruct_text,
                        "prompt_wav": prompt_speech_16k,
                        "prompt_speech_16k": prompt_speech_16k,
                        "stream": False,
                        "text_frontend": False,
                    },
                    {
                        "text": text,
                        "prompt_text": normalized_instruct_text,
                        "prompt_wav": prompt_speech_16k,
                        "prompt_speech_16k": prompt_speech_16k,
                        "stream": False,
                        "text_frontend": False,
                    },
                ],
                [
                    (text, normalized_instruct_text, prompt_speech_16k),
                ],
            ),
            (
                "inference_instruct",
                [
                    {
                        "tts_text": text,
                        "instruct_text": normalized_instruct_text,
                        "spk_id": spk_id,
                        "stream": False,
                    },
                    {
                        "text": text,
                        "instruct_text": normalized_instruct_text,
                        "spk_id": spk_id,
                        "stream": False,
                    },
                ],
                [
                    (text, normalized_instruct_text),
                ],
            ),
            (
                "inference_sft",
                [
                    {
                        "tts_text": text,
                        "spk_id": spk_id,
                        "stream": False,
                    },
                    {
                        "text": text,
                        "spk_id": spk_id,
                        "stream": False,
                    },
                ],
                [
                    (text, spk_id),
                ],
            ),
            (
                "inference",
                [
                    kwargs,
                ],
                [
                    (text,),
                ],
            ),
        ]

        for method, kwargs_patterns, args_patterns in planned_calls:
            # If no speaker id is provided, skip SFT paths.
            if method == "inference_sft" and not spk_id:
                continue
            try:
                outputs = self._call_with_patterns(method, kwargs_patterns, args_patterns)
                if outputs:
                    return [self._pick_wave_tensor(sample) for sample in outputs]
            except Exception as exc:  # noqa: BLE001
                method_errors.append(f"{method}: {type(exc).__name__}: {exc}")

        # Last fallback: try call model directly as callable.
        if callable(self.model):
            try:
                out = self.model(text=text, instruct_text=normalized_instruct_text)
                samples = self._materialize_output(out)
                if samples:
                    return [self._pick_wave_tensor(sample) for sample in samples]
            except Exception as exc:  # noqa: BLE001
                method_errors.append(f"__call__: {type(exc).__name__}: {exc}")

        hint = ""
        if prompt_speech_16k is None and "inference_instruct2" in available_methods:
            hint = (
                "Hint: current CosyVoice2 version may require a reference prompt audio for "
                "`inference_instruct2` (`prompt_wav` / `prompt_speech_16k`). "
                "Try passing CLI `--prompt_audio`."
            )
        error_block = "\n- ".join(method_errors[:8]) if method_errors else "No callable inference methods succeeded."
        raise RuntimeError(
            "No valid inference method found on loaded CosyVoice model. "
            "Please check installed cosyvoice version.\n"
            f"Available inference methods: {available_methods}\n"
            f"Method errors:\n- {error_block}" + ("\n" + hint if hint else "")
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
