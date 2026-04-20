"""Main inference pipeline for fine-grained emotion-controllable TTS.

Features:
- Single sentence synthesis
- Batch synthesis from jsonl/csv
- Prompt A/B (baseline vs optimized)
- Metadata saving for thesis experiments
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from model_loader import CosyVoiceModelLoader, get_default_loader
from prompt_engineering import EmotionPromptEngineer, PromptConfig


@dataclass
class SynthesisRequest:
    text: str
    primary_emotion: str = "neutral"
    intensity: str = "moderately"  # slightly | moderately | very
    secondary_emotion: Optional[str] = None
    context: Optional[str] = None
    language: str = "zh"  # zh | en
    use_optimized_prompt: bool = True


@dataclass
class SynthesisResult:
    request_id: str
    audio_path: str
    sample_rate: int
    used_prompt: str
    request: Dict[str, Any]
    model_repo: str


class EmotionTTSPipeline:
    """Orchestrates prompt engineering + model inference + persistence."""

    def __init__(
        self,
        model_loader: Optional[CosyVoiceModelLoader] = None,
        output_dir: str = "./outputs",
        sample_rate: int = 22050,
    ) -> None:
        self.model_loader = model_loader or get_default_loader()
        self.prompt_engineer = EmotionPromptEngineer(PromptConfig(language="zh"))
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate

        self.audio_dir = self.output_dir / "audio"
        self.meta_path = self.output_dir / "metadata.jsonl"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_2d_wave(wave: torch.Tensor) -> torch.Tensor:
        """torchaudio.save expects [channels, time]."""
        if wave.ndim == 1:
            return wave.unsqueeze(0)
        if wave.ndim == 2:
            return wave
        return wave.reshape(1, -1)

    def _build_prompt(self, req: SynthesisRequest) -> str:
        if req.use_optimized_prompt:
            return self.prompt_engineer.build_optimized_prompt(
                text=req.text,
                primary_emotion=req.primary_emotion,
                intensity=req.intensity,
                secondary_emotion=req.secondary_emotion,
                context=req.context,
                language=req.language,
            )

        return self.prompt_engineer.build_baseline_prompt(
            text=req.text,
            emotion=req.primary_emotion,
            language=req.language,
        )

    def synthesize_single(self, req: SynthesisRequest) -> SynthesisResult:
        """Synthesize one utterance and save audio + metadata."""
        used_prompt = self._build_prompt(req)
        waves = self.model_loader.synthesize(
            text=req.text,
            instruct_text=used_prompt,
            sample_rate=self.sample_rate,
        )

        if not waves:
            raise RuntimeError("Model returned empty audio list.")

        wave = self._ensure_2d_wave(waves[0])
        request_id = uuid.uuid4().hex[:12]
        tag = "opt" if req.use_optimized_prompt else "base"
        audio_name = f"{request_id}_{req.language}_{req.primary_emotion}_{tag}.wav"
        audio_path = self.audio_dir / audio_name

        torchaudio.save(str(audio_path), wave, self.sample_rate)

        result = SynthesisResult(
            request_id=request_id,
            audio_path=str(audio_path.resolve()),
            sample_rate=self.sample_rate,
            used_prompt=used_prompt,
            request=asdict(req),
            model_repo=self.model_loader.config.hf_repo_id,
        )
        self._append_metadata(result)
        return result

    def _append_metadata(self, result: SynthesisResult) -> None:
        with self.meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    @staticmethod
    def _row_to_request(row: Dict[str, Any], use_optimized_prompt: bool = True) -> SynthesisRequest:
        return SynthesisRequest(
            text=str(row.get("text", "")).strip(),
            primary_emotion=str(row.get("primary_emotion", row.get("emotion", "neutral"))).strip(),
            intensity=str(row.get("intensity", "moderately")).strip(),
            secondary_emotion=(str(row["secondary_emotion"]).strip() if row.get("secondary_emotion") else None),
            context=(str(row["context"]).strip() if row.get("context") else None),
            language=str(row.get("language", "zh")).strip().lower(),
            use_optimized_prompt=bool(row.get("use_optimized_prompt", use_optimized_prompt)),
        )

    def synthesize_batch(
        self,
        requests: Iterable[SynthesisRequest],
        show_progress: bool = True,
    ) -> List[SynthesisResult]:
        """Batch synthesis for experiment generation."""
        results: List[SynthesisResult] = []
        iterator = tqdm(list(requests), desc="Batch Synth") if show_progress else requests
        for req in iterator:
            if not req.text:
                continue
            result = self.synthesize_single(req)
            results.append(result)
        return results

    def synthesize_from_file(
        self,
        input_path: str,
        use_optimized_prompt: bool = True,
    ) -> List[SynthesisResult]:
        """Load batch requests from .jsonl or .csv and run synthesis."""
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        rows: List[Dict[str, Any]] = []
        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif p.suffix.lower() == ".csv":
            rows = pd.read_csv(p).to_dict(orient="records")
        else:
            raise ValueError("Only .jsonl and .csv are supported for batch input")

        reqs = [self._row_to_request(row, use_optimized_prompt=use_optimized_prompt) for row in rows]
        return self.synthesize_batch(reqs)



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-grained emotion-controllable TTS inference")

    parser.add_argument("--model_dir", type=str, default="./models/CosyVoice2-0.5B")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--sample_rate", type=int, default=22050)

    # Single inference args
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--emotion", type=str, default="neutral")
    parser.add_argument("--intensity", type=str, default="moderately", choices=["slightly", "moderately", "very"])
    parser.add_argument("--secondary_emotion", type=str, default="")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"])

    # Batch inference args
    parser.add_argument("--batch_file", type=str, default="", help="Path to jsonl/csv batch file")

    # Prompt mode
    parser.add_argument("--prompt_mode", type=str, default="optimized", choices=["baseline", "optimized"])

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    loader = get_default_loader(model_dir=args.model_dir)
    loader.load_model(force_download=False)

    pipeline = EmotionTTSPipeline(
        model_loader=loader,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
    )

    use_optimized = args.prompt_mode == "optimized"

    if args.batch_file:
        results = pipeline.synthesize_from_file(args.batch_file, use_optimized_prompt=use_optimized)
        print(f"Batch done, generated {len(results)} samples.")
        print(f"Metadata: {pipeline.meta_path.resolve()}")
        return

    if not args.text:
        raise ValueError("For single inference, --text is required.")

    req = SynthesisRequest(
        text=args.text,
        primary_emotion=args.emotion,
        intensity=args.intensity,
        secondary_emotion=args.secondary_emotion or None,
        context=args.context or None,
        language=args.language,
        use_optimized_prompt=use_optimized,
    )
    result = pipeline.synthesize_single(req)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
