"""Batch intensity-comparison inference runner.

This script expands each input row into multiple runs over:
- intensity levels: slightly/moderately/very (customizable)
- prompt modes: baseline/optimized (customizable)

Input supports csv/jsonl with fields:
- text (required)
- primary_emotion / emotion
- secondary_emotion (optional)
- context (optional)
- language (optional, zh/en)
- prompt_audio_path or prompt_audio (optional, recommended for CosyVoice2 instruct2)
- spk_id (optional)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from tqdm import tqdm

from inference_pipeline import EmotionTTSPipeline, SynthesisRequest
from model_loader import get_default_loader


def _load_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif p.suffix.lower() == ".csv":
        rows = pd.read_csv(p).to_dict(orient="records")
    else:
        raise ValueError("input_file must be .jsonl or .csv")
    return rows


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": str(row.get("text", "")).strip(),
        "primary_emotion": str(row.get("primary_emotion", row.get("emotion", "neutral"))).strip(),
        "secondary_emotion": (str(row["secondary_emotion"]).strip() if row.get("secondary_emotion") else None),
        "context": (str(row["context"]).strip() if row.get("context") else None),
        "language": str(row.get("language", "zh")).strip().lower() or "zh",
        "prompt_audio_path": (
            str(row.get("prompt_audio_path", row.get("prompt_audio", ""))).strip() or None
        ),
        "spk_id": (str(row["spk_id"]).strip() if row.get("spk_id") else None),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch intensity comparison inference runner")
    parser.add_argument("--input_file", type=str, required=True, help="csv/jsonl with base samples")
    parser.add_argument("--model_dir", type=str, default="./models/CosyVoice2-0.5B")
    parser.add_argument("--output_dir", type=str, default="./outputs/intensity_batch")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--intensities", type=str, default="slightly,moderately,very")
    parser.add_argument("--prompt_modes", type=str, default="baseline,optimized")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    intensities = [x.strip() for x in args.intensities.split(",") if x.strip()]
    prompt_modes = [x.strip().lower() for x in args.prompt_modes.split(",") if x.strip()]
    valid_modes = {"baseline", "optimized"}
    for m in prompt_modes:
        if m not in valid_modes:
            raise ValueError(f"Invalid prompt mode: {m}")

    rows = _load_rows(args.input_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    base_samples = [_normalize_row(r) for r in rows if str(r.get("text", "")).strip()]

    loader = get_default_loader(model_dir=args.model_dir)
    loader.load_model(force_download=False)
    pipeline = EmotionTTSPipeline(model_loader=loader, output_dir=args.output_dir, sample_rate=args.sample_rate)

    run_plan: List[Dict[str, Any]] = []
    for idx, sample in enumerate(base_samples):
        for mode in prompt_modes:
            for intensity in intensities:
                run_plan.append(
                    {
                        "sample_index": idx,
                        "prompt_mode": mode,
                        "intensity": intensity,
                        **sample,
                    }
                )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "run_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as f:
        for row in run_plan:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result_path = out_dir / "run_results.jsonl"
    error_path = out_dir / "run_errors.jsonl"

    ok_count = 0
    err_count = 0

    with result_path.open("w", encoding="utf-8") as f_ok, error_path.open("w", encoding="utf-8") as f_err:
        for item in tqdm(run_plan, desc="Intensity Batch"):
            req = SynthesisRequest(
                text=item["text"],
                primary_emotion=item["primary_emotion"],
                intensity=item["intensity"],
                secondary_emotion=item["secondary_emotion"],
                context=item["context"],
                language=item["language"],
                use_optimized_prompt=(item["prompt_mode"] == "optimized"),
                prompt_audio_path=item["prompt_audio_path"],
                spk_id=item["spk_id"],
            )
            try:
                result = pipeline.synthesize_single(req)
                record = {
                    "sample_index": item["sample_index"],
                    "prompt_mode": item["prompt_mode"],
                    "intensity": item["intensity"],
                    "request": asdict(req),
                    "result": asdict(result),
                }
                f_ok.write(json.dumps(record, ensure_ascii=False) + "\n")
                ok_count += 1
            except Exception as exc:  # noqa: BLE001
                err = {
                    "sample_index": item["sample_index"],
                    "prompt_mode": item["prompt_mode"],
                    "intensity": item["intensity"],
                    "request": asdict(req),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                f_err.write(json.dumps(err, ensure_ascii=False) + "\n")
                err_count += 1

    print(f"Planned runs: {len(run_plan)}")
    print(f"Succeeded: {ok_count}")
    print(f"Failed: {err_count}")
    print(f"Plan file: {plan_path.resolve()}")
    print(f"Results file: {result_path.resolve()}")
    print(f"Errors file: {error_path.resolve()}")
    print(f"Pipeline metadata file: {(Path(args.output_dir) / 'metadata.jsonl').resolve()}")


if __name__ == "__main__":
    main()
