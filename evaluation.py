"""Evaluation script for emotion-controllable TTS experiments.

Includes:
- Prompt version comparison (baseline vs optimized)
- Optional automatic scoring with emotion2vec / SenseVoice
- MOS template generation for subjective listening tests
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from inference_pipeline import EmotionTTSPipeline, SynthesisRequest
from model_loader import get_default_loader


@dataclass
class EvalSample:
    text: str
    primary_emotion: str
    intensity: str
    secondary_emotion: Optional[str] = None
    context: Optional[str] = None
    language: str = "zh"


class EmotionAutoEvaluator:
    """Best-effort evaluator using optional FunASR models."""

    def __init__(self) -> None:
        self.emotion2vec_model = None
        self.sensevoice_model = None
        self._init_models()

    def _init_models(self) -> None:
        try:
            from funasr import AutoModel  # type: ignore

            # emotion2vec for embedding-level similarity.
            self.emotion2vec_model = AutoModel(model="iic/emotion2vec_plus_large")
        except Exception:
            self.emotion2vec_model = None

        try:
            from funasr import AutoModel  # type: ignore

            # SenseVoice can output textual style/emotion tags.
            self.sensevoice_model = AutoModel(model="iic/SenseVoiceSmall")
        except Exception:
            self.sensevoice_model = None

    def get_emotion2vec_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        if self.emotion2vec_model is None:
            return None
        try:
            out = self.emotion2vec_model.generate(input=audio_path)
            # Handle several possible output structures.
            if isinstance(out, dict):
                for key in ["feats", "emb", "embedding", "emo_embedding"]:
                    if key in out:
                        return np.asarray(out[key]).reshape(1, -1)
            if isinstance(out, list) and out:
                item = out[0]
                if isinstance(item, dict):
                    for key in ["feats", "emb", "embedding", "emo_embedding"]:
                        if key in item:
                            return np.asarray(item[key]).reshape(1, -1)
            return None
        except Exception:
            return None

    def parse_sensevoice_emotion(self, audio_path: str) -> Optional[str]:
        if self.sensevoice_model is None:
            return None
        try:
            out = self.sensevoice_model.generate(input=audio_path)
            text = str(out)
            # Simple tag parsing fallback.
            candidates = ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "SURPRISED", "FEARFUL"]
            for c in candidates:
                if c in text.upper():
                    return c.lower()
            return None
        except Exception:
            return None

    def compare_audio_emotion_similarity(self, audio_a: str, audio_b: str) -> Optional[float]:
        emb_a = self.get_emotion2vec_embedding(audio_a)
        emb_b = self.get_emotion2vec_embedding(audio_b)
        if emb_a is None or emb_b is None:
            return None
        return float(cosine_similarity(emb_a, emb_b)[0][0])


class ExperimentRunner:
    """Run prompt ablation and collect objective/subjective evaluation artifacts."""

    def __init__(self, output_dir: str = "./experiments") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        loader = get_default_loader(model_dir="./models/CosyVoice2-0.5B")
        loader.load_model(force_download=False)
        self.pipeline = EmotionTTSPipeline(model_loader=loader, output_dir=str(self.output_dir / "synthesis"))
        self.evaluator = EmotionAutoEvaluator()

    @staticmethod
    def load_eval_samples(path: str) -> List[EvalSample]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        if p.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif p.suffix.lower() == ".csv":
            rows = pd.read_csv(p).to_dict(orient="records")
        else:
            raise ValueError("Evaluation file should be jsonl or csv")

        samples = []
        for row in rows:
            samples.append(
                EvalSample(
                    text=str(row.get("text", "")).strip(),
                    primary_emotion=str(row.get("primary_emotion", row.get("emotion", "neutral"))).strip(),
                    intensity=str(row.get("intensity", "moderately")).strip(),
                    secondary_emotion=(str(row["secondary_emotion"]).strip() if row.get("secondary_emotion") else None),
                    context=(str(row["context"]).strip() if row.get("context") else None),
                    language=str(row.get("language", "zh")).strip().lower(),
                )
            )
        return samples

    def run_prompt_comparison(self, samples: List[EvalSample]) -> pd.DataFrame:
        """Generate baseline/optimized pairs and compute optional metrics."""
        records: List[Dict[str, Any]] = []

        for sample in samples:
            req_base = SynthesisRequest(
                text=sample.text,
                primary_emotion=sample.primary_emotion,
                intensity=sample.intensity,
                secondary_emotion=sample.secondary_emotion,
                context=sample.context,
                language=sample.language,
                use_optimized_prompt=False,
            )
            req_opt = SynthesisRequest(
                text=sample.text,
                primary_emotion=sample.primary_emotion,
                intensity=sample.intensity,
                secondary_emotion=sample.secondary_emotion,
                context=sample.context,
                language=sample.language,
                use_optimized_prompt=True,
            )

            base_result = self.pipeline.synthesize_single(req_base)
            opt_result = self.pipeline.synthesize_single(req_opt)

            emb_sim = self.evaluator.compare_audio_emotion_similarity(
                base_result.audio_path,
                opt_result.audio_path,
            )
            sv_base = self.evaluator.parse_sensevoice_emotion(base_result.audio_path)
            sv_opt = self.evaluator.parse_sensevoice_emotion(opt_result.audio_path)

            records.append(
                {
                    "text": sample.text,
                    "language": sample.language,
                    "target_primary_emotion": sample.primary_emotion,
                    "target_secondary_emotion": sample.secondary_emotion,
                    "target_intensity": sample.intensity,
                    "baseline_audio": base_result.audio_path,
                    "optimized_audio": opt_result.audio_path,
                    "baseline_prompt": base_result.used_prompt,
                    "optimized_prompt": opt_result.used_prompt,
                    "emotion2vec_cosine_between_versions": emb_sim,
                    "sensevoice_pred_baseline": sv_base,
                    "sensevoice_pred_optimized": sv_opt,
                }
            )

        df = pd.DataFrame(records)
        save_path = self.output_dir / "prompt_comparison_results.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Saved comparison results to: {save_path.resolve()}")
        return df

    def generate_mos_template(self, df: pd.DataFrame) -> Path:
        """Create a MOS evaluation form template (CSV)."""
        mos_path = self.output_dir / "mos_template.csv"
        headers = [
            "sample_id",
            "text",
            "target_emotion",
            "audio_version",
            "audio_path",
            "naturalness_1to5",
            "emotion_accuracy_1to5",
            "intensity_match_1to5",
            "comment",
        ]

        rows = []
        for idx, row in df.iterrows():
            rows.append(
                {
                    "sample_id": f"{idx}_baseline",
                    "text": row["text"],
                    "target_emotion": row["target_primary_emotion"],
                    "audio_version": "baseline",
                    "audio_path": row["baseline_audio"],
                    "naturalness_1to5": "",
                    "emotion_accuracy_1to5": "",
                    "intensity_match_1to5": "",
                    "comment": "",
                }
            )
            rows.append(
                {
                    "sample_id": f"{idx}_optimized",
                    "text": row["text"],
                    "target_emotion": row["target_primary_emotion"],
                    "audio_version": "optimized",
                    "audio_path": row["optimized_audio"],
                    "naturalness_1to5": "",
                    "emotion_accuracy_1to5": "",
                    "intensity_match_1to5": "",
                    "comment": "",
                }
            )

        with mos_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved MOS template to: {mos_path.resolve()}")
        return mos_path

    @staticmethod
    def _load_tendency_rows(path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if p.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p).to_dict(orient="records")
        raise ValueError("tendency_file should be jsonl or csv")

    def evaluate_emotion_tendency(self, tendency_file: str) -> Path:
        """Evaluate emotion tendency on generated audios.

        Input file can be:
        - outputs/metadata.jsonl from inference_pipeline.py
        - custom csv/jsonl with at least `audio_path`
        Optional target columns:
        - `primary_emotion` or `target_emotion`
        """
        rows = self._load_tendency_rows(tendency_file)
        records: List[Dict[str, Any]] = []
        target_list: List[str] = []
        pred_list: List[str] = []

        for row in rows:
            audio_path = row.get("audio_path")
            if not audio_path:
                continue

            target = (
                row.get("target_emotion")
                or row.get("primary_emotion")
                or (row.get("request") or {}).get("primary_emotion")
            )
            predicted = self.evaluator.parse_sensevoice_emotion(str(audio_path))
            records.append(
                {
                    "audio_path": audio_path,
                    "target_emotion": target,
                    "predicted_emotion": predicted,
                    "match": bool(target and predicted and str(target).lower() == str(predicted).lower()),
                }
            )
            if target and predicted:
                target_list.append(str(target).lower())
                pred_list.append(str(predicted).lower())

        df = pd.DataFrame(records)
        detail_path = self.output_dir / "emotion_tendency_results.csv"
        df.to_csv(detail_path, index=False, encoding="utf-8-sig")

        summary = {
            "num_samples": int(len(df)),
            "num_with_target_and_prediction": int(len(target_list)),
            "prediction_distribution": dict(Counter([str(x).lower() for x in df["predicted_emotion"].dropna().tolist()])),
        }
        if target_list:
            acc = float(np.mean([t == p for t, p in zip(target_list, pred_list)]))
            summary["target_match_accuracy"] = acc

            labels = sorted(set(target_list + pred_list))
            cm = confusion_matrix(target_list, pred_list, labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"target_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
            cm_path = self.output_dir / "emotion_tendency_confusion_matrix.csv"
            cm_df.to_csv(cm_path, encoding="utf-8-sig")
            summary["confusion_matrix_csv"] = str(cm_path.resolve())

        summary_path = self.output_dir / "emotion_tendency_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved tendency detail to: {detail_path.resolve()}")
        print(f"Saved tendency summary to: {summary_path.resolve()}")
        return detail_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation for emotion-controllable TTS")
    parser.add_argument("--eval_file", type=str, default="", help="Path to eval csv/jsonl for prompt comparison")
    parser.add_argument("--tendency_file", type=str, default="", help="Path to csv/jsonl for emotion tendency evaluation")
    parser.add_argument("--output_dir", type=str, default="./experiments")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    runner = ExperimentRunner(output_dir=args.output_dir)
    if args.tendency_file:
        runner.evaluate_emotion_tendency(args.tendency_file)
        return
    if not args.eval_file:
        raise ValueError("Either --eval_file or --tendency_file must be provided.")
    samples = runner.load_eval_samples(args.eval_file)
    df = runner.run_prompt_comparison(samples)
    runner.generate_mos_template(df)


if __name__ == "__main__":
    main()
