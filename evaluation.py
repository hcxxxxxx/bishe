"""Evaluation script for emotion-controllable TTS experiments.

Includes:
- Prompt version comparison (baseline vs optimized)
- Optional automatic scoring with emotion2vec / SenseVoice
- MOS template generation for subjective listening tests
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import classification_report
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
    def _save_df_txt(df: pd.DataFrame, path: Path) -> Path:
        """Save aligned plain-text table for terminal-friendly reading."""
        if len(df) == 0:
            path.write_text("(empty)\n", encoding="utf-8")
            return path
        text = df.to_string(index=False, max_colwidth=120)
        path.write_text(text + "\n", encoding="utf-8")
        return path

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
        jsonl_path = self.output_dir / "prompt_comparison_results.jsonl"
        txt_path = self.output_dir / "prompt_comparison_results.txt"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in df.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._save_df_txt(df, txt_path)
        print(f"Saved comparison results to: {jsonl_path.resolve()}")
        print(f"Saved comparison table to: {txt_path.resolve()}")
        return df

    def generate_mos_template(self, df: pd.DataFrame) -> Path:
        """Create a MOS evaluation form template (JSONL + TXT)."""
        mos_jsonl_path = self.output_dir / "mos_template.jsonl"
        mos_txt_path = self.output_dir / "mos_template.txt"
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

        with mos_jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._save_df_txt(pd.DataFrame(rows, columns=headers), mos_txt_path)

        print(f"Saved MOS template to: {mos_jsonl_path.resolve()}")
        print(f"Saved MOS table to: {mos_txt_path.resolve()}")
        return mos_jsonl_path

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    @staticmethod
    def _safe_parse_request(row: Dict[str, Any]) -> Dict[str, Any]:
        req = row.get("request")
        if isinstance(req, dict):
            return req
        if isinstance(req, str):
            s = req.strip()
            if not s:
                return {}
            for parser in (json.loads, ast.literal_eval):
                try:
                    obj = parser(s)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        return {}

    @classmethod
    def _extract_field(cls, row: Dict[str, Any], *keys: str) -> Any:
        req = cls._safe_parse_request(row)
        for key in keys:
            val = row.get(key)
            if not cls._is_missing(val):
                return val
        for key in keys:
            val = req.get(key)
            if not cls._is_missing(val):
                return val
        return None

    @staticmethod
    def _normalize_emotion_label(label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        s = str(label).strip().lower()
        if not s:
            return None
        mapping = {
            "开心": "happy",
            "高兴": "happy",
            "愉快": "happy",
            "伤心": "sad",
            "悲伤": "sad",
            "难过": "sad",
            "愤怒": "angry",
            "生气": "angry",
            "惊讶": "surprised",
            "紧张": "fearful",
            "害怕": "fearful",
            "恐惧": "fearful",
            "温柔": "gentle",
            "严肃": "serious",
            "中性": "neutral",
        }
        return mapping.get(s, s)

    @staticmethod
    def _normalize_intensity_label(label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        s = str(label).strip().lower()
        if not s:
            return None
        mapping = {
            "轻微": "slightly",
            "略微": "slightly",
            "稍微": "slightly",
            "适度": "moderately",
            "中等": "moderately",
            "中度": "moderately",
            "非常": "very",
            "强烈": "very",
        }
        return mapping.get(s, s)

    @staticmethod
    def _normalize_prompt_mode(row: Dict[str, Any]) -> str:
        """Resolve prompt mode from top-level fields or nested request."""
        req = ExperimentRunner._safe_parse_request(row)

        mode = row.get("prompt_mode")
        if ExperimentRunner._is_missing(mode):
            mode = req.get("prompt_mode")
        if not ExperimentRunner._is_missing(mode):
            s = str(mode).strip().lower()
            if s in ("baseline", "optimized"):
                return s

        use_opt = row.get("use_optimized_prompt")
        if ExperimentRunner._is_missing(use_opt):
            use_opt = req.get("use_optimized_prompt")
        if isinstance(use_opt, bool):
            return "optimized" if use_opt else "baseline"
        if isinstance(use_opt, str):
            s = use_opt.strip().lower()
            if s in ("1", "true", "yes", "y", "optimized"):
                return "optimized"
            if s in ("0", "false", "no", "n", "baseline"):
                return "baseline"

        return "unknown"

    @staticmethod
    def _extract_audio_features(audio_path: str) -> Dict[str, float]:
        wav, sr = torchaudio.load(audio_path)
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.float()
        duration_sec = float(wav.shape[-1] / max(sr, 1))
        rms = float(torch.sqrt(torch.mean(wav**2) + 1e-9))
        rms_db = float(20.0 * np.log10(max(rms, 1e-6)))
        peak = float(torch.max(torch.abs(wav)))
        if wav.shape[-1] > 1:
            zcr = float(torch.mean((wav[:, 1:] * wav[:, :-1] < 0).float()))
        else:
            zcr = 0.0
        return {
            "duration_sec": duration_sec,
            "rms_db": rms_db,
            "peak_abs": peak,
            "zcr": zcr,
        }

    @staticmethod
    def _intensity_proxy_score(features: Dict[str, float], text: str) -> float:
        duration = max(features.get("duration_sec", 0.0), 1e-4)
        text_len = len((text or "").strip())
        speech_rate = text_len / duration
        # Simple interpretable proxy: louder + faster + more dynamic => stronger.
        return float(features.get("rms_db", -60.0) + 0.05 * speech_rate + 10.0 * features.get("zcr", 0.0))

    @staticmethod
    def _evaluate_intensity_monotonicity(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        order = {"slightly": 0, "moderately": 1, "very": 2}
        intensity_rows: List[Dict[str, Any]] = []
        compare_cols = ["text", "target_primary_emotion", "target_secondary_emotion", "language"]

        def _calc(sub_df: pd.DataFrame, mode_label: str, collect_rows: bool = True) -> Dict[str, Any]:
            total_groups = 0
            pass_groups = 0
            grouped = sub_df.groupby(compare_cols, dropna=False)
            for (text, p, s, lang), g in grouped:
                sub = g.copy()
                sub["intensity_order"] = sub["target_intensity"].map(order)
                sub = sub.dropna(subset=["intensity_order", "intensity_proxy"])
                if len(sub) < 2:
                    continue

                total_groups += 1
                sub = sub.sort_values("intensity_order")
                vals = sub["intensity_proxy"].tolist()
                intensities = sub["target_intensity"].tolist()
                is_mono = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
                if is_mono:
                    pass_groups += 1

                if collect_rows:
                    intensity_rows.append(
                        {
                            "prompt_mode": mode_label,
                            "text": text,
                            "target_primary_emotion": p,
                            "target_secondary_emotion": s,
                            "language": lang,
                            "intensity_sequence": " -> ".join(intensities),
                            "score_sequence": " -> ".join([f"{v:.3f}" for v in vals]),
                            "monotonic_pass": is_mono,
                        }
                    )

            return {
                "num_intensity_groups": total_groups,
                "num_monotonic_pass_groups": pass_groups,
                "monotonic_pass_rate": (pass_groups / total_groups) if total_groups > 0 else None,
            }

        has_mode = "prompt_mode" in df.columns and df["prompt_mode"].notna().any()
        overall_stats = _calc(df, "all", collect_rows=not has_mode)
        by_mode: Dict[str, Any] = {}
        if has_mode:
            for mode, sub in df.groupby("prompt_mode", dropna=False):
                by_mode[str(mode)] = _calc(sub, str(mode))

        summary = {
            "num_intensity_groups": overall_stats["num_intensity_groups"],
            "num_monotonic_pass_groups": overall_stats["num_monotonic_pass_groups"],
            "monotonic_pass_rate": overall_stats["monotonic_pass_rate"],
            "monotonic_by_prompt_mode": by_mode,
        }
        return summary, pd.DataFrame(intensity_rows)

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
        target_primary_list: List[str] = []
        pred_list: List[str] = []

        for row in rows:
            audio_path = row.get("audio_path")
            if not audio_path:
                continue

            prompt_mode = self._normalize_prompt_mode(row)
            target_primary = self._normalize_emotion_label(
                self._extract_field(row, "target_emotion", "target_primary_emotion", "primary_emotion")
            )
            target_secondary = self._normalize_emotion_label(
                self._extract_field(row, "target_secondary_emotion", "secondary_emotion")
            )
            target_intensity = self._normalize_intensity_label(
                self._extract_field(row, "target_intensity", "intensity")
            )
            text = str(self._extract_field(row, "text") or "")
            predicted = self.evaluator.parse_sensevoice_emotion(str(audio_path))
            predicted = self._normalize_emotion_label(predicted)
            feats = self._extract_audio_features(str(audio_path))
            intensity_proxy = self._intensity_proxy_score(feats, text)

            records.append(
                {
                    "audio_path": audio_path,
                    "text": text,
                    "language": self._extract_field(row, "language"),
                    "prompt_mode": prompt_mode,
                    "target_primary_emotion": target_primary,
                    "target_secondary_emotion": target_secondary,
                    "target_intensity": target_intensity,
                    "predicted_emotion": predicted,
                    "primary_match": bool(target_primary and predicted and target_primary == predicted),
                    "secondary_match": bool(target_secondary and predicted and target_secondary == predicted),
                    "composite_relaxed_match": bool(
                        predicted and (predicted == target_primary or predicted == target_secondary)
                    ),
                    "duration_sec": feats["duration_sec"],
                    "rms_db": feats["rms_db"],
                    "peak_abs": feats["peak_abs"],
                    "zcr": feats["zcr"],
                    "intensity_proxy": intensity_proxy,
                }
            )
            if target_primary and predicted:
                target_primary_list.append(target_primary)
                pred_list.append(predicted)

        df = pd.DataFrame(records)
        detail_jsonl_path = self.output_dir / "emotion_tendency_results.jsonl"
        detail_txt_path = self.output_dir / "emotion_tendency_results.txt"
        with detail_jsonl_path.open("w", encoding="utf-8") as f:
            for row in df.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._save_df_txt(df, detail_txt_path)

        summary = {
            "num_samples": int(len(df)),
            "num_with_primary_target_and_prediction": int(len(target_primary_list)),
            "prediction_distribution": dict(Counter([str(x).lower() for x in df["predicted_emotion"].dropna().tolist()])),
            "primary_match_rate": float(df["primary_match"].mean()) if len(df) > 0 else None,
            "secondary_match_rate": float(df["secondary_match"].mean()) if len(df) > 0 else None,
            "composite_relaxed_match_rate": float(df["composite_relaxed_match"].mean()) if len(df) > 0 else None,
        }
        if target_primary_list:
            acc = float(np.mean([t == p for t, p in zip(target_primary_list, pred_list)]))
            summary["target_primary_match_accuracy"] = acc

            labels = sorted(set(target_primary_list + pred_list))
            cm = confusion_matrix(target_primary_list, pred_list, labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"target_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
            cm_txt_path = self.output_dir / "emotion_tendency_confusion_matrix.txt"
            self._save_df_txt(cm_df.reset_index().rename(columns={"index": "target"}), cm_txt_path)
            summary["confusion_matrix_txt"] = str(cm_txt_path.resolve())

            report = classification_report(target_primary_list, pred_list, output_dict=True, zero_division=0)
            report_path = self.output_dir / "emotion_tendency_classification_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["classification_report_json"] = str(report_path.resolve())

        # Intensity-control metric: monotonicity over grouped triples.
        intensity_summary, intensity_df = self._evaluate_intensity_monotonicity(df)
        summary.update(intensity_summary)
        intensity_txt_path = self.output_dir / "emotion_tendency_intensity_groups.txt"
        self._save_df_txt(intensity_df, intensity_txt_path)
        summary["intensity_group_txt"] = str(intensity_txt_path.resolve())

        summary_path = self.output_dir / "emotion_tendency_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved tendency detail to: {detail_jsonl_path.resolve()}")
        print(f"Saved tendency table to: {detail_txt_path.resolve()}")
        print(f"Saved tendency summary to: {summary_path.resolve()}")
        return detail_jsonl_path


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
