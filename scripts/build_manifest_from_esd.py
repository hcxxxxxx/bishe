#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import List


EMO_HINTS = {
    "sad": "sad",
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "surprise": "surprise",
    "fear": "fear",
    "悲": "sad",
    "怒": "angry",
    "喜": "happy",
    "中性": "neutral",
    "惊": "surprise",
    "惧": "fear",
}


def _auto_prompt(emotion: str, intensity: float, style: str) -> str:
    if intensity < 0.4:
        level = "略带"
    elif intensity < 0.7:
        level = "比较"
    else:
        level = "非常"
    return f"请用{level}{emotion}且{style}的语气朗读这句话。"


def _label_to_va(emotion: str):
    table = {
        "happy": (0.8, 0.7, "高兴"),
        "sad": (-0.75, -0.35, "悲伤"),
        "angry": (-0.85, 0.9, "愤怒"),
        "neutral": (0.0, 0.0, "中性"),
        "surprise": (0.45, 0.9, "兴奋"),
        "fear": (-0.5, 0.7, "紧张"),
    }
    return table.get(emotion.lower(), (0.0, 0.0, "中性"))


def _load_text_table(path: str):
    d = {}
    if not path or not os.path.isfile(path):
        return d
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("|", 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def collect_wavs(root: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".wav"):
                out.append(os.path.join(r, fn))
    return out


def infer_emotion_from_path(wav_path: str) -> str:
    low = wav_path.lower()
    for k, emo in EMO_HINTS.items():
        if k in low:
            return emo
    return "neutral"


def main():
    ap = argparse.ArgumentParser(description="Build train/val jsonl manifest from ESD only.")
    ap.add_argument("--esd_root", type=str, default="../dataset_esd_sorted")
    ap.add_argument("--text_table", type=str, default="", help="optional: utt_id|text")
    ap.add_argument("--out_train", type=str, default="data/manifests/train.jsonl")
    ap.add_argument("--out_val", type=str, default="data/manifests/val.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--default_text", type=str, default="这是一条用于情感可控语音合成训练的样本文本。")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means use all")
    args = ap.parse_args()

    random.seed(args.seed)
    txt = _load_text_table(args.text_table)

    all_wavs = collect_wavs(args.esd_root)
    if not all_wavs:
        raise RuntimeError(f"No wav found under esd_root: {os.path.abspath(args.esd_root)}")

    random.shuffle(all_wavs)
    if args.max_samples > 0:
        all_wavs = all_wavs[: args.max_samples]

    rows = []
    for wp in all_wavs:
        emo = infer_emotion_from_path(wp)
        valence, arousal, emo_cn = _label_to_va(emo)
        intensity = random.uniform(0.3, 0.9)
        style = random.choice(["自然", "温柔", "克制", "坚定", "低沉"])

        utt = os.path.splitext(os.path.basename(wp))[0]
        text = txt.get(utt, args.default_text)
        prompt = _auto_prompt(emo_cn, intensity, style)

        rows.append(
            {
                "text": text,
                "prompt": prompt,
                "emotion": emo_cn,
                "intensity": round(float(intensity), 3),
                "arousal": float(arousal),
                "valence": float(valence),
                "style": style,
                "wav_path": os.path.abspath(wp),
            }
        )

    n_val = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(args.out_val), exist_ok=True)
    with open(args.out_val, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"esd_root={os.path.abspath(args.esd_root)}")
    print(f"train={len(train_rows)}, val={len(val_rows)}")


if __name__ == "__main__":
    main()
