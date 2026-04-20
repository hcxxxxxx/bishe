#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

EMO_ZH = {
    "neutral": "中性",
    "happy": "高兴",
    "sad": "悲伤",
    "angry": "愤怒",
    "surprise": "惊喜",
}

STYLE_BANK = ["自然", "温柔", "克制", "坚定", "低沉", "轻快"]


def load_transcriptions(esd_root: str) -> Dict[str, str]:
    trans_dir = os.path.join(esd_root, "transcription")
    utt2text: Dict[str, str] = {}
    for fn in os.listdir(trans_dir):
        if not fn.endswith(".txt"):
            continue
        p = os.path.join(trans_dir, fn)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split("\t")
                if len(parts) < 2:
                    continue
                utt = parts[0].strip()
                txt = parts[1].strip()
                utt2text[utt] = txt
    return utt2text


def build_prompt_text(base_text: str, emotion: str, mode: str, rng: random.Random) -> str:
    emo_zh = EMO_ZH.get(emotion, "中性")
    if mode == "none":
        return base_text
    if mode == "rule_only":
        return f"请用{emo_zh}的语气说：{base_text}"
    intensity = rng.choice(["略带", "比较", "非常"])
    style = rng.choice(STYLE_BANK)
    return f"请用{intensity}{emo_zh}且{style}的语气说：{base_text}"


def collect_samples(esd_root: str, utt2text: Dict[str, str]) -> List[Tuple[str, str, str, str, str]]:
    # (utt, wav_path, spk, emotion, text)
    out: List[Tuple[str, str, str, str, str]] = []
    emotions = ["neutral", "happy", "sad", "angry", "surprise"]
    for emo in emotions:
        emo_dir = os.path.join(esd_root, emo)
        if not os.path.isdir(emo_dir):
            continue
        for spk in sorted(os.listdir(emo_dir)):
            spk_dir = os.path.join(emo_dir, spk)
            if not os.path.isdir(spk_dir):
                continue
            for fn in os.listdir(spk_dir):
                if not fn.lower().endswith(".wav"):
                    continue
                utt = os.path.splitext(fn)[0]
                if utt not in utt2text:
                    continue
                wav = os.path.abspath(os.path.join(spk_dir, fn))
                out.append((utt, wav, spk, emo, utt2text[utt]))
    return out


def split_samples(rows: List[Tuple[str, str, str, str, str]], val_ratio: float, seed: int):
    rnd = random.Random(seed)
    rows = rows[:]
    rnd.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]


def write_kaldi_dir(rows: List[Tuple[str, str, str, str, str]], out_dir: str, mode: str, seed: int):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    wav_scp = os.path.join(out_dir, "wav.scp")
    text_f = os.path.join(out_dir, "text")
    utt2spk_f = os.path.join(out_dir, "utt2spk")
    spk2utt_f = os.path.join(out_dir, "spk2utt")

    spk2utts = defaultdict(list)
    rng = random.Random(seed)

    with open(wav_scp, "w", encoding="utf-8") as fw, open(text_f, "w", encoding="utf-8") as ft, open(utt2spk_f, "w", encoding="utf-8") as fu:
        for idx, (utt, wav, spk, emo, txt) in enumerate(rows):
            prompt_text = build_prompt_text(txt, emo, mode, random.Random(rng.randint(0, 10**9) + idx))
            fw.write(f"{utt} {wav}\n")
            ft.write(f"{utt} {prompt_text}\n")
            fu.write(f"{utt} spk_{spk}\n")
            spk2utts[f"spk_{spk}"].append(utt)

    with open(spk2utt_f, "w", encoding="utf-8") as fs:
        for spk, utts in sorted(spk2utts.items()):
            fs.write(f"{spk} {' '.join(utts)}\n")


def make_parquet(cosyvoice_root: str, src_dir: str, num_utts_per_parquet: int, num_processes: int):
    parquet_dir = os.path.join(src_dir, "parquet")
    Path(parquet_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        os.path.join(cosyvoice_root, "tools", "make_parquet_list.py"),
        "--num_utts_per_parquet",
        str(num_utts_per_parquet),
        "--num_processes",
        str(num_processes),
        "--src_dir",
        src_dir,
        "--des_dir",
        parquet_dir,
    ]
    print("[make_parquet]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esd_root", type=str, default="../dataset_esd_sorted")
    ap.add_argument("--cosyvoice_root", type=str, default="../CosyVoice")
    ap.add_argument("--out_root", type=str, default="data/cosyvoice_esd")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_utts_per_parquet", type=int, default=1000)
    ap.add_argument("--num_processes", type=int, default=8)
    args = ap.parse_args()

    esd_root = os.path.abspath(args.esd_root)
    cosyvoice_root = os.path.abspath(args.cosyvoice_root)
    out_root = os.path.abspath(args.out_root)

    utt2text = load_transcriptions(esd_root)
    rows = collect_samples(esd_root, utt2text)
    if not rows:
        raise RuntimeError(f"No usable ESD wav found under: {esd_root}")

    train_rows, dev_rows = split_samples(rows, args.val_ratio, args.seed)
    print(f"total={len(rows)}, train={len(train_rows)}, dev={len(dev_rows)}")

    for mode in ["none", "rule_only", "full"]:
        train_dir = os.path.join(out_root, mode, "train")
        dev_dir = os.path.join(out_root, mode, "dev")

        write_kaldi_dir(train_rows, train_dir, mode=mode, seed=args.seed)
        write_kaldi_dir(dev_rows, dev_dir, mode=mode, seed=args.seed + 1)

        make_parquet(cosyvoice_root, train_dir, args.num_utts_per_parquet, args.num_processes)
        make_parquet(cosyvoice_root, dev_dir, args.num_utts_per_parquet, args.num_processes)

    print(f"prepared: {out_root}")


if __name__ == "__main__":
    main()
