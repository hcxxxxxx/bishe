#!/usr/bin/env python3
import argparse
import glob
import os
import shutil


def latest_ckpt(model_exp_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(model_exp_dir, "epoch_*_whole.pt")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(model_exp_dir, "*.pt")))
    if not cands:
        raise FileNotFoundError(f"No checkpoint found in: {model_exp_dir}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", type=str, default="../CosyVoice/pretrained_models/CosyVoice2-0.5B")
    ap.add_argument("--exp_root", type=str, required=True, help="e.g. exp/cosyvoice_esd/full")
    ap.add_argument("--output_model_dir", type=str, required=True)
    ap.add_argument("--train_engine", type=str, default="torch_ddp")
    ap.add_argument("--use_llm", action="store_true", default=True)
    ap.add_argument("--use_flow", action="store_true", default=True)
    args = ap.parse_args()

    base = os.path.abspath(args.base_model_dir)
    out = os.path.abspath(args.output_model_dir)
    exp = os.path.abspath(args.exp_root)

    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(base, out)

    llm_exp = os.path.join(exp, "llm", args.train_engine)
    flow_exp = os.path.join(exp, "flow", args.train_engine)

    if args.use_llm and os.path.isdir(llm_exp):
        src = latest_ckpt(llm_exp)
        shutil.copy2(src, os.path.join(out, "llm.pt"))
        print(f"llm <- {src}")

    if args.use_flow and os.path.isdir(flow_exp):
        src = latest_ckpt(flow_exp)
        shutil.copy2(src, os.path.join(out, "flow.pt"))
        print(f"flow <- {src}")

    print(f"assembled model dir: {out}")


if __name__ == "__main__":
    main()
