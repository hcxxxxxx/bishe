import argparse
import glob
import os

from fgemo_tts.models.cosyvoice_adapter import CosyVoice2BackboneAdapter, CosyVoiceTrainArgs


def resolve_checkpoint(model_dir: str, model: str) -> str:
    if model == "hifigan":
        cands = [os.path.join(model_dir, "hifigan.pt"), os.path.join(model_dir, "hift.pt")]
    else:
        cands = [os.path.join(model_dir, f"{model}.pt")]
    for p in cands:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No checkpoint found for model={model}, candidates={cands}")


def main():
    ap = argparse.ArgumentParser(description="Launch real CosyVoice2 training via official train.py")
    ap.add_argument("--cosyvoice_root", type=str, default="../CosyVoice")
    ap.add_argument("--cosyvoice_model_dir", type=str, default="../CosyVoice/pretrained_models/CosyVoice2-0.5B")
    ap.add_argument("--ablation", type=str, default="full", choices=["none", "rule_only", "full"])
    ap.add_argument("--models", type=str, default="llm,flow", help="comma-separated subset of llm,flow,hifigan")
    ap.add_argument("--config", type=str, default="../CosyVoice/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml")
    ap.add_argument("--train_data", type=str, default="")
    ap.add_argument("--cv_data", type=str, default="")
    ap.add_argument("--data_root", type=str, default="data/cosyvoice_esd")
    ap.add_argument("--exp_root", type=str, default="exp/cosyvoice_esd")
    ap.add_argument("--tensorboard_root", type=str, default="tensorboard/cosyvoice_esd")
    ap.add_argument("--train_engine", type=str, default="torch_ddp", choices=["torch_ddp", "deepspeed"])
    ap.add_argument("--deepspeed_config", type=str, default="../CosyVoice/examples/libritts/cosyvoice2/conf/ds_stage2.json")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=100)
    ap.add_argument("--nproc_per_node", type=int, default=8)
    ap.add_argument("--master_port", type=int, default=29511)
    ap.add_argument("--cuda_visible_devices", type=str, default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    cosyvoice_root = os.path.abspath(args.cosyvoice_root)
    cosyvoice_model_dir = os.path.abspath(args.cosyvoice_model_dir)
    config = os.path.abspath(args.config)

    if not os.path.isdir(cosyvoice_root):
        raise FileNotFoundError(f"cosyvoice_root not found: {cosyvoice_root}")
    if not os.path.isdir(cosyvoice_model_dir):
        raise FileNotFoundError(f"cosyvoice_model_dir not found: {cosyvoice_model_dir}")
    if not os.path.isfile(config):
        raise FileNotFoundError(f"config not found: {config}")

    if args.train_data:
        train_data = os.path.abspath(args.train_data)
    else:
        train_data = os.path.abspath(os.path.join(args.data_root, args.ablation, "train", "parquet", "data.list"))
    if args.cv_data:
        cv_data = os.path.abspath(args.cv_data)
    else:
        cv_data = os.path.abspath(os.path.join(args.data_root, args.ablation, "dev", "parquet", "data.list"))

    if not os.path.isfile(train_data):
        raise FileNotFoundError(f"train_data list not found: {train_data}")
    if not os.path.isfile(cv_data):
        raise FileNotFoundError(f"cv_data list not found: {cv_data}")

    qwen_pretrain_path = os.path.join(cosyvoice_model_dir, "CosyVoice-BlankEN")
    if not os.path.isdir(qwen_pretrain_path):
        raise FileNotFoundError(
            f"qwen pretrain dir missing: {qwen_pretrain_path}. "
            "Please download complete CosyVoice2-0.5B pretrained package first."
        )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid = {"llm", "flow", "hifigan"}
    for m in models:
        if m not in valid:
            raise ValueError(f"unsupported model: {m}, expected one of {sorted(valid)}")

    adapter = CosyVoice2BackboneAdapter(
        cosyvoice_root=cosyvoice_root,
        model_dir=cosyvoice_model_dir,
        load_for_infer=False,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
    )

    env = {
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "PYTHONPATH": f"{cosyvoice_root}:{os.path.join(cosyvoice_root, 'third_party', 'Matcha-TTS')}:{os.environ.get('PYTHONPATH', '')}",
    }

    for idx, model in enumerate(models):
        ckpt = resolve_checkpoint(cosyvoice_model_dir, model)
        out_dir = os.path.abspath(os.path.join(args.exp_root, args.ablation, model, args.train_engine))
        tb_dir = os.path.abspath(os.path.join(args.tensorboard_root, args.ablation, model, args.train_engine))

        train_args = CosyVoiceTrainArgs(
            model=model,
            config=config,
            train_data=train_data,
            cv_data=cv_data,
            qwen_pretrain_path=qwen_pretrain_path,
            onnx_path=cosyvoice_model_dir,
            checkpoint=ckpt,
            model_dir=out_dir,
            tensorboard_dir=tb_dir,
            train_engine=args.train_engine,
            num_workers=args.num_workers,
            prefetch=args.prefetch,
            use_amp=True,
            deepspeed_config=os.path.abspath(args.deepspeed_config),
        )

        adapter.launch_train_cmd(
            train_args=train_args,
            nproc_per_node=args.nproc_per_node,
            master_port=args.master_port + idx,
            env=env,
        )

        snapshots = sorted(glob.glob(os.path.join(out_dir, "epoch_*_whole.pt")))
        if snapshots:
            print(f"latest checkpoint for {model}: {snapshots[-1]}")


if __name__ == "__main__":
    main()
