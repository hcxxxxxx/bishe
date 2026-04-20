import argparse
import json
import os
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fgemo_tts.config.schema import PromptCondition, TrainConfig
from fgemo_tts.data.dataset import JsonlPromptTTSDataset, collate_fn
from fgemo_tts.models.mock_backbone import MockTTSBackbone
from fgemo_tts.models.prompt_control_model import PromptControlledTTS
from fgemo_tts.models.rule_condition_encoder import RuleConditionEncoder
from fgemo_tts.utils.seed import set_seed


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_dist():
    if is_dist():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_main() -> bool:
    return get_rank() == 0


def make_neutral_conds(n: int):
    return [PromptCondition(emotion="中性", intensity=0.0, arousal=0.0, valence=0.0, style="自然") for _ in range(n)]


def evaluate(model, val_loader, device, ablation: str, rule_encoder=None):
    model.eval()
    total = 0.0
    cnt = 0
    with torch.no_grad():
        for text_tokens, acoustic, conds, _ in val_loader:
            text_tokens = text_tokens.to(device)
            acoustic = acoustic.to(device)

            if ablation == "none":
                conds_use = make_neutral_conds(len(conds))
                out = model(text_tokens, acoustic, conds_use)
            elif ablation == "rule_only":
                cond_vec = rule_encoder(conds).to(device)
                cond = model.module.adaptor(cond_vec) if isinstance(model, DDP) else model.adaptor(cond_vec)
                backbone = model.module.backbone if isinstance(model, DDP) else model.backbone
                out = backbone(text_tokens=text_tokens, acoustic=acoustic, cond=cond)
            else:
                out = model(text_tokens, acoustic, conds)

            total += float(out["loss"].item())
            cnt += 1
    model.train()
    return total / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--val_manifest", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="exp/fgemo")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--ablation", type=str, default="full", choices=["none", "rule_only", "full"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        ablation=args.ablation,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    init_dist()
    rank = get_rank()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")

    train_ds = JsonlPromptTTSDataset(cfg.train_manifest)
    val_ds = JsonlPromptTTSDataset(cfg.val_manifest)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_dist() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    backbone = MockTTSBackbone(hidden_dim=256)
    model = PromptControlledTTS(backbone=backbone, cond_dim=256, backbone_hidden_dim=512).to(device)
    rule_encoder = RuleConditionEncoder(out_dim=256)

    if cfg.ablation == "none":
        for p in model.prompt_encoder.parameters():
            p.requires_grad = False
        for p in model.adaptor.parameters():
            p.requires_grad = False

    if cfg.ablation == "rule_only":
        for p in model.prompt_encoder.parameters():
            p.requires_grad = False

    model_to_opt = model
    if is_dist():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False)

    params = [p for p in model_to_opt.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.output_dir, exist_ok=True)
    if is_main():
        with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    global_step = 0
    while global_step < cfg.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(global_step)

        for text_tokens, acoustic, conds, _ in train_loader:
            text_tokens = text_tokens.to(device)
            acoustic = acoustic.to(device)

            if cfg.ablation == "none":
                conds_use = make_neutral_conds(len(conds))
                out = model(text_tokens, acoustic, conds_use)
            elif cfg.ablation == "rule_only":
                cond_vec = rule_encoder(conds).to(device)
                cond = model.module.adaptor(cond_vec) if isinstance(model, DDP) else model.adaptor(cond_vec)
                backbone = model.module.backbone if isinstance(model, DDP) else model.backbone
                out = backbone(text_tokens=text_tokens, acoustic=acoustic, cond=cond)
            else:
                out = model(text_tokens, acoustic, conds)

            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1

            if is_main() and global_step % cfg.log_every == 0:
                print(f"[rank{rank}] step={global_step} loss={loss.item():.4f}")

            if global_step % cfg.eval_every == 0:
                val_loss = evaluate(model, val_loader, device, cfg.ablation, rule_encoder)
                if is_main():
                    print(f"[eval] step={global_step} val_loss={val_loss:.4f}")

            if is_main() and global_step % cfg.save_every == 0:
                core = model.module if isinstance(model, DDP) else model
                ckpt = {
                    "step": global_step,
                    "model": core.state_dict(),
                    "opt": opt.state_dict(),
                    "ablation": cfg.ablation,
                }
                torch.save(ckpt, os.path.join(cfg.output_dir, f"ckpt_{global_step}.pt"))

            if global_step >= cfg.max_steps:
                break

    if is_main():
        core = model.module if isinstance(model, DDP) else model
        torch.save({"step": global_step, "model": core.state_dict(), "ablation": cfg.ablation}, os.path.join(cfg.output_dir, "last.pt"))
        print(f"training finished: {cfg.output_dir}")

    cleanup_dist()


if __name__ == "__main__":
    main()
