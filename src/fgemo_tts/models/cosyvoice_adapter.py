import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from fgemo_tts.models.backbone_interface import TTSBackboneBase


@dataclass
class CosyVoiceTrainArgs:
    model: str
    config: str
    train_data: str
    cv_data: str
    qwen_pretrain_path: str
    onnx_path: str
    checkpoint: str
    model_dir: str
    tensorboard_dir: str
    train_engine: str = "torch_ddp"
    dist_backend: str = "nccl"
    num_workers: int = 8
    prefetch: int = 100
    use_amp: bool = True
    deepspeed_config: str = ""
    save_states: str = "model+optimizer"


class CosyVoice2BackboneAdapter(TTSBackboneBase):
    """
    Real adapter for local CosyVoice2 repo.

    - Inference: wraps CosyVoice AutoModel API.
    - Training: wraps official cosyvoice/bin/train.py command.
    """

    def __init__(
        self,
        cosyvoice_root: str = "../CosyVoice",
        model_dir: str = "../CosyVoice/pretrained_models/CosyVoice2-0.5B",
        load_for_infer: bool = True,
        load_jit: bool = False,
        load_trt: bool = False,
        load_vllm: bool = False,
        fp16: bool = False,
    ):
        super().__init__()
        self.cosyvoice_root = os.path.abspath(cosyvoice_root)
        self.model_dir = os.path.abspath(model_dir)

        third_party_matcha = os.path.join(self.cosyvoice_root, "third_party", "Matcha-TTS")
        for p in [self.cosyvoice_root, third_party_matcha]:
            if p not in sys.path:
                sys.path.insert(0, p)

        self._auto_model = None
        self._model = None
        if load_for_infer:
            from cosyvoice.cli.cosyvoice import AutoModel  # pylint: disable=import-error

            self._auto_model = AutoModel
            self._model = self._auto_model(
                model_dir=self.model_dir,
                load_jit=load_jit,
                load_trt=load_trt,
                load_vllm=load_vllm,
                fp16=fp16,
            )

    @property
    def sample_rate(self) -> int:
        if self._model is None:
            raise RuntimeError("CosyVoice model is not loaded. Recreate adapter with load_for_infer=True.")
        return int(self._model.sample_rate)

    def _collect_chunks(self, gen: Iterable[Dict[str, torch.Tensor]]) -> torch.Tensor:
        chunks = []
        for out in gen:
            wav = out["tts_speech"]
            if wav.dim() == 2:
                wav = wav.squeeze(0)
            chunks.append(wav.detach().cpu())
        if not chunks:
            raise RuntimeError("CosyVoice inference returned no audio chunk.")
        return torch.cat(chunks, dim=-1)

    @torch.inference_mode()
    def infer_instruct2(self, text: str, instruct_text: str, prompt_wav: str, stream: bool = False, speed: float = 1.0) -> torch.Tensor:
        gen = self._model.inference_instruct2(
            tts_text=text,
            instruct_text=instruct_text,
            prompt_wav=prompt_wav,
            stream=stream,
            speed=speed,
            text_frontend=False,
        )
        return self._collect_chunks(gen)

    @torch.inference_mode()
    def infer_zero_shot(self, text: str, prompt_text: str, prompt_wav: str, stream: bool = False, speed: float = 1.0) -> torch.Tensor:
        gen = self._model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            stream=stream,
            speed=speed,
            text_frontend=False,
        )
        return self._collect_chunks(gen)

    def forward(self, text_tokens: torch.Tensor, acoustic: torch.Tensor, cond: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "CosyVoice2 training is launched via launch_train_cmd() using official cosyvoice/bin/train.py, "
            "instead of calling a direct forward() in this wrapper."
        )

    def infer(self, text: str, cond: Dict[str, torch.Tensor], speaker_wav: str = "") -> torch.Tensor:
        instruct_text = cond.get("instruct_text", "")
        prompt_text = cond.get("prompt_text", "")
        speed = float(cond.get("speed", 1.0))

        if instruct_text and speaker_wav:
            return self.infer_instruct2(text=text, instruct_text=instruct_text, prompt_wav=speaker_wav, speed=speed)
        if speaker_wav:
            return self.infer_zero_shot(text=text, prompt_text=prompt_text, prompt_wav=speaker_wav, speed=speed)
        raise ValueError("speaker_wav is required for CosyVoice2 zero-shot/instruct2 inference.")

    def launch_train_cmd(
        self,
        train_args: CosyVoiceTrainArgs,
        nproc_per_node: int = 8,
        master_port: int = 29511,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        train_py = os.path.join(self.cosyvoice_root, "cosyvoice", "bin", "train.py")
        if not os.path.isfile(train_py):
            raise FileNotFoundError(f"CosyVoice train.py not found: {train_py}")

        cmd = [
            "torchrun",
            "--nnodes=1",
            f"--nproc_per_node={nproc_per_node}",
            f"--rdzv_id={random.randint(1000, 9999)}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint=localhost:{master_port}",
            train_py,
            "--train_engine",
            train_args.train_engine,
            "--config",
            train_args.config,
            "--train_data",
            train_args.train_data,
            "--cv_data",
            train_args.cv_data,
            "--qwen_pretrain_path",
            train_args.qwen_pretrain_path,
            "--onnx_path",
            train_args.onnx_path,
            "--model",
            train_args.model,
            "--checkpoint",
            train_args.checkpoint,
            "--model_dir",
            train_args.model_dir,
            "--tensorboard_dir",
            train_args.tensorboard_dir,
            "--ddp.dist_backend",
            train_args.dist_backend,
            "--num_workers",
            str(train_args.num_workers),
            "--prefetch",
            str(train_args.prefetch),
            "--pin_memory",
        ]

        if train_args.use_amp:
            cmd.append("--use_amp")

        if train_args.train_engine == "deepspeed":
            if not train_args.deepspeed_config:
                raise ValueError("deepspeed_config is required when train_engine=deepspeed")
            cmd.extend([
                "--deepspeed_config",
                train_args.deepspeed_config,
                "--deepspeed.save_states",
                train_args.save_states,
            ])

        print("[CosyVoice train cmd]", " ".join(cmd))
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        Path(train_args.model_dir).mkdir(parents=True, exist_ok=True)
        Path(train_args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, env=run_env, check=True)
