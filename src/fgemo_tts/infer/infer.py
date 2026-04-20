import argparse
import os

import soundfile as sf

from fgemo_tts.models.cosyvoice_adapter import CosyVoice2BackboneAdapter
from fgemo_tts.models.prompt_parser import RulePromptParser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice_root", type=str, default="../CosyVoice")
    ap.add_argument("--model_dir", type=str, default="../CosyVoice/pretrained_models/CosyVoice2-0.5B")
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True, help="Natural language emotion prompt")
    ap.add_argument("--speaker_wav", type=str, required=True, help="Reference wav for zero-shot/instruct2")
    ap.add_argument("--mode", type=str, default="instruct2", choices=["instruct2", "zero_shot"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out_wav", type=str, default="exp/demo/demo_prompt.wav")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_wav)), exist_ok=True)

    adapter = CosyVoice2BackboneAdapter(
        cosyvoice_root=args.cosyvoice_root,
        model_dir=args.model_dir,
        load_for_infer=True,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
    )

    parser = RulePromptParser()
    cond = parser.parse(args.prompt)

    if args.mode == "instruct2":
        instruct_text = f"{args.prompt}<|endofprompt|>"
        wav = adapter.infer(
            text=args.text,
            cond={"instruct_text": instruct_text, "speed": args.speed},
            speaker_wav=args.speaker_wav,
        )
    else:
        prompt_text = f"请用{cond.emotion}且{cond.style}的语气说这句话。"
        wav = adapter.infer(
            text=args.text,
            cond={"prompt_text": prompt_text, "speed": args.speed},
            speaker_wav=args.speaker_wav,
        )

    sf.write(args.out_wav, wav.numpy(), adapter.sample_rate)
    print(f"saved: {args.out_wav}")


if __name__ == "__main__":
    main()
