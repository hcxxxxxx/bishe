import argparse
import os

import soundfile as sf
import torch

from fgemo_tts.models.mock_backbone import MockTTSBackbone
from fgemo_tts.models.prompt_control_model import PromptControlledTTS
from fgemo_tts.models.prompt_parser import RulePromptParser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--speaker_wav", type=str, default="")
    ap.add_argument("--out_wav", type=str, default="demo_out.wav")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PromptControlledTTS(backbone=MockTTSBackbone(hidden_dim=256), cond_dim=256, backbone_hidden_dim=512).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    parser = RulePromptParser()
    cond = parser.parse(args.prompt)

    wav = model.infer(text=args.text, cond=cond, speaker_wav=args.speaker_wav)
    wav_np = wav.detach().float().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_wav)), exist_ok=True)
    sf.write(args.out_wav, wav_np, 16000)
    print(f"saved: {args.out_wav}")


if __name__ == "__main__":
    main()
