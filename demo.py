"""Gradio demo for fine-grained emotion-controllable TTS."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import gradio as gr

from inference_pipeline import EmotionTTSPipeline, SynthesisRequest
from model_loader import get_default_loader


PIPELINE: Optional[EmotionTTSPipeline] = None


def get_pipeline() -> EmotionTTSPipeline:
    global PIPELINE
    if PIPELINE is None:
        loader = get_default_loader(model_dir="./models/CosyVoice2-0.5B")
        loader.load_model(force_download=False)
        PIPELINE = EmotionTTSPipeline(model_loader=loader, output_dir="./demo_outputs")
    return PIPELINE


def run_tts(
    text: str,
    language: str,
    primary_emotion: str,
    intensity: str,
    secondary_emotion: str,
    context: str,
    prompt_audio: str | None,
    spk_id: str,
    prompt_mode: str,
):
    if not text.strip():
        raise gr.Error("请输入文本 / Please input text.")

    req = SynthesisRequest(
        text=text.strip(),
        primary_emotion=primary_emotion.strip() or "neutral",
        intensity=intensity,
        secondary_emotion=secondary_emotion.strip() or None,
        context=context.strip() or None,
        language=language,
        use_optimized_prompt=(prompt_mode == "optimized"),
        prompt_audio_path=prompt_audio or None,
        spk_id=spk_id.strip() or None,
    )

    pipeline = get_pipeline()
    result = pipeline.synthesize_single(req)

    show = {
        "audio_path": result.audio_path,
        "used_prompt": result.used_prompt,
        "request": result.request,
        "model_repo": result.model_repo,
    }
    return result.audio_path, json.dumps(show, ensure_ascii=False, indent=2)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Fine-grained Emotion TTS Demo") as demo:
        gr.Markdown("# 基于自然语言提示的细粒度情感可控 TTS Demo")
        gr.Markdown("支持中英文情感控制、强度调节、复合情感与上下文感知。")

        with gr.Row():
            text = gr.Textbox(
                label="输入文本 / Text",
                value="收到这份礼物时，我真的很开心，但也有一点点想念远方的朋友。",
                lines=4,
            )

        with gr.Row():
            language = gr.Dropdown(["zh", "en"], value="zh", label="语言")
            primary_emotion = gr.Dropdown(
                ["neutral", "happy", "sad", "angry", "surprised", "fearful", "gentle", "serious"],
                value="happy",
                label="主情感",
            )
            intensity = gr.Dropdown(["slightly", "moderately", "very"], value="moderately", label="强度")

        with gr.Row():
            secondary_emotion = gr.Textbox(label="次情感（可选）", value="sad")
            context = gr.Textbox(label="上下文（可选）", value="生日后独自回家时")
            spk_id = gr.Textbox(label="spk_id（可选，SFT兜底）", value="")
            prompt_mode = gr.Dropdown(["baseline", "optimized"], value="optimized", label="Prompt版本")

        prompt_audio = gr.Audio(label="参考音频（推荐，CosyVoice2 instruct2）", type="filepath")
        btn = gr.Button("生成语音")
        audio = gr.Audio(label="生成音频", type="filepath")
        info = gr.Code(label="元数据", language="json")

        btn.click(
            fn=run_tts,
            inputs=[text, language, primary_emotion, intensity, secondary_emotion, context, prompt_audio, spk_id, prompt_mode],
            outputs=[audio, info],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="0.0.0.0", server_port=7860)
