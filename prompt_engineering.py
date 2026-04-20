"""Prompt engineering module for fine-grained emotion control.

Goals:
1) Emotion category control (happy/sad/angry/...)
2) Intensity control (slightly/moderately/very)
3) Compound emotion control (e.g. happy with a touch of sadness)
4) Context-aware style adaptation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


INTENSITY_ZH = {
    "slightly": "略微",
    "moderately": "适度",
    "very": "非常",
}

INTENSITY_EN = {
    "slightly": "slightly",
    "moderately": "moderately",
    "very": "very",
}


EMOTION_TEMPLATES = {
    "zh": {
        # IMPORTANT:
        # For CosyVoice2 inference_instruct2, instruct_text is consumed as prompt_text.
        # So do not include the target `text` itself in instruction; otherwise the
        # model may read instruction content.
        "neutral": "语气自然、平稳。",
        "happy": "语气高兴、明亮且有感染力。",
        "sad": "语气悲伤、低沉且克制。",
        "angry": "语气生气但清晰克制。",
        "surprised": "语气惊讶、语速略快且有变化。",
        "fearful": "语气紧张、谨慎并带有不安。",
        "gentle": "语气温柔、放松且亲切。",
        "serious": "语气严肃、庄重且清晰。",
    },
    "en": {
        "neutral": "Use a natural and steady tone.",
        "happy": "Use a cheerful, bright and engaging tone.",
        "sad": "Use a sad, low and restrained tone.",
        "angry": "Use an angry but controlled and clear tone.",
        "surprised": "Use a surprised tone with dynamic variation.",
        "fearful": "Use a tense and cautious tone.",
        "gentle": "Use a gentle, relaxed and warm tone.",
        "serious": "Use a serious, formal and clear tone.",
    },
}


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""

    language: str = "zh"
    use_context: bool = True
    optimize_prompt: bool = True


class EmotionPromptEngineer:
    """Build baseline and optimized natural language prompts."""

    def __init__(self, config: Optional[PromptConfig] = None) -> None:
        self.config = config or PromptConfig()

    @staticmethod
    def _normalize_emotion(emotion: str) -> str:
        alias = {
            "开心": "happy",
            "高兴": "happy",
            "伤心": "sad",
            "悲伤": "sad",
            "愤怒": "angry",
            "生气": "angry",
            "惊讶": "surprised",
            "害怕": "fearful",
            "恐惧": "fearful",
            "温柔": "gentle",
            "严肃": "serious",
            "中性": "neutral",
        }
        e = emotion.strip().lower()
        return alias.get(e, e)

    def build_baseline_prompt(self, text: str, emotion: str, language: Optional[str] = None) -> str:
        """Simple baseline prompt for A/B experiments."""
        lang = (language or self.config.language).lower()
        e = self._normalize_emotion(emotion)

        if lang == "zh":
            return f"语气为{emotion}。"
        return f"Use {e} emotion."

    def build_intensity_modifier(self, intensity: str, language: Optional[str] = None) -> str:
        """Convert intensity tag to natural language modifier."""
        lang = (language or self.config.language).lower()
        key = intensity.lower().strip()
        if key not in ("slightly", "moderately", "very"):
            key = "moderately"

        return INTENSITY_ZH[key] if lang == "zh" else INTENSITY_EN[key]

    def build_compound_emotion_clause(
        self,
        primary_emotion: str,
        secondary_emotion: Optional[str],
        intensity: str,
        language: Optional[str] = None,
    ) -> str:
        """Build compound emotion instructions."""
        lang = (language or self.config.language).lower()
        primary = self._normalize_emotion(primary_emotion)
        secondary = self._normalize_emotion(secondary_emotion) if secondary_emotion else None
        intensity_modifier = self.build_intensity_modifier(intensity, lang)

        if lang == "zh":
            if secondary:
                return f"整体情绪以{primary_emotion}为主，并{intensity_modifier}带有{secondary_emotion}。"
            return f"请表现出{intensity_modifier}的{primary_emotion}情绪。"

        if secondary:
            return (
                f"Keep {primary} as the dominant emotion, while being "
                f"{intensity_modifier} influenced by {secondary}."
            )
        return f"Express a {intensity_modifier} level of {primary} emotion."

    def build_context_clause(self, context: Optional[str], language: Optional[str] = None) -> str:
        """Inject context awareness into prompt."""
        if not context:
            return ""

        lang = (language or self.config.language).lower()
        if lang == "zh":
            return f"上下文场景：{context}。请让语气与场景一致。"
        return f"Context: {context}. Keep the prosody consistent with this context."

    def optimize_prompt(self, prompt: str, language: Optional[str] = None) -> str:
        """Prompt refinement rules for stronger emotional control."""
        lang = (language or self.config.language).lower()

        if lang == "zh":
            suffix = " 保持吐字清晰、停连自然，并避免情绪过度夸张。"
        else:
            suffix = " Keep articulation clear, pauses natural, and avoid exaggerated acting."

        if prompt.endswith(("。", ".")):
            return prompt + suffix.strip()
        return prompt + "." + suffix

    def build_optimized_prompt(
        self,
        text: str,
        primary_emotion: str,
        intensity: str = "moderately",
        secondary_emotion: Optional[str] = None,
        context: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """Build a multi-layer optimized natural language prompt."""
        lang = (language or self.config.language).lower()
        primary = self._normalize_emotion(primary_emotion)

        base = EMOTION_TEMPLATES.get(lang, EMOTION_TEMPLATES["zh"]).get(
            primary,
            EMOTION_TEMPLATES[lang]["neutral"],
        )
        # Keep instruction concise and style-only for CosyVoice2 instruct2.
        base_clause = base
        compound_clause = self.build_compound_emotion_clause(
            primary_emotion=primary_emotion,
            secondary_emotion=secondary_emotion,
            intensity=intensity,
            language=lang,
        )
        context_clause = self.build_context_clause(context=context, language=lang)

        full_prompt = " ".join(part for part in [compound_clause, context_clause, base_clause] if part)
        if self.config.optimize_prompt:
            full_prompt = self.optimize_prompt(full_prompt, language=lang)
        return full_prompt

    def build_prompt_pair(
        self,
        text: str,
        primary_emotion: str,
        intensity: str = "moderately",
        secondary_emotion: Optional[str] = None,
        context: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Return (baseline_prompt, optimized_prompt) for comparison studies."""
        baseline = self.build_baseline_prompt(text=text, emotion=primary_emotion, language=language)
        optimized = self.build_optimized_prompt(
            text=text,
            primary_emotion=primary_emotion,
            intensity=intensity,
            secondary_emotion=secondary_emotion,
            context=context,
            language=language,
        )
        return baseline, optimized
