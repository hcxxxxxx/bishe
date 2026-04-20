import re
from typing import Dict

from fgemo_tts.config.schema import PromptCondition


EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
    "高兴": {"valence": 0.8, "arousal": 0.7},
    "开心": {"valence": 0.75, "arousal": 0.65},
    "悲伤": {"valence": -0.75, "arousal": -0.35},
    "难过": {"valence": -0.7, "arousal": -0.3},
    "温柔": {"valence": 0.35, "arousal": -0.25},
    "平静": {"valence": 0.1, "arousal": -0.55},
    "愤怒": {"valence": -0.85, "arousal": 0.9},
    "严肃": {"valence": -0.1, "arousal": 0.1},
    "紧张": {"valence": -0.4, "arousal": 0.8},
    "兴奋": {"valence": 0.65, "arousal": 0.9},
}

INTENSITY_LEXICON = {
    "略带": 0.35,
    "稍微": 0.3,
    "有点": 0.25,
    "比较": 0.55,
    "明显": 0.7,
    "非常": 0.9,
    "特别": 0.95,
}

STYLE_WORDS = ["温柔", "克制", "坚定", "轻快", "低沉", "柔和", "自然"]


class RulePromptParser:
    def parse(self, prompt: str) -> PromptCondition:
        norm = re.sub(r"\s+", "", prompt)
        hit_emotion = "中性"
        valence, arousal = 0.0, 0.0

        for k, v in EMOTION_LEXICON.items():
            if k in norm:
                hit_emotion = k
                valence = v["valence"]
                arousal = v["arousal"]
                break

        intensity = 0.5
        for k, v in INTENSITY_LEXICON.items():
            if k in norm:
                intensity = v
                break

        style = "自然"
        for s in STYLE_WORDS:
            if s in norm:
                style = s
                break

        return PromptCondition(
            emotion=hit_emotion,
            intensity=float(intensity),
            arousal=float(arousal),
            valence=float(valence),
            style=style,
        )
