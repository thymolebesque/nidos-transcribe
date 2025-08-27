from __future__ import annotations
from typing import Dict, List, Optional

import os
import torch

from faster_whisper import WhisperModel

from app.config import settings

_model: Optional[WhisperModel] = None
_model_name_display: Optional[str] = None

def _assert_whisper_model_local():
    local_dir = settings.whisper_model_path
    if not local_dir.exists() and settings.OFFLINE_ONLY:
        raise RuntimeError(
            f"Faster-Whisper model not found at '{local_dir}'. "
            "Place a local CTranslate2 model dir there (see README), "
            "or disable OFFLINE_ONLY to allow an initial download."
        )

def get_model() -> WhisperModel:
    global _model, _model_name_display
    if _model is not None:
        return _model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Prefer local path for offline
    _assert_whisper_model_local()
    model_path = str(settings.whisper_model_path) if settings.whisper_model_path.exists() else settings.WHISPER_MODEL
    _model = WhisperModel(model_path, device=device, compute_type=compute_type)
    _model_name_display = f"faster-whisper {settings.WHISPER_MODEL}"
    return _model

def transcribe(
    audio_path: str,
    language: str = "nl",
    word_timestamps: bool = True,
) -> List[Dict]:
    """
    Returns list of segments:
    {
      "start": float, "end": float, "text": str,
      "words": [{"word": str, "start": float, "end": float}, ...]  # if available
    }
    """
    model = get_model()

    # We do not apply Faster-Whisper’s built-in VAD filter; our VAD is separate.
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        word_timestamps=word_timestamps,
        beam_size=5,
        vad_filter=False,
    )

    out = []
    for seg in segments:
        item = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        }
        if seg.words:
            words = []
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
            item["words"] = words
        out.append(item)
    return out

def model_name_display() -> str:
    return _model_name_display or f"faster-whisper {settings.WHISPER_MODEL}"
