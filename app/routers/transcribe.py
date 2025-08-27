from __future__ import annotations
import uuid
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from typing import Optional

from app.schemas import TranscribeResponse, Speaker, Utterance, Metrics
from app.config import settings
from app.services.io_utils import load_audio, save_temp_wav
from app.services.vad import detect_voiced_segments
from app.services.embeddings import embed_segments, load_coach_embedding
from app.services.diarization import diarize
from app.services.asr import transcribe as asr_transcribe, model_name_display
from app.services.align import assign_speakers_to_words
from app.utils import stopwatch

router = APIRouter()

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default=settings.LANGUAGE),
    coach_threshold: float = Form(default=settings.COACH_THRESHOLD),
    max_speakers: int = Form(default=2),
    use_word_timestamps: bool = Form(default=True),
):
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Please upload a WAV file.")
    data = await file.read()

    with stopwatch() as t0:
        # Load audio (ensure mono/16k)
        wav, sr = load_audio(data, target_sr=settings.SAMPLE_RATE, mono=True)

        # VAD → segments
        segments = detect_voiced_segments(
            wav, sr,
            frame_ms=settings.VAD_FRAME_MS,
            aggressiveness=2,
            min_seg_dur=settings.MIN_SEG_DUR,
            merge_gap=settings.MERGE_GAP,
        )

        # Embeddings for segments
        embs = embed_segments(wav, sr, segments)

        # Load coach embedding, diarize with constrained labeling
        coach_emb = load_coach_embedding("COACH")
        diar = diarize(
            segments=segments,
            embs=embs,
            coach_emb=coach_emb,
            thr=coach_threshold,
            max_speakers=max_speakers,
        )

        # Run ASR (Faster-Whisper). Use a temp WAV path (more efficient).
        tmp = save_temp_wav(data)
        try:
            asr_segments = asr_transcribe(
                audio_path=str(tmp),
                language=language,
                word_timestamps=bool(use_word_timestamps),
            )
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

        # Align diarization to ASR words/segments
        utterances_dicts = assign_speakers_to_words(
            diar_segments=diar,
            whisper_segments=asr_segments,
            merge_gap=settings.MERGE_GAP,
            min_turn_dur=settings.MIN_SEG_DUR,
        )

        # Build response
        utterances = [
            Utterance(
                start=u["start"], end=u["end"], speaker=u["speaker"], text=u["text"],
                words=[{"w": w["w"], "start": w["start"], "end": w["end"], "speaker": w["speaker"]} for w in u.get("words", [])]
            )
            for u in utterances_dicts
        ]

        # Speakers list: keep COACH and JONGERE for UI
        speakers = [
            Speaker(id="COACH", display="Coach"),
            Speaker(id="JONGERE", display="Jongere"),
        ]

        processing_sec = float(__import__("time").time() - t0)

        resp = TranscribeResponse(
            session_id=str(uuid.uuid4()),
            language=language,
            speakers=speakers,
            utterances=utterances,
            metrics=Metrics(processing_sec=processing_sec, model=model_name_display()),
        )
        return resp
