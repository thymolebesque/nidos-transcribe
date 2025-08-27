from __future__ import annotations
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from typing import Optional
import numpy as np

from app.schemas import EnrollResponse
from app.config import settings
from app.services.io_utils import load_audio
from app.services.vad import detect_voiced_segments
from app.services.embeddings import embed_segments, mean_pool, save_coach_embedding

router = APIRouter()

@router.post("/enroll", response_model=EnrollResponse)
async def enroll_coach(
    file: UploadFile = File(...),
    speaker_name: str = Form("COACH"),
):
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Please upload a WAV file.")
    data = await file.read()
    wav, sr = load_audio(data, target_sr=settings.SAMPLE_RATE, mono=True)
    duration = len(wav) / sr
    segments = detect_voiced_segments(
        wav, sr,
        frame_ms=settings.VAD_FRAME_MS,
        aggressiveness=2,
        min_seg_dur=settings.MIN_SEG_DUR,
        merge_gap=settings.MERGE_GAP,
    )
    if not segments:
        raise HTTPException(status_code=400, detail="No voiced segments detected in enrollment audio.")
    embs = embed_segments(wav, sr, segments)
    coach_emb = mean_pool(embs)

    save_coach_embedding(coach_emb, sr=sr, name=speaker_name, duration=duration)
    return EnrollResponse(
        speaker=speaker_name,
        duration_sec=float(duration),
        embedding_dim=int(coach_emb.shape[0]),
        saved=True,
    )
