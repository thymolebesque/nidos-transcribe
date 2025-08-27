#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

from app.config import settings
from app.services.io_utils import load_audio
from app.services.vad import detect_voiced_segments
from app.services.embeddings import embed_segments, mean_pool, save_coach_embedding

def main():
    ap = argparse.ArgumentParser(description="Enroll coach voice from WAV")
    ap.add_argument("--wav", required=True, help="Path to coach reference WAV (30-60s)")
    ap.add_argument("--name", default="COACH", help="Speaker name key (default: COACH)")
    args = ap.parse_args()

    wav_path = Path(args.wav)
    wav, sr = load_audio(wav_path, target_sr=settings.SAMPLE_RATE, mono=True)
    duration = len(wav) / sr
    print(f"[enroll] loaded {wav_path} duration={duration:.2f}s sr={sr}")

    segments = detect_voiced_segments(
        wav, sr,
        frame_ms=settings.VAD_FRAME_MS,
        aggressiveness=2,
        min_seg_dur=settings.MIN_SEG_DUR,
        merge_gap=settings.MERGE_GAP,
    )
    print(f"[enroll] detected {len(segments)} voiced segments")

    embs = embed_segments(wav, sr, segments)
    coach_emb = mean_pool(embs)
    save_coach_embedding(coach_emb, sr=sr, name=args.name, duration=duration)
    print(f"[enroll] saved embedding as '{args.name}' (dim={coach_emb.shape[0]}) at {settings.SPEAKER_DB_PATH}")

if __name__ == "__main__":
    main()
