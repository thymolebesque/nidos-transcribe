#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import uuid
from pathlib import Path
import time

from app.config import settings
from app.services.io_utils import load_audio, save_temp_wav
from app.services.vad import detect_voiced_segments
from app.services.embeddings import embed_segments, load_coach_embedding
from app.services.diarization import diarize
from app.services.asr import transcribe as asr_transcribe, model_name_display
from app.services.align import assign_speakers_to_words

def main():
    ap = argparse.ArgumentParser(description="Run diarization + transcription on a WAV")
    ap.add_argument("--wav", required=True, help="Path to session WAV")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--lang", default="nl", help="Language code (default: nl)")
    ap.add_argument("--thr", type=float, default=0.72, help="Coach similarity threshold (default: 0.72)")
    ap.add_argument("--max_speakers", type=int, default=2, help="Max non-coach speakers (default: 2)")
    ap.add_argument("--no_words", action="store_true", help="Disable word timestamps")
    args = ap.parse_args()

    t0 = time.time()

    wav_path = Path(args.wav)
    wav, sr = load_audio(wav_path, target_sr=settings.SAMPLE_RATE, mono=True)
    duration = len(wav) / sr
    print(f"[batch] audio: {wav_path} duration={duration:.2f}s")

    segments = detect_voiced_segments(
        wav, sr,
        frame_ms=settings.VAD_FRAME_MS,
        aggressiveness=2,
        min_seg_dur=settings.MIN_SEG_DUR,
        merge_gap=settings.MERGE_GAP,
    )
    print(f"[batch] VAD segments: {len(segments)}")

    embs = embed_segments(wav, sr, segments)
    coach_emb = load_coach_embedding("COACH")
    diar = diarize(segments, embs, coach_emb, thr=args.thr, max_speakers=args.max_speakers)
    n_coach = sum(1 for _,_,lab in diar if lab == "COACH")
    n_noncoach = len(diar) - n_coach
    print(f"[batch] diarization: coach={n_coach} noncoach={n_noncoach}")

    tmp = save_temp_wav(wav_path.read_bytes())
    try:
        asr_segments = asr_transcribe(
            audio_path=str(tmp),
            language=args.lang,
            word_timestamps=not args.no_words,
        )
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    utterances = assign_speakers_to_words(
        diar_segments=diar,
        whisper_segments=asr_segments,
        merge_gap=settings.MERGE_GAP,
        min_turn_dur=settings.MIN_SEG_DUR,
    )

    # Build output JSON
    out = {
        "session_id": str(uuid.uuid4()),
        "language": args.lang,
        "speakers": [
            {"id": "COACH", "display": "Coach"},
            {"id": "JONGERE", "display": "Jongere"},
        ],
        "utterances": [
            {
                "start": float(u["start"]),
                "end": float(u["end"]),
                "speaker": u["speaker"],
                "text": u["text"],
                "words": [
                    {"w": w["w"], "start": float(w["start"]), "end": float(w["end"]), "speaker": w["speaker"]}
                    for w in u.get("words", [])
                ]
            } for u in utterances
        ],
        "metrics": {
            "processing_sec": float(time.time() - t0),
            "model": model_name_display(),
        }
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[batch] wrote {out_path}")

if __name__ == "__main__":
    main()
