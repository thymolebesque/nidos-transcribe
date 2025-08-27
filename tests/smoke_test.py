import os
import numpy as np

# Ensure nothing tries to download during import
os.environ.setdefault("OFFLINE_ONLY", "true")

def test_imports_and_vad_pipeline():
    from app.config import settings
    from app.services.vad import detect_voiced_segments
    from app.services.io_utils import load_audio, float_to_int16_pcm

    # Generate 1s of near-silence to ensure VAD returns []
    sr = settings.SAMPLE_RATE
    wav = np.zeros(int(sr), dtype=np.float32)
    segs = detect_voiced_segments(wav, sr, frame_ms=settings.VAD_FRAME_MS,
                                  aggressiveness=2, min_seg_dur=settings.MIN_SEG_DUR, merge_gap=settings.MERGE_GAP)
    assert isinstance(segs, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in segs)

def test_modules_load():
    import app.services.embeddings as embeddings
    import app.services.asr as asr
    import app.services.diarization as diarization
    import app.services.align as align

    assert hasattr(embeddings, "embed_signal")
    assert hasattr(asr, "transcribe")
    assert hasattr(diarization, "diarize")
    assert hasattr(align, "assign_speakers_to_words")
