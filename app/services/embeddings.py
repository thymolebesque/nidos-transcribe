from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier

from app.config import settings

_classifier: Optional[EncoderClassifier] = None

def _assert_local_model_exists(path: Path):
    if not path.exists():
        raise RuntimeError(
            f"ECAPA model not found at '{path}'. "
            "Place a local copy of 'speechbrain/spkrec-ecapa-voxceleb' there.\n"
            "See README (Hugging Face CLI instructions)."
        )

def get_classifier() -> EncoderClassifier:
    global _classifier
    if _classifier is not None:
        return _classifier

    local_path = settings.ecapa_local_path
    _assert_local_model_exists(local_path)

    # Use local dir. If OFFLINE_ONLY and files missing, raise early.
    _classifier = EncoderClassifier.from_hparams(
        source=str(local_path),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        savedir=str(local_path),  # avoid new dirs
    )
    _classifier.eval()
    return _classifier

def embed_signal(wav: np.ndarray, sr: int, t0: Optional[float] = None, t1: Optional[float] = None) -> np.ndarray:
    """
    Compute ECAPA embedding for [t0, t1] slice (seconds) or full wav.
    Returns np.ndarray (dim ~192).
    """
    if t0 is not None and t1 is not None:
        start = max(0, int(t0 * sr))
        end = min(len(wav), int(t1 * sr))
        chunk = wav[start:end]
    else:
        chunk = wav

    if len(chunk) == 0:
        # zero-length fallback
        return np.zeros((192,), dtype=np.float32)

    x = torch.from_numpy(chunk).float().unsqueeze(0)  # [1, T]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(device)
    with torch.no_grad():
        clf = get_classifier()
        emb = clf.encode_batch(x)  # [1, D]
        emb = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return emb

def embed_segments(wav: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> List[np.ndarray]:
    return [embed_signal(wav, sr, t0, t1) for (t0, t1) in segments]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def mean_pool(vectors: List[np.ndarray]) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((192,), dtype=np.float32)
    X = np.vstack(vectors)
    return np.mean(X, axis=0).astype(np.float32)

# -------- Speaker DB --------

def _default_db() -> Dict:
    return {}

def load_speaker_db(path: Path = settings.SPEAKER_DB_PATH) -> Dict:
    if not path.exists():
        return _default_db()
    try:
        return json.loads(path.read_text())
    except Exception:
        return _default_db()

def save_speaker_db(db: Dict, path: Path = settings.SPEAKER_DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False, indent=2))
    tmp.replace(path)

def save_coach_embedding(embedding: np.ndarray, sr: int, name: str = "COACH", duration: float = 0.0) -> None:
    db = load_speaker_db()
    db[name] = {
        "embedding": embedding.tolist(),
        "sr": sr,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": float(duration),
        "name": name,
    }
    save_speaker_db(db)

def load_coach_embedding(name: str = "COACH") -> Optional[np.ndarray]:
    db = load_speaker_db()
    entry = db.get(name)
    if not entry:
        return None
    emb = np.array(entry["embedding"], dtype=np.float32)
    return emb
