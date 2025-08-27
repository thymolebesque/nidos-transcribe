from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
import logging
logger = logging.getLogger(__name__)

from sklearn.cluster import AgglomerativeClustering

from app.config import settings
from .embeddings import cosine

def label_segments_with_coach(
    segments: List[Tuple[float, float]],
    embs: List[np.ndarray],
    coach_emb: Optional[np.ndarray],
    thr: float = 0.72,
    smooth_window: int = 3,
) -> List[str]:
    """
    Returns a list of labels with initial COACH/UNK classification and temporal smoothing.
    Also logs cosine similarities if enabled in settings.
    """
    if coach_emb is None:
        return ["UNK"] * len(segments)

    # --- NEW: log cosine similarities ---
    if settings.LOG_COSINE_SCORES and len(segments) == len(embs):
        logger.info("=== Coach cosine similarities per VAD segment ===")
        for (t0, t1), emb in zip(segments, embs):
            sim = float(cosine(emb, coach_emb))
            logger.info(f"{t0:7.2f}–{t1:7.2f}  sim={sim:.3f}")
        logger.info("=== end similarities ===")

    raw = np.array([1 if cosine(e, coach_emb) >= thr else 0 for e in embs], dtype=np.int32)

    if len(raw) == 0:
        return []

    # median smoothing over window
    k = max(1, smooth_window)
    smoothed = raw.copy()
    if k > 1 and len(raw) >= k:
        pad = k // 2
        padded = np.pad(raw, (pad, pad), mode="edge")
        out = []
        for i in range(len(raw)):
            window = padded[i : i + k]
            out.append(int(np.median(window)))
        smoothed = np.array(out, dtype=np.int32)

    return ["COACH" if v == 1 else "UNK" for v in smoothed]

def _dur(seg: Tuple[float, float]) -> float:
    return max(0.0, seg[1] - seg[0])

def cluster_unknowns(
    segments: List[Tuple[float, float]],
    embs: List[np.ndarray],
    labels: List[str],
    max_speakers: int = 2,
) -> List[str]:
    """Cluster segments labeled UNK into OTHER_1/OTHER_2..., then promote dominant to JONGERE."""
    final = labels.copy()
    unk_idx = [i for i, lab in enumerate(labels) if lab == "UNK"]
    if len(unk_idx) == 0:
        return final

    X = np.vstack([embs[i] for i in unk_idx])
    n_clusters = min(max_speakers, max(1, min(len(unk_idx), 4)))
    if len(unk_idx) == 1 or n_clusters == 1:
        cluster_labels = np.zeros((len(unk_idx),), dtype=int)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = model.fit_predict(X)

    for j, i in enumerate(unk_idx):
        final[i] = f"OTHER_{int(cluster_labels[j])+1}"

    durations: Dict[str, float] = {}
    for i, lab in enumerate(final):
        if lab != "COACH":
            durations[lab] = durations.get(lab, 0.0) + _dur(segments[i])

    if durations:
        dominant = max(durations.items(), key=lambda kv: kv[1])[0]
        final = ["JONGERE" if lab == dominant else lab for lab in final]

    return final

def diarize(
    segments: List[Tuple[float, float]],
    embs: List[np.ndarray],
    coach_emb: Optional[np.ndarray],
    thr: float = 0.72,
    max_speakers: int = 2,
) -> List[Tuple[float, float, str]]:
    """
    Full diarization pipeline: COACH matching + smoothing + clustering unknowns.
    Returns [(t0, t1, label), ...]
    """
    initial = label_segments_with_coach(segments, embs, coach_emb, thr=thr, smooth_window=3)
    final_labels = cluster_unknowns(segments, embs, initial, max_speakers=max_speakers)
    return [(a, b, lab) for (a, b), lab in zip(segments, final_labels)]
