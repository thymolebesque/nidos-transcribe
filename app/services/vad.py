from __future__ import annotations
from typing import List, Tuple

import numpy as np
import webrtcvad

from .io_utils import float_to_int16_pcm

def detect_voiced_segments(
    wav: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    aggressiveness: int = 2,
    min_seg_dur: float = 0.5,
    merge_gap: float = 0.2,
) -> List[Tuple[float, float]]:
    """
    Returns a list of (t0, t1) voiced segments in seconds using WebRTC VAD.

    - Assumes wav is mono float32 at sr=16k ideally (function works at any sr; VAD requires 8k/16k/32k/48k; we use 16k).
    - frame_ms must be one of {10,20,30}.
    """
    assert frame_ms in (10, 20, 30), "webrtcvad supports 10/20/30ms frames"
    vad = webrtcvad.Vad(int(aggressiveness))
    pcm = float_to_int16_pcm(wav)
    bytes_per_sample = 2  # int16
    frame_bytes = int(sr * (frame_ms / 1000.0)) * bytes_per_sample
    n_frames = len(pcm) // frame_bytes

    flags = []
    for i in range(n_frames):
        start_b = i * frame_bytes
        frame = pcm[start_b : start_b + frame_bytes]
        is_speech = vad.is_speech(frame, sr)
        flags.append(1 if is_speech else 0)

    # Merge contiguous speech frames into segments
    segments: List[Tuple[float, float]] = []
    i = 0
    while i < n_frames:
        if flags[i] == 1:
            start_i = i
            while i < n_frames and flags[i] == 1:
                i += 1
            end_i = i  # exclusive
            t0 = start_i * frame_ms / 1000.0
            t1 = end_i * frame_ms / 1000.0
            segments.append((t0, t1))
        else:
            i += 1

    # Merge small gaps (<merge_gap)
    merged: List[Tuple[float, float]] = []
    for s in segments:
        if not merged:
            merged.append(s)
        else:
            (p0, p1) = merged[-1]
            (q0, q1) = s
            if q0 - p1 <= merge_gap:
                merged[-1] = (p0, q1)
            else:
                merged.append(s)

    # Remove too-short segments
    filtered = [(a, b) for (a, b) in merged if (b - a) >= min_seg_dur]
    return filtered
