from __future__ import annotations
from typing import Dict, List, Tuple

import itertools
import numpy as np

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def _assign_label_by_overlap(t0: float, t1: float, diar: List[Tuple[float, float, str]]) -> str:
    # Majority overlap; fallback to nearest segment center
    if not diar:
        return "UNKNOWN"
    totals: Dict[str, float] = {}
    for (a, b, lab) in diar:
        ov = _overlap(t0, t1, a, b)
        if ov > 0:
            totals[lab] = totals.get(lab, 0.0) + ov
    if totals:
        return max(totals.items(), key=lambda kv: kv[1])[0]

    # fallback: center-time nearest
    center = 0.5 * (t0 + t1)
    best_lab = "UNKNOWN"
    best_dist = float("inf")
    for (a, b, lab) in diar:
        c = 0.5 * (a + b)
        d = abs(center - c)
        if d < best_dist:
            best_dist = d
            best_lab = lab
    return best_lab

def assign_speakers_to_words(
    diar_segments: List[Tuple[float, float, str]],
    whisper_segments: List[Dict],
    merge_gap: float = 0.2,
    min_turn_dur: float = 0.5,
) -> List[Dict]:
    """
    Returns "utterances": list of {start, end, speaker, text, words: [{w,start,end,speaker}, ...]}
    We prefer word-level when available, else segment-level.
    """
    utterances: List[Dict] = []

    # Case 1: word timestamps available for majority
    has_words = any("words" in s and s["words"] for s in whisper_segments)

    if has_words:
        # Build per-word stream with assigned speakers
        word_stream = []
        for seg in whisper_segments:
            seg_words = seg.get("words") or []
            for w in seg_words:
                w0 = float(w["start"])
                w1 = float(w["end"])
                spk = _assign_label_by_overlap(w0, w1, diar_segments)
                word_stream.append({
                    "w": w["word"],
                    "start": w0,
                    "end": w1,
                    "speaker": spk
                })

        # Group consecutive words by speaker with small gap tolerance
        if not word_stream:
            return []

        current = {
            "start": word_stream[0]["start"],
            "end": word_stream[0]["end"],
            "speaker": word_stream[0]["speaker"],
            "words": [word_stream[0]]
        }

        for w in word_stream[1:]:
            same_spk = (w["speaker"] == current["speaker"])
            gap = w["start"] - current["end"]
            if same_spk and gap <= merge_gap:
                current["end"] = max(current["end"], w["end"])
                current["words"].append(w)
            else:
                # finalize
                current["text"] = " ".join(x["w"] for x in current["words"]).strip()
                utterances.append(current)
                # start new
                current = {"start": w["start"], "end": w["end"], "speaker": w["speaker"], "words": [w]}

        # finalize last
        current["text"] = " ".join(x["w"] for x in current["words"]).strip()
        utterances.append(current)

    else:
        # Segment-level only
        for seg in whisper_segments:
            s0 = float(seg["start"])
            s1 = float(seg["end"])
            lab = _assign_label_by_overlap(s0, s1, diar_segments)
            utterances.append({
                "start": s0, "end": s1, "speaker": lab,
                "text": seg["text"], "words": []
            })

    # Merge too-short turns (< min_turn_dur) with neighbors of the same speaker
    if utterances:
        merged: List[Dict] = []
        for utt in utterances:
            if not merged:
                merged.append(utt)
                continue
            prev = merged[-1]
            if utt["speaker"] == prev["speaker"] and (utt["start"] - prev["end"]) <= merge_gap:
                # merge
                prev["end"] = utt["end"]
                if prev["words"] and utt["words"]:
                    prev["words"].extend(utt["words"])
                    prev["text"] = " ".join(x["w"] for x in prev["words"]).strip()
                else:
                    prev["text"] = (prev["text"] + " " + utt["text"]).strip()
            else:
                merged.append(utt)
        utterances = merged

        # second pass: if an utterance still < min_turn_dur, try merging with neighbor speakers if same label exists
        final: List[Dict] = []
        for utt in utterances:
            if final and (utt["end"] - utt["start"]) < min_turn_dur and utt["speaker"] == final[-1]["speaker"]:
                # merge backward
                prev = final[-1]
                prev["end"] = max(prev["end"], utt["end"])
                if prev["words"] and utt["words"]:
                    prev["words"].extend(utt["words"])
                    prev["text"] = " ".join(x["w"] for x in prev["words"]).strip()
                else:
                    prev["text"] = (prev["text"] + " " + utt["text"]).strip()
            else:
                final.append(utt)
        utterances = final

    # Normalize: strip text, ensure float types
    for u in utterances:
        u["text"] = u.get("text", "").strip()
        u["start"] = float(u["start"])
        u["end"] = float(u["end"])
        for w in u.get("words", []):
            w["w"] = w["w"].strip()
            w["start"] = float(w["start"])
            w["end"] = float(w["end"])

    return utterances
