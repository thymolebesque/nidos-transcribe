from __future__ import annotations
import io
import tempfile
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def load_audio(path_or_bytes: Union[str, Path, bytes, io.BytesIO], target_sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Loads audio using soundfile and returns float32 mono @ target_sr.
    """
    if isinstance(path_or_bytes, (str, Path)):
        data, sr = sf.read(str(path_or_bytes), always_2d=True, dtype="float32")
    else:
        if isinstance(path_or_bytes, bytes):
            bio = io.BytesIO(path_or_bytes)
        else:
            bio = path_or_bytes
        data, sr = sf.read(bio, always_2d=True, dtype="float32")

    # to mono
    if mono and data.shape[1] > 1:
        data = np.mean(data, axis=1, dtype=np.float32).reshape(-1, 1)
    else:
        data = data[:, :1]  # ensure shape (N,1)

    wav = data.squeeze(1)  # (N,)
    # resample if needed
    if sr != target_sr:
        # resample with resample_poly for efficiency
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        wav = resample_poly(wav, up, down).astype(np.float32)
        sr = target_sr

    # sanity: clip to [-1, 1]
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    return wav, sr

def save_temp_wav(data_bytes: bytes) -> Path:
    """
    Save uploaded bytes to a temp WAV file and return the path.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(data_bytes)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)

def float_to_int16_pcm(wav: np.ndarray) -> bytes:
    """
    Converts float32 [-1,1] wav to 16-bit PCM bytes (little-endian).
    """
    wav16 = np.clip(wav, -1.0, 1.0)
    wav16 = (wav16 * 32767.0).astype(np.int16)
    return wav16.tobytes()
