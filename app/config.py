import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT_DIR / "app" / "store"
MODELS_DIR = STORE_DIR / "models"

def _getenv_float(key: str, default: float) -> float:
    v = os.getenv(key, str(default))
    try:
        return float(v)
    except Exception:
        return default

def _getenv_int(key: str, default: int) -> int:
    v = os.getenv(key, str(default))
    try:
        return int(v)
    except Exception:
        return default

def _getenv_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

class Settings:
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    LANGUAGE: str = os.getenv("LANGUAGE", "nl")
    COACH_THRESHOLD: float = _getenv_float("COACH_THRESHOLD", 0.72)
    SAMPLE_RATE: int = _getenv_int("SAMPLE_RATE", 16000)
    VAD_FRAME_MS: int = _getenv_int("VAD_FRAME_MS", 30)
    MIN_SEG_DUR: float = _getenv_float("MIN_SEG_DUR", 0.5)
    MERGE_GAP: float = _getenv_float("MERGE_GAP", 0.2)
    OFFLINE_ONLY: bool = _getenv_bool("OFFLINE_ONLY", True)

    # Paths
    SPEAKER_DB_PATH: Path = STORE_DIR / "speaker_db.json"

    # Local model roots (must exist for offline-only)
    WHISPER_LOCAL_DIR: Path = Path(os.getenv("WHISPER_LOCAL_DIR", str(MODELS_DIR / "faster-whisper")))
    SB_ECAPA_LOCAL_DIR: Path = Path(os.getenv("SB_ECAPA_LOCAL_DIR", str(MODELS_DIR / "spkrec-ecapa-voxceleb")))

    @property
    def whisper_model_path(self) -> Path:
        return self.WHISPER_LOCAL_DIR / self.WHISPER_MODEL

    @property
    def ecapa_local_path(self) -> Path:
        return self.SB_ECAPA_LOCAL_DIR

settings = Settings()
STORE_DIR.mkdir(parents=True, exist_ok=True)
(settings.WHISPER_LOCAL_DIR).mkdir(parents=True, exist_ok=True)
(settings.SB_ECAPA_LOCAL_DIR).mkdir(parents=True, exist_ok=True)
