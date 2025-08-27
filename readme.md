# nidos-transcribe — Local Dutch Transcription + Diarization (Coach Enrollment)

A **fully local/offline** FastAPI demo for **Dutch** transcription with **speaker diarization** and **coach voice enrollment**.

- **ASR**: [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) (CTranslate2)
- **VAD**: `webrtcvad`
- **Speaker embeddings**: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- **Clustering**: Agglomerative (scikit-learn)
- **Constrained labels**: segments similar to the enrolled **COACH** embedding are labeled `COACH`; the dominant non‑coach cluster becomes `JONGERE` (others are tagged `OTHER_1`, `OTHER_2` internally).
- **Word timestamps**: uses Faster‑Whisper word-level timestamps when available
- **All processing local**. **No external APIs.**

---

## Quick start (local, CPU)

> **Python**: 3.10 recommended  
> **OS packages** (Linux/WSL/Ubuntu): `libsndfile1`, `ffmpeg` (optional, for convenience)

```bash
git clone <this-repo> nidos-transcribe
cd nidos-transcribe

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
