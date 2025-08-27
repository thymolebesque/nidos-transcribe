from typing import List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"

class EnrollResponse(BaseModel):
    speaker: str
    duration_sec: float
    embedding_dim: int
    saved: bool = True

class Word(BaseModel):
    w: str = Field(..., description="Word text (no trailing space)")
    start: float
    end: float
    speaker: str

class Utterance(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    words: List[Word] = []

class Speaker(BaseModel):
    id: str
    display: str

class Metrics(BaseModel):
    processing_sec: float
    model: str

class TranscribeResponse(BaseModel):
    session_id: str
    language: str
    speakers: List[Speaker]
    utterances: List[Utterance]
    metrics: Metrics