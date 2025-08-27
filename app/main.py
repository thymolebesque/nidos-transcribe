from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import health, enroll, transcribe

import logging
# show INFO from our packages
logging.basicConfig(level=logging.INFO)
logging.getLogger("app").setLevel(logging.INFO)
logging.getLogger("app.services.diarization").setLevel(logging.INFO)


app = FastAPI(title="nidos-transcribe", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(enroll.router)
app.include_router(transcribe.router)

# Static demo UI
app.mount("/web", StaticFiles(directory="web", html=True), name="web")
