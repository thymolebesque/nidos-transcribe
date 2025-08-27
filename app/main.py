from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import health, enroll, transcribe

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
