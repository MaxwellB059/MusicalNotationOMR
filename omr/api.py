"""FastAPI app factory (PLAN.md S8/S8a). The model is injected, so the API is testable with a fake
predictor and the production image can use ONNX Runtime instead of TensorFlow.

Backward compatible: POST /api/omr with one image returns MusicXML text, exactly as before.
"""
import os
import re

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from . import db as dbmod

MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 15 * 1024 * 1024))
LOCAL_ORIGIN = r"http://(localhost|127\.0\.0\.1)(:\d+)?"


def allowed_origins_from_env():
    """ALLOWED_ORIGINS='https://app.up.railway.app,https://other' -> list. Empty -> localhost only."""
    raw = os.environ.get("ALLOWED_ORIGINS", "")
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


def create_app(recognize, db=None, origins=None):
    """recognize(bgr_image) -> MusicXML string. db may be None (stateless)."""
    app = FastAPI(title="Secundo OMR")
    origins = origins if origins is not None else allowed_origins_from_env()
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_origin_regex=LOCAL_ORIGIN,
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "ok", "db": "connected" if db is not None else "stateless"}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        doc = dbmod.get_job(db, job_id)
        if doc is None:
            raise HTTPException(404, "Unknown job (or running without a database)")
        return doc

    @app.post("/api/omr", response_class=PlainTextResponse)
    async def omr(file: UploadFile = File(...)):
        contents = await file.read()
        if len(contents) > MAX_BYTES:
            raise HTTPException(413, f"File too large (max {MAX_BYTES // (1024 * 1024)} MB)")
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode image")
        job_id = dbmod.create_job(db, file.filename, len(contents))
        try:
            dbmod.update_job(db, job_id, status="running", stage="recognize")
            xml = recognize(img)
        except Exception as exc:
            dbmod.update_job(db, job_id, status="error", error=str(exc)[:500])
            raise HTTPException(500, "Recognition failed") from exc
        dbmod.save_result(db, job_id, xml)
        return PlainTextResponse(xml, headers={"X-Job-Id": job_id})

    return app
