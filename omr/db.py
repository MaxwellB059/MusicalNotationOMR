"""MongoDB Atlas data layer (PLAN.md S8a). Optional by design.

If MONGODB_URI is unset, or Atlas is unreachable, `get_db()` returns None and the app runs
statelessly: recognition must never depend on the database. Images are NOT stored here
(GridFS/object storage + a reference); documents hold metadata, results and corrections.

Collections:
  jobs         id, status, stage, created, updated, error, filename, bytes   (TTL-indexed)
  results      job_id, musicxml, detections, model_versions
  corrections  job_id, symbol_id, old_label, new_label, created             (future training data)
  eval_runs    model_version, metrics, created
"""
import os
import uuid
from datetime import datetime, timedelta, timezone

JOB_TTL_DAYS = 14


def _now():
    return datetime.now(timezone.utc)


def get_db(uri=None, db_name="secundo", client_factory=None):
    """Return a database handle, or None (stateless mode) if unconfigured/unreachable."""
    uri = uri if uri is not None else os.environ.get("MONGODB_URI", "")
    if not uri and client_factory is None:
        return None
    try:
        if client_factory is not None:
            client = client_factory()
        else:
            from pymongo import MongoClient
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
        db = client[db_name]
        ensure_indexes(db)
        return db
    except Exception as exc:                      # never let the DB block recognition
        print(f"[db] running stateless: {exc}")
        return None


def ensure_indexes(db):
    db.jobs.create_index("updated", expireAfterSeconds=JOB_TTL_DAYS * 86400)
    db.jobs.create_index("id", unique=True)
    db.results.create_index("job_id")
    db.corrections.create_index("job_id")
    db.eval_runs.create_index("model_version")


def create_job(db, filename=None, nbytes=0):
    job = {"id": uuid.uuid4().hex, "status": "queued", "stage": None, "created": _now(),
           "updated": _now(), "error": None, "filename": filename, "bytes": nbytes}
    if db is not None:
        db.jobs.insert_one(dict(job))
    return job["id"]


def update_job(db, job_id, **fields):
    if db is None:
        return
    db.jobs.update_one({"id": job_id}, {"$set": {**fields, "updated": _now()}})


def get_job(db, job_id):
    if db is None:
        return None
    doc = db.jobs.find_one({"id": job_id}, {"_id": 0})
    return doc


def save_result(db, job_id, musicxml, detections=None, model_versions=None):
    if db is None:
        return
    db.results.insert_one({"job_id": job_id, "musicxml": musicxml, "detections": detections or [],
                           "model_versions": model_versions or {}, "created": _now()})
    update_job(db, job_id, status="done")


def save_correction(db, job_id, symbol_id, old_label, new_label):
    if db is None:
        return
    db.corrections.insert_one({"job_id": job_id, "symbol_id": symbol_id, "old_label": old_label,
                               "new_label": new_label, "created": _now()})


def record_eval(db, model_version, metrics):
    if db is None:
        return
    db.eval_runs.insert_one({"model_version": model_version, "metrics": metrics, "created": _now()})
