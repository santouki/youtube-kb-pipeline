"""
FastAPI microservice wrapper for the YouTube KB pipeline.

Endpoints:
  POST   /run                     Start a pipeline job (async)
  GET    /jobs                    List all jobs
  GET    /jobs/{job_id}           Get job status and progress
  DELETE /jobs/{job_id}           Remove a completed/failed job
  GET    /database/{folder}       Return full database.json as JSON
  GET    /database/{folder}/csv   Download database.csv
  GET    /health                  Health check
"""

import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline_core import load_db, run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube KB Pipeline API",
    description="Fetch YouTube transcripts, extract structured insights with Claude, query results.",
    version="1.0.0",
)

# Base directory for all database output (override with DATABASES_DIR env var)
DATABASES_DIR = Path(os.getenv("DATABASES_DIR", "./databases"))

# In-memory job store  {job_id: dict}
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


# ─── Request / Response models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    channel_url: str = Field(..., description="YouTube channel or playlist URL")
    folder_name: str = Field(..., description="Output subfolder inside DATABASES_DIR")
    extract_fields: dict[str, str] = Field(
        ...,
        description='Fields to extract per video. Key = field name, value = description for Claude. '
                    'Example: {"techniques": "List of BJJ techniques shown", "position": "Guard or position covered"}'
    )
    filter_keywords: list[str] = Field(default=[], description="Title keyword filters (any match passes). Empty = no filter.")
    months: int = Field(default=12, description="Only fetch videos from the last N months")
    max_videos: int | None = Field(default=None, description="Cap number of videos processed (useful for testing)")
    skip_existing: bool = Field(default=False, description="Skip video IDs already present in the database")
    cookies_file: str | None = Field(default=None, description="Path to Netscape cookies.txt for YouTube auth")
    extract_model: str = Field(default="claude-haiku-4-5-20251001", description="Claude model for extraction")


class JobStatus(BaseModel):
    job_id: str
    status: str           # pending | running | completed | failed
    total: int            # videos to process (set after filter step)
    processed: int        # successfully extracted and saved
    skipped: int          # no transcript available
    failed: int           # extraction failed
    current_video: str | None
    started_at: str
    completed_at: str | None
    error: str | None     # top-level error if job itself crashed
    failed_videos: list[str]


# ─── Job runner ───────────────────────────────────────────────────────────────

def _safe_folder(folder_name: str) -> Path:
    """Resolve output path and guard against path traversal."""
    resolved = (DATABASES_DIR / folder_name).resolve()
    if not str(resolved).startswith(str(DATABASES_DIR.resolve())):
        raise HTTPException(400, "Invalid folder_name")
    return resolved


def _run_job(job_id: str, req: RunRequest):
    output_dir = _safe_folder(req.folder_name)

    def on_progress(event: dict):
        with _lock:
            job = _jobs[job_id]
            t = event["type"]
            if t == "matched":
                job["total"] = event["count"]
                job["status"] = "running"
            elif t == "start":
                job["current_video"] = event["title"]
            elif t == "processed":
                job["processed"] += 1
                job["current_video"] = None
            elif t == "skipped":
                job["skipped"] += 1
                job["current_video"] = None
            elif t == "failed":
                job["failed"] += 1
                job["failed_videos"].append(event["title"])
                job["current_video"] = None

    try:
        with _lock:
            _jobs[job_id]["status"] = "running"

        run_pipeline(
            channel_url=req.channel_url,
            folder_name=str(output_dir),
            extract_fields=req.extract_fields,
            filter_keywords=req.filter_keywords,
            months=req.months,
            max_videos=req.max_videos,
            skip_existing=req.skip_existing,
            cookies_file=req.cookies_file,
            extract_model=req.extract_model,
            on_progress=on_progress,
        )

        with _lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/run", response_model=JobStatus, status_code=202)
def start_run(req: RunRequest, background_tasks: BackgroundTasks):
    """
    Start a pipeline job. Returns immediately with a job_id.
    Poll GET /jobs/{job_id} to track progress.
    """
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "current_video": None,
            "started_at": now,
            "completed_at": None,
            "error": None,
            "failed_videos": [],
        }

    background_tasks.add_task(_run_job, job_id, req)
    logger.info(f"Job {job_id} queued for channel: {req.channel_url}")
    return _jobs[job_id]


@app.get("/jobs", response_model=list[JobStatus])
def list_jobs():
    """List all jobs (running, completed, and failed)."""
    with _lock:
        return list(_jobs.values())


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    """Get status and progress for a specific job."""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return job


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    """Remove a completed or failed job from the store."""
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job '{job_id}' not found")
        if job["status"] == "running":
            raise HTTPException(409, "Cannot delete a running job")
        del _jobs[job_id]


@app.get("/database/{folder_name}")
def get_database(folder_name: str):
    """Return the full database.json for a folder as a JSON array."""
    path = _safe_folder(folder_name) / "database.json"
    if not path.exists():
        raise HTTPException(404, f"No database found for '{folder_name}'. Run a pipeline job first.")
    return load_db(str(path))


@app.get("/database/{folder_name}/csv")
def get_database_csv(folder_name: str):
    """Download database.csv for a folder."""
    path = _safe_folder(folder_name) / "database.csv"
    if not path.exists():
        raise HTTPException(404, f"No CSV found for '{folder_name}'.")
    return FileResponse(str(path), media_type="text/csv", filename=f"{folder_name}.csv")


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "databases_dir": str(DATABASES_DIR)}
