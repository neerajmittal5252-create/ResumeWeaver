import json
import base64
import uuid
import threading
from datetime import datetime, timezone
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from review_loop import generate_and_review, ReviewOutcome

api = FastAPI(title="Resume Generator API")

RESUME_BANK_PATH = "resume_bank.json"
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Resumerequest(BaseModel):
    job_description: str
    company_name: str


class Resumeresponse(BaseModel):
    company_name: str
    tailored_resume_md: str
    pdf_base64: str
    review_passed: bool
    attempts: int
    review: list[ReviewOutcome]


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Resumeresponse | None = None
    error: str | None = None


@api.get("/health")
def health_check():
    return {"status": "ok"}


def _run_job(job_id: str, job_description: str, company_name: str):
    """Runs on a background thread. Always writes a final status — never
    lets an exception vanish silently."""
    with jobs_lock:
        jobs[job_id]["status"] = JobStatus.running

    try:
        resume_bank = json.load(open(RESUME_BANK_PATH))

        result, outcomes, attempts = generate_and_review(
            job_description=job_description,
            company_name=company_name,
            resume_bank=resume_bank,
            max_attempts=1,  
        )

        pdf_path = result.get("pdf_path")
        if not pdf_path:
            raise RuntimeError("Graph did not produce a pdf_path")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        response = Resumeresponse(
            company_name=company_name,
            tailored_resume_md=result["tailored_resume"],
            pdf_base64=pdf_b64,
            review_passed=result["review_passed"],
            attempts=attempts,
            review=outcomes,
        )

        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.completed
            jobs[job_id]["result"] = response
            jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.failed
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@api.post("/generate-resume", response_model=JobSubmitResponse)
def generate_resume(req: Resumerequest):
    """Fix 2: returns immediately with a job_id instead of blocking on the
    full LLM pipeline. Poll GET /status/{job_id} for the result."""
    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "status": JobStatus.pending,
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, req.job_description, req.company_name),
        daemon=True,
    )
    thread.start()

    return JobSubmitResponse(job_id=job_id, status=JobStatus.pending)


@api.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        error=job["error"],
    )
from fastapi.responses import FileResponse

@api.get("/download-pdf/{job_id}")
def download_pdf(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found — server may have restarted")
    if job["status"] != JobStatus.completed:
        raise HTTPException(status_code=404, detail=f"Not ready — status: {job['status']}")
    result = job["result"]
    company_name = result["company_name"] if isinstance(result, dict) else result.company_name
    pdf_path = f"/tmp/resume_{company_name}.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file no longer on disk")
    return FileResponse(pdf_path, media_type="application/pdf", filename="resume.pdf")
