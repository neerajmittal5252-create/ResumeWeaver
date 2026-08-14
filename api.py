import json
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from generate_resume import app as resume_graph
from review_loop import generate_and_review

api=FastAPI(title="Resume Generator API")

RESUME_BANK_PATH="resume_bank.json"

class Resumerequest(BaseModel):
    job_description:str
    company_name:str

class Resumeresponse(BaseModel):
    company_name:str
    tailored_resume_md:str
    pdf_base64:str

@api.get("/health")
def health_check():
    return {"status":"ok"}    

@api.post("/generate-resume", response_model=Resumeresponse)
def generate_resume(req: Resumerequest):
    try:
        resume_bank = json.load(open(RESUME_BANK_PATH))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"{RESUME_BANK_PATH} not found on server")
 
    try:
        result = resume_graph.invoke({
            "job_description": req.job_description,
            "resume_bank": resume_bank,
            "tailored_resume": "",
            "company_name": req.company_name,
            "pdf_path": "",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume generation failed: {str(e)}")
 
    pdf_path = result.get("pdf_path")
    if not pdf_path:
        raise HTTPException(status_code=500, detail="Graph did not produce a pdf_path")
 
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"PDF file not found at {pdf_path}")
 
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
 
    return Resumeresponse(
        company_name=req.company_name,
        tailored_resume_md=result["tailored_resume"],
        pdf_base64=pdf_b64,
    )

@api.post("/generate-resume", response_model=Resumeresponse)
def generate_resume(req: Resumerequest):
    try:
        resume_bank = json.load(open(RESUME_BANK_PATH))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"{RESUME_BANK_PATH} not found on server")

    try:
        result, outcomes, attempts = generate_and_review(
            job_description=req.job_description,
            company_name=req.company_name,
            resume_bank=resume_bank,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume generation failed: {str(e)}")

    pdf_path = result.get("pdf_path")
    if not pdf_path:
        raise HTTPException(status_code=500, detail="Graph did not produce a pdf_path")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    return Resumeresponse(
        company_name=req.company_name,
        tailored_resume_md=result["tailored_resume"],
        pdf_base64=pdf_b64,
    )
