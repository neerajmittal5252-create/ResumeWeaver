import os
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
    max_tokens=1500,
)

class ReviewResult(BaseModel):
    score: int
    pass_: bool = Field(alias="pass")
    issues: list[str]
    class Config:
        populate_by_name = True

structured_llm = llm.with_structured_output(ReviewResult)

REVIEW_PROMPTS = {
    "ats": """You are an ATS (Applicant Tracking System) compatibility reviewer.
Evaluate this tailored resume against the job description for ATS-friendliness:
formatting, keyword match, parseable structure.

TAILORED RESUME:
{tailored_resume_md}

JOB DESCRIPTION:
{job_description}

Give a score (0-100), pass (true if score >= 70), and a list of specific issues.""",

    "technical": """You are a technical accuracy reviewer.
Check that every technical claim, metric, and skill in this tailored resume is
actually supported by the candidate's original source bullets — flag anything
invented or exaggerated.

TAILORED RESUME:
{tailored_resume_md}

CANDIDATE SOURCE BULLETS:
{candidate_source}

Give a score (0-100), pass (true if score >= 70), and a list of specific issues.""",

    "readability": """You are a readability and clarity reviewer.
Evaluate this tailored resume for clarity, conciseness, active voice, and
professional tone.

TAILORED RESUME:
{tailored_resume_md}

Give a score (0-100), pass (true if score >= 70), and a list of specific issues.""",
}

def run_review_crew(tailored_markdown: str, job_description: str, resume_bank: dict | None = None) -> dict:
    """
    Runs ATS/technical/readability review as three direct LLM calls.
    Returns the same shape as before: scores, feedback, per_agent_passed.
    """
    candidate_source = json.dumps(resume_bank or {}, indent=2)

    inputs = {
        "ats": {"tailored_resume_md": tailored_markdown, "job_description": job_description},
        "technical": {"tailored_resume_md": tailored_markdown, "candidate_source": candidate_source},
        "readability": {"tailored_resume_md": tailored_markdown},
    }

    results = {}
    for key, template_str in REVIEW_PROMPTS.items():
        prompt = ChatPromptTemplate.from_template(template_str)
        chain = prompt | structured_llm
        try:
            results[key] = chain.invoke(inputs[key])
        except Exception as e:
            results[key] = ReviewResult(score=0, pass_=False, issues=[f"Reviewer error: {repr(e)}"])

    scores = {k: v.score for k, v in results.items()}
    feedback = {k: v.issues for k, v in results.items()}
    passed = {k: v.pass_ for k, v in results.items()}

    return {
        "scores": scores,
        "feedback": feedback,
        "per_agent_passed": passed,
    }
