import os
import json
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=1500,
    reasoning_effort="low",
)

class ReviewResult(BaseModel):
    score: int
    pass_: bool = Field(alias="pass")
    issues: list[str]
    class Config:
        populate_by_name = True

structured_llm = llm.with_structured_output(ReviewResult, method="json_mode")

ATS_PROMPT = """You are an expert ATS resume reviewer and technical recruiter specializing
in AI/ML, software engineering, backend development, and GenAI roles. You understand how
Applicant Tracking Systems extract information from resumes and compare candidates with job
descriptions. You carefully check keyword matching, technical skills, job titles, section
headings, formatting, and readability. You identify missing relevant keywords and ATS-related
problems, but you must never recommend adding skills or experience that the candidate does
not actually have.

Review the tailored resume against the job description specifically for Applicant Tracking
System (ATS) compatibility.

Check:
- How closely the resume matches the job title and role
- Relevant keywords from the job description
- Required technical skills that are genuinely present in the resume
- Relevant tools, technologies, and qualifications
- Whether important information is easy for an ATS to extract
- Standard resume section headings and structure
- Formatting or structure that could cause ATS parsing problems
- Whether the most relevant qualifications are clearly represented

Do not recommend adding skills, technologies, qualifications, or experience that the
candidate does not actually have.

RESUME:
{tailored_resume_md}

JOB DESCRIPTION:
{job_description}

Respond with ONLY a valid JSON object containing exactly these keys: "score" (integer),
"pass" (boolean), "issues" (array of strings). Output valid JSON and nothing else."""

TECHNICAL_PROMPT = """You are a senior technical hiring manager with expertise in AI/ML,
Generative AI, LLMs, backend engineering, and software development. Your job is to carefully
compare every technical claim in the tailored resume with the candidate's verified source
information. You detect fabricated technologies, unsupported experience, exaggerated
achievements, invented metrics, and incorrect project descriptions. A technology being
required by the job does not mean the candidate has that technology, so never treat missing
skills as evidence of dishonesty. Your priority is factual accuracy and preventing the resume
from claiming experience that the candidate does not have.

Review the tailored resume for technical and factual accuracy.

Identify anything that is:
- Invented or fabricated
- Exaggerated
- Unsupported by the candidate's source information
- Technically incorrect
- An incorrect description of a project
- An incorrect technology or tool claim
- An invented or unsupported metric
- An exaggerated responsibility or achievement

The job description is provided only to understand the technical requirements and context
of the role. Do not assume the candidate has a skill simply because the job description
requires it. Do not flag a missing job requirement as a false claim unless the resume
incorrectly claims the candidate has that skill. Never recommend adding a technology,
experience, achievement, or qualification not supported by the candidate's source material.

RESUME:
{tailored_resume_md}

JOB DESCRIPTION:
{job_description}

CANDIDATE SOURCE MATERIAL:
{candidate_source}

Respond with ONLY a valid JSON object containing exactly these keys: "score" (integer),
"pass" (boolean), "issues" (array of strings). Output valid JSON and nothing else."""

READABILITY_PROMPT = """You are an experienced technical recruiter who has reviewed thousands
of resumes for AI/ML, software engineering, backend, and GenAI positions. You understand that
recruiters often spend only a few seconds scanning a resume before deciding whether to read
it in detail. You evaluate the resume's information hierarchy, section ordering, bullet-point
clarity, conciseness, project relevance, technical skill visibility, and overall first
impression. You identify content that is confusing, repetitive, overly long, or difficult to
scan. You focus on making the candidate's genuine qualifications immediately clear without
adding false or unsupported information.

Review the tailored resume from the perspective of a technical recruiter performing a quick
6-second scan.

Check:
- Whether the candidate's target role is immediately clear
- Whether the most relevant skills are easy to find
- Whether the most relevant projects or experience are easy to find
- Whether the resume has a clear information hierarchy
- Whether sections are ordered logically
- Whether bullet points are concise and easy to scan
- Whether achievements and impact are clearly communicated
- Whether important information is buried under unnecessary content
- Whether the resume contains repetitive, vague, or unnecessarily long content
- Whether the candidate's fit for the specific job is immediately understandable

Focus only on readability, clarity, skimmability, and recruiter impact. Do not evaluate ATS
parsing or technical truthfulness unless an issue directly affects readability.

RESUME:
{tailored_resume_md}

JOB DESCRIPTION:
{job_description}

Respond with ONLY a valid JSON object containing exactly these keys: "score" (integer),
"pass" (boolean), "issues" (array of strings). Output valid JSON and nothing else."""

REVIEW_PROMPTS = {
    "ats": ATS_PROMPT,
    "technical": TECHNICAL_PROMPT,
    "readability": READABILITY_PROMPT,
}

def run_review_crew(tailored_markdown: str, job_description: str, resume_bank: dict | None = None) -> dict:
    candidate_source = json.dumps(resume_bank or {}, indent=2)

    inputs = {
        "ats": {"tailored_resume_md": tailored_markdown, "job_description": job_description},
        "technical": {
            "tailored_resume_md": tailored_markdown,
            "job_description": job_description,
            "candidate_source": candidate_source,
        },
        "readability": {"tailored_resume_md": tailored_markdown, "job_description": job_description},
    }

    results = {}
    for key, template_str in REVIEW_PROMPTS.items():
        prompt = ChatPromptTemplate.from_template(template_str)
        chain = prompt | structured_llm
        try:
            parsed = chain.invoke(inputs[key])
            if parsed is None:
                raise ValueError("Model returned no parseable structured output")
            parsed.pass_ = parsed.score >= 75  # enforce threshold ourselves, don't trust model's own pass/fail
            results[key] = parsed
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
