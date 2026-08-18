python
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
            # Don't let one reviewer's failure kill the whole batch
            results[key] = ReviewResult(score=0, pass_=False, issues=[f"Reviewer error: {repr(e)}"])

    scores = {k: v.score for k, v in results.items()}
    feedback = {k: v.issues for k, v in results.items()}
    passed = {k: v.pass_ for k, v in results.items()}

    return {
        "scores": scores,
        "feedback": feedback,
        "per_agent_passed": passed,
    }
What changed and why:

No crewai import at all → no LiteLLM in the dependency chain for this file.
with_structured_output(ReviewResult) replaces output_pydantic — LangChain forces the model to return matching JSON, same guarantee CrewAI gave you.
Each reviewer prompt is pulled from your config/tasks.yaml/config/agents.yaml logic — you'll want to copy your actual task/agent instructions from those YAML files into the REVIEW_PROMPTS dict above (I used placeholders based on the reviewer names).
Added a per-reviewer try/except so one bad OpenRouter response doesn't kill all three reviews — errors get logged into that reviewer's issues instead of crashing _run_job.
max_tokens=1500 caps each review call — should be plenty for a score + issues list, and prevents the runaway-token problem you saw before.
One thing to check: open resume_analyzer/config/agents.yaml and tasks.yaml and paste them here — I used generic prompts above, but your actual configured instructions might have more specific scoring criteria you don't want to lose. Once I see them I'll merge the real wording in.

Also — since this removes CrewAI's dependency entirely for review, you can drop crewai from requirements.txt unless you use it elsewhere, which slightly speeds up your Render builds too.


ats_review_task: description: > Review the tailored resume against the job description specifically for Applicant Tracking System (ATS) compatibility. Check: - How closely the resume matches the job title and role - Relevant keywords from the job description - Requ

pasted


Message collapsed
Perfect, now I have the real prompts. Here's the complete replacement for resume_analyzer/crew.py using your actual task descriptions, no CrewAI/LiteLLM involved:

python
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

Return score (0-100), pass (true when score >= 75, else false), and short actionable issues."""

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

Return score (0-100), pass (true when score >= 75 and no serious fabricated/unsupported
claims, else false), and short specific issues."""

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

Return score (0-100), pass (true when score >= 75, else false), and short, specific,
actionable issues."""

REVIEW_PROMPTS = {
    "ats": ATS_PROMPT,
    "technical": TECHNICAL_PROMPT,
    "readability": READABILITY_PROMPT,
}

def run_review_crew(tailored_markdown: str, job_description: str, resume_bank: dict | None = None) -> dict:
    """
    Runs ATS/technical/readability review as three direct LLM calls.
    Returns the same shape the LangGraph pipeline expects: scores, feedback, per_agent_passed.
    """
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
Why this is a safe drop-in:

run_review_crew() keeps the exact same function signature and return shape (scores, feedback, per_agent_passed) — generate_resume.py's review_resume_node needs zero changes.
Threshold logic (score >= 75) is preserved exactly as your YAML specified.
Each of the three reviewer personas/instructions is carried over verbatim from your agents.yaml/tasks.yaml, just merged into one prompt per reviewer instead of split across agent backstory + task description.
.with_structured_output() forces valid JSON matching ReviewResult, same guarantee output_pydantic gave you in CrewAI.
Each reviewer is wrapped independently in try/except — one failing OpenRouter call won't crash the whole review step, it just surfaces as an issue for that reviewer with the real error message (repr(e)), unlike the blank error you got before.
Cleanup once this works:

Delete/stop importing resume_analyzer/config/agents.yaml and tasks.yaml (no longer read).
Remove crewai from requirements.txt if nothing else in the project uses it — check first.
generate_resume.py and review_loop.py need no changes — the interface is identical.
Want me to also add basic parallelization (run all 3 reviews concurrently instead of sequentially, since they're independent) to cut review latency roughly 3x?







Claude is AI and can make mistakes. Please double-check responses.



