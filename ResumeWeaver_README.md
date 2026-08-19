# ResumeWeaver

ResumeWeaver is an AI-powered resume tailoring and generation service that takes a candidate's structured resume bank and a target job description, generates a role-specific resume, reviews it through multiple specialized AI reviewers, and renders the final resume as a PDF.

The repository is built around a FastAPI service, a LangGraph generation/review pipeline, Groq-hosted structured-output LLM calls, and WeasyPrint PDF generation.

## What the project does

At a high level, ResumeWeaver performs this flow:

```text
Job Description + Resume Bank
            |
            v
   LangGraph Resume Generator
            |
            v
     Structured Resume JSON
            |
            v
      Markdown Resume
            |
            v
   Multi-Agent Review Layer
     /        |         \
   ATS    Technical   Readability
     \        |         /
            v
       Review Result
            |
            v
       HTML -> PDF
            |
            v
       FastAPI Response
```

The generation prompt explicitly instructs the LLM to use only information present in `resume_bank.json`, which is an important design choice for reducing fabricated resume claims.

## Repository structure

```text
ResumeWeaver/
├── resume_analyzer/
│   ├── __init__.py
│   └── crew.py
├── .python-version
├── Dockerfile
├── api.py
├── generate_resume.py
├── requirements.txt
├── resume_bank.json
└── review_loop.py
```

### `resume_analyzer/__init__.py`

The package initializer is intentionally empty. Its purpose is to make `resume_analyzer` a Python package so that modules such as `resume_analyzer.crew` can be imported.

### `resume_analyzer/crew.py`

This file implements the resume review layer.

It creates a Groq-backed structured-output LLM and defines a `ReviewResult` Pydantic model:

- `score`: reviewer score
- `pass_`: whether the resume passes
- `issues`: detected problems

Three specialized review prompts are defined:

1. **ATS reviewer**
   - Checks keyword matching.
   - Checks role alignment.
   - Checks technical skills that genuinely exist in the resume bank.
   - Checks ATS-readable structure and headings.
   - Looks for formatting/structure problems.

2. **Technical reviewer**
   - Checks whether technical claims are supported by the source resume bank.
   - Detects fabricated technologies, metrics, experience, and responsibilities.
   - Checks for technically incorrect project descriptions.

3. **Readability reviewer**
   - Evaluates the resume from a recruiter's quick-scan perspective.
   - Checks information hierarchy, clarity, conciseness, relevance, and project visibility.

`run_review_crew()` executes all three reviewers independently and returns:

```python
{
    "scores": {...},
    "feedback": {...},
    "per_agent_passed": {...}
}
```

The pass threshold is enforced in Python at `score >= 75` instead of trusting the LLM's boolean output.

### `.python-version`

Pins the intended Python version to:

```text
3.11.9
```

### `Dockerfile`

The Docker image uses Python 3.11 slim.

It installs the native libraries required by WeasyPrint, installs the Python dependencies, copies the application into `/app`, and starts FastAPI with Uvicorn on port `10000`.

Container startup command:

```bash
uvicorn api:api --host 0.0.0.0 --port 10000
```

### `api.py`

This is the HTTP API layer.

The FastAPI application is created as:

```python
api = FastAPI(title="Resume Generator API")
```

#### Data models

`Resumerequest`

```json
{
  "job_description": "...",
  "company_name": "..."
}
```

`Resumeresponse` contains:

- company name
- tailored Markdown resume
- Base64-encoded PDF
- review status
- number of attempts
- per-reviewer outcomes

`JobStatus` supports:

- `pending`
- `running`
- `completed`
- `failed`

#### Endpoints

### `GET /health`

Simple health check.

Response:

```json
{
  "status": "ok"
}
```

### `POST /generate-resume`

Starts resume generation in a background thread and immediately returns a job ID.

Example request:

```json
{
  "job_description": "We are hiring a backend engineer with Python and FastAPI experience...",
  "company_name": "Example Corp"
}
```

Example response:

```json
{
  "job_id": "generated-uuid",
  "status": "pending"
}
```

### `GET /status/{job_id}`

Returns the current status of a generation job.

When complete, the response contains the tailored resume, Base64 PDF, review results, and attempt count.

### `GET /download-pdf/{job_id}`

Returns the generated PDF as an HTTP file response.

The current implementation expects the PDF at:

```text
/tmp/resume_<company_name>.pdf
```

### Background processing

`_run_job()`:

1. Marks the job as running.
2. Loads `resume_bank.json`.
3. Calls `generate_and_review()`.
4. Reads the generated PDF.
5. Encodes the PDF using Base64.
6. Stores the result in the in-memory `jobs` dictionary.
7. Marks the job completed or failed.

A `threading.Lock` protects concurrent access to the in-memory job store.

## `generate_resume.py`

This file contains the core LangGraph resume-generation pipeline.

### LLM configuration

The project uses `ChatGroq` with:

```text
Model: openai/gpt-oss-20b
Temperature: 0.3
Max tokens: 2500
Reasoning effort: low
```

The API key is read from:

```text
GROQ_API_KEY
```

### Structured resume schema

The LLM output is constrained through Pydantic models.

`EducationEntry`:

- institution
- degree
- location
- dates

`ProjectEntry`:

- title
- link text
- link URL
- dates
- bullets

`ResumeContent`:

- name
- phone
- email
- GitHub
- LinkedIn
- summary
- education
- projects
- skills
- coursework
- achievements

The structured-output model is:

```python
structured_resume_llm = llm.with_structured_output(
    ResumeContent,
    method="json_mode"
)
```

### Resume generation prompt

The prompt tells the LLM to:

- select relevant projects, bullets, and skills
- lightly adapt wording to the job description
- use job-description keywords where appropriate
- never invent metrics, skills, degrees, achievements, dates, or experience
- return only the expected JSON structure

This makes `resume_bank.json` the source of truth.

### `resume_to_markdown()`

Converts the structured `ResumeContent` object into Markdown.

The generated structure is:

```text
# Name

Contact information

## Summary

...

## Education

...

## Projects

...

## Technical Skills

...

## Relevant Coursework

...

## Achievements

...
```

### `generate_node()`

Runs the LLM chain and stores:

- structured resume object
- Markdown resume

inside the LangGraph state.

### `make_contact_html()`

Converts contact information into HTML.

GitHub and LinkedIn are converted into clickable links, and email is rendered as a `mailto:` link.

### `make_link_html()`

Converts a project URL into a clickable project link when a URL exists.

### `pdf_node()`

Converts the structured resume into HTML and uses WeasyPrint to generate:

```text
/tmp/resume_<company_name>.pdf
```

The PDF includes:

- name
- contact information
- summary
- education
- projects
- technical skills
- coursework
- achievements

### `review_resume_node()`

Calls:

```python
run_review_crew(...)
```

and stores:

- reviewer scores
- reviewer feedback
- per-agent pass/fail values
- overall pass/fail result

### LangGraph

The graph is:

```text
START
  |
  v
generate
  |
  v
review_resume
  |
  v
to_pdf
  |
  v
 END
```

The compiled graph is exposed as:

```python
app = graph.compile()
```

When executed directly, the file expects a `temp_job.txt` file containing the job description.

## `review_loop.py`

This file adds retry/orchestration logic around the LangGraph pipeline.

### `ReviewOutcome`

A Pydantic model containing:

- reviewer name
- score
- pass/fail result
- issues

### `generate_and_review()`

The function:

```python
generate_and_review(
    job_description,
    company_name,
    resume_bank,
    max_attempts=2
)
```

runs the resume graph and checks the reviewer results.

If the resume fails review, reviewer feedback is added back into the generation state and the graph is invoked again.

It returns:

```text
(result_state, outcomes_for_last_attempt, attempts_used)
```

This creates a feedback loop:

```text
Generate
   |
   v
Review
   |
   +---- Pass ----> Finish
   |
   +---- Fail ----> Feed feedback back
                         |
                         v
                      Generate
```

## `resume_bank.json`

This is the project's source-of-truth candidate profile.

It contains:

- candidate contact information
- professional summary
- education
- projects
- project bullets
- skills
- coursework
- achievements

The project data currently covers:

- Sigma Chat
- PneumoScan AI
- IPL Match Win Prediction

Each project bullet includes associated skills, allowing the generation system to select relevant information for a target job.

The resume bank is deliberately structured as data instead of hard-coded prose so that the same candidate profile can be reused across many job descriptions.

## `requirements.txt`

The main dependencies are:

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
langchain==0.3.14
langchain_groq
langgraph==0.2.62
python-dotenv==1.0.1
markdown==3.7
weasyprint==63.1
```

The application therefore combines:

- FastAPI for HTTP APIs
- Pydantic for validation and structured LLM output
- LangChain for LLM prompting
- LangGraph for workflow orchestration
- Groq for LLM inference
- python-dotenv for environment variables
- WeasyPrint for PDF generation

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/neerajmittal5252-create/ResumeWeaver.git
cd ResumeWeaver
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit the `.env` file.

### 5. Start the API

```bash
uvicorn api:api --reload
```

The API will normally be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

## Docker

Build:

```bash
docker build -t resumeweaver .
```

Run:

```bash
docker run -p 10000:10000 \
  -e GROQ_API_KEY=your_groq_api_key \
  resumeweaver
```

The container exposes port `10000`.

## API usage example

Submit a resume-generation job:

```bash
curl -X POST "http://localhost:8000/generate-resume" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Looking for a Python backend engineer with FastAPI, PostgreSQL and AI experience.",
    "company_name": "Example Corp"
  }'
```

The API returns a `job_id`.

Poll the job:

```bash
curl "http://localhost:8000/status/<job_id>"
```

Once completed, download the PDF:

```bash
curl -o resume.pdf \
  "http://localhost:8000/download-pdf/<job_id>"
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Authenticates Groq LLM requests |

## Design decisions

### 1. Resume bank as source of truth

Instead of allowing the model to freely invent resume content, the generator receives a structured candidate profile.

This is critical for job-application systems because hallucinated skills or experience can make a resume misleading.

### 2. Structured LLM output

The generation model produces a Pydantic-validated `ResumeContent` object instead of unconstrained text.

That makes downstream Markdown and PDF generation more deterministic.

### 3. Multiple reviewers

The project does not rely on a single LLM judgment.

It separately evaluates:

```text
ATS compatibility
Technical/factual correctness
Recruiter readability
```

### 4. Asynchronous API submission

`POST /generate-resume` does not wait for the full LLM pipeline.

Instead, it creates a job and runs generation in a background thread. Clients poll the status endpoint.

### 5. Deterministic review threshold

Reviewer scores are converted into pass/fail in Python:

```python
pass_ = score >= 75
```

This prevents the model from returning an inconsistent boolean relative to its own numeric score.

## Important implementation notes

The current repository is functional in concept, but there are several engineering limitations worth knowing.

### In-memory job storage

Jobs are stored in:

```python
jobs: dict[str, dict] = {}
```

This means job state disappears when the process restarts.

It also means multiple API instances will not share job state. A production deployment should use a persistent store such as PostgreSQL or Redis and a proper worker queue.

### Background threads

The API uses Python daemon threads for long-running LLM work.

That is simple, but it is not a robust distributed job-processing architecture. For production, a queue/worker system would be more appropriate.

### API currently disables the retry loop

`generate_and_review()` defaults to two attempts, but `api.py` explicitly calls it with:

```python
max_attempts=1
```

Therefore API requests currently perform only one generation/review attempt.

If iterative correction is desired, this should be changed to a value greater than one.

### PDF path depends on company name

The generated PDF is stored as:

```text
/tmp/resume_<company_name>.pdf
```

Company names containing special characters can create filesystem/path problems.

A safer design would use the job UUID as the filename.

### PDF storage is ephemeral

PDF files are written to `/tmp`.

They can disappear after container/process restarts. Production deployments should use durable object storage if generated resumes need to persist.

### `temp_job.txt`

The `__main__` section of `generate_resume.py` reads:

```python
temp_job.txt
```

This file is not part of the repository root shown in the current source tree. Running `generate_resume.py` directly therefore requires creating that file first.

### HTML escaping

Resume fields are interpolated directly into HTML before being passed to WeasyPrint.

For untrusted input, HTML escaping should be added to prevent malformed markup and injection-like content.

### Dependency consistency

The source imports `langchain_core.prompts.ChatPromptTemplate`, `langchain_groq`, and other LangChain-related packages. These should be tested together after installation because some transitive versions are not explicitly pinned in `requirements.txt`.

## Production improvement roadmap

A stronger production architecture would look like:

```text
                    ┌──────────────────┐
                    │   Client / UI    │
                    └────────┬─────────┘
                             |
                             v
                    ┌──────────────────┐
                    │    FastAPI       │
                    └────────┬─────────┘
                             |
                             v
                    ┌──────────────────┐
                    │ Job Queue / Redis│
                    └────────┬─────────┘
                             |
             ┌───────────────┴───────────────┐
             v                               v
      ┌──────────────┐               ┌──────────────┐
      │ Resume Worker│               │ Review Worker│
      └──────┬───────┘               └──────┬───────┘
             |                               |
             └──────────────┬────────────────┘
                            v
                     ┌──────────────┐
                     │ PostgreSQL   │
                     └──────────────┘
                            |
                            v
                     ┌──────────────┐
                     │ Object Store │
                     │ PDF resumes  │
                     └──────────────┘
```

Potential improvements:

- Redis/PostgreSQL-backed job state
- Celery, Dramatiq, RQ, or another worker system
- persistent PDF/object storage
- authentication and rate limiting
- request validation and size limits
- HTML escaping
- structured logging
- retry policies for transient LLM failures
- observability/tracing
- model fallback strategy
- versioned resume-bank schemas
- automated tests
- CI/CD
- explicit dependency pinning
- unique UUID-based output filenames

## Security considerations

Never commit:

```text
.env
GROQ_API_KEY
```

The resume bank currently contains personal candidate information. If this repository is intended to be public, consider whether exposing personal contact information in `resume_bank.json` is acceptable.

For a production system, add:

- authentication
- authorization
- API rate limiting
- secret management
- input sanitization
- secure file handling
- persistent storage access controls
- logging that avoids leaking sensitive resume information

## Technology stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Validation | Pydantic |
| LLM orchestration | LangChain + LangGraph |
| LLM provider | Groq |
| Model | `openai/gpt-oss-20b` |
| Review system | Multi-agent reviewer prompts |
| Resume source | JSON |
| Intermediate format | Markdown / structured JSON |
| PDF generation | WeasyPrint |
| Configuration | python-dotenv |
| Containerization | Docker |
| Runtime | Python 3.11.9 |

## End-to-end flow

1. Client submits a job description and company name.
2. FastAPI creates a UUID job.
3. A background thread starts the generation pipeline.
4. `resume_bank.json` is loaded.
5. LangGraph invokes the resume-generation LLM.
6. The LLM produces structured resume content.
7. The content is converted into Markdown.
8. Three AI reviewers evaluate the resume.
9. Review scores and feedback are stored.
10. The structured resume is converted into HTML.
11. WeasyPrint generates a PDF.
12. The API reads and Base64-encodes the PDF.
13. The job is marked completed.
14. The client polls the status endpoint and can download the PDF.

## Current status

ResumeWeaver is a compact prototype of an AI resume-tailoring backend. Its strongest architectural idea is the combination of:

- structured resume data
- constrained LLM generation
- LangGraph workflow orchestration
- independent ATS/technical/readability review
- automated PDF rendering
- asynchronous API submission

The main gap between this prototype and a production-grade service is operational infrastructure: persistent jobs, distributed workers, durable file storage, authentication, observability, stronger validation, and automated testing.

## License

No license file is currently present in the repository. If this project is intended for public reuse, add an explicit license such as MIT, Apache-2.0, or another license appropriate to the project.
