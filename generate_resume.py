import json, os
from weasyprint import HTML
from typing import TypedDict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from resume_analyzer.crew import run_review_crew

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=2500,
    reasoning_effort="low",
)

class EducationEntry(BaseModel):
    institution: str
    degree: str
    location: str = ""
    dates: str = ""

class ProjectEntry(BaseModel):
    title: str
    link_text: str = ""
    dates: str = ""
    bullets: list[str]

class ResumeContent(BaseModel):
    name: str
    phone: str = ""
    email: str = ""
    github: str = ""
    linkedin: str = ""
    summary: str
    education: list[EducationEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    skills: dict[str, str] = Field(default_factory=dict)
    coursework: str = ""
    achievements: list[str] = Field(default_factory=list)

structured_resume_llm = llm.with_structured_output(ResumeContent, method="json_mode")


class ResumeState(TypedDict):
    job_description: str
    resume_bank: list
    tailored_resume: str          
    resume_obj: Optional[dict]    
    company_name: str
    pdf_path: str
    feedback: str
    review_scores: dict
    review_feedback: dict
    review_per_agent_passed: dict
    review_passed: bool


resume_prompt = ChatPromptTemplate.from_template("""
You are a resume-tailoring assistant. You will be given:
1. A candidate's full bank of resume bullets/info (as JSON).
2. A job description.

Select the most relevant projects, bullets, and skills for this job, and lightly
rewrite them to mirror the job description's language and keywords — WITHOUT
inventing any claim, metric, skill, degree, or achievement not present in the
source RESUME BANK below.

Return ONLY a valid JSON object with exactly this structure:
{{
  "name": "...",
  "phone": "...",
  "email": "...",
  "github": "...",
  "linkedin": "...",
  "summary": "2-3 sentence professional summary tailored to this role",
  "education": [
    {{"institution": "...", "degree": "...", "location": "...", "dates": "..."}}
  ],
  "projects": [
    {{"title": "...", "link_text": "[GitHub]", "dates": "...", "bullets": ["...", "..."]}}
  ],
  "skills": {{"Category Name": "comma, separated, skills"}},
  "coursework": "comma separated relevant coursework",
  "achievements": ["...", "..."]
}}

Use only real information from the RESUME BANK — do not fabricate institutions,
dates, metrics, or skills. If a field genuinely has no source data, use an empty
string or empty list rather than inventing content.

RESUME BANK:
{resume_bank}

JOB DESCRIPTION:
{job_description}

PREVIOUS REVIEW FEEDBACK (address these issues if present):
{feedback}

Output valid JSON only.
""")

def resume_to_markdown(r: ResumeContent) -> str:
    """Flatten structured resume into markdown, used for the review crew."""
    lines = [f"# {r.name}"]
    contact = " | ".join(x for x in [r.phone, r.email, r.github, r.linkedin] if x)
    if contact:
        lines.append(contact)
    lines.append("")
    lines.append("## Summary")
    lines.append(r.summary)
    lines.append("")
    if r.education:
        lines.append("## Education")
        for e in r.education:
            lines.append(f"**{e.institution}** — {e.degree} ({e.dates}) {e.location}")
        lines.append("")
    if r.projects:
        lines.append("## Projects")
        for p in r.projects:
            lines.append(f"### {p.title} {p.link_text} ({p.dates})")
            for b in p.bullets:
                lines.append(f"- {b}")
        lines.append("")
    if r.skills:
        lines.append("## Technical Skills")
        for k, v in r.skills.items():
            lines.append(f"**{k}:** {v}")
        lines.append("")
    if r.coursework:
        lines.append("## Relevant Coursework")
        lines.append(r.coursework)
        lines.append("")
    if r.achievements:
        lines.append("## Achievements")
        for a in r.achievements:
            lines.append(f"- {a}")
    return "\n".join(lines)

def generate_node(state: ResumeState) -> ResumeState:
    chain = resume_prompt | structured_resume_llm
    result: ResumeContent = chain.invoke({
        "resume_bank": json.dumps(state["resume_bank"], indent=2),
        "job_description": state["job_description"],
        "feedback": state.get("feedback", "") or "None — first draft.",
    })
    state["resume_obj"] = result.model_dump()
    state["tailored_resume"] = resume_to_markdown(result)
    return state

def pdf_node(state: ResumeState) -> ResumeState:
    r = ResumeContent(**state["resume_obj"])

    edu_html = "".join(f"""
        <div class="entry-row">
            <div><strong>{e.institution}</strong><br>{e.degree}</div>
            <div class="right">{e.dates}<br>{e.location}</div>
        </div>
    """ for e in r.education)

    proj_html = "".join(f"""
        <div class="proj-title">{p.title} <span class="links">{p.link_text}</span>
            <span class="right">{p.dates}</span></div>
        <ul>{''.join(f'<li>{b}</li>' for b in p.bullets)}</ul>
    """ for p in r.projects)

    skills_html = "".join(f"<p><strong>{k}:</strong> {v}</p>" for k, v in r.skills.items())
    achievements_html = "".join(f"<li>{a}</li>" for a in r.achievements)
    contact_line = " | ".join(x for x in [r.phone, r.email, r.github, r.linkedin] if x)

    education_section = f"<h2>Education</h2>{edu_html}" if r.education else ""
    projects_section = f"<h2>Projects</h2>{proj_html}" if r.projects else ""
    skills_section = f"<h2>Technical Skills</h2>{skills_html}" if r.skills else ""
    coursework_section = f"<h2>Relevant Coursework</h2><p>{r.coursework}</p>" if r.coursework else ""
    achievements_section = f"<h2>Achievements</h2><ul>{achievements_html}</ul>" if r.achievements else ""

    styled_html = f"""
    <html><head><style>
        @page {{ margin: 1.3cm; }}
        body {{ font-family: 'Helvetica', 'Arial', sans-serif; font-size: 9.5pt; color: #111; line-height: 1.4; }}
        h1 {{ text-align: center; font-size: 18pt; margin: 0; }}
        .contact {{ text-align: center; font-size: 9pt; margin-bottom: 10px; color: #333; }}
        h2 {{ font-size: 11pt; text-transform: uppercase; letter-spacing: 0.5px;
              border-bottom: 1px solid #000; margin: 12px 0 6px 0; padding-bottom: 2px; }}
        .entry-row {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
        .right {{ text-align: right; }}
        .proj-title {{ font-weight: bold; margin-top: 6px; display: flex; justify-content: space-between; }}
        .links {{ font-weight: normal; font-size: 8.5pt; margin-left: 4px; }}
        ul {{ margin: 2px 0 8px 0; padding-left: 16px; }}
        li {{ margin-bottom: 2px; }}
        p {{ margin: 3px 0; }}
        a {{ color: #0645AD; text-decoration: none; }}
    </style></head>
    <body>
        <h1>{r.name}</h1>
        <div class="contact">{contact_line}</div>
        <h2>Summary</h2><p>{r.summary}</p>
        {education_section}
        {projects_section}
        {skills_section}
        {coursework_section}
        {achievements_section}
    </body></html>
    """
    output_path = f"/tmp/resume_{state['company_name']}.pdf"
    HTML(string=styled_html).write_pdf(output_path)
    state["pdf_path"] = output_path
    return state


def review_resume_node(state: ResumeState) -> ResumeState:
    review = run_review_crew(
        tailored_markdown=state["tailored_resume"],
        job_description=state["job_description"],
        resume_bank=state["resume_bank"],
    )
    state["review_scores"] = review["scores"]
    state["review_feedback"] = review["feedback"]
    state["review_per_agent_passed"] = review["per_agent_passed"]
    state["review_passed"] = all(review["per_agent_passed"].values())
    return state

graph = StateGraph(ResumeState)
graph.add_node("generate", generate_node)
graph.add_node("review_resume", review_resume_node)
graph.add_node("to_pdf", pdf_node)
graph.set_entry_point("generate")
graph.add_edge("generate", "review_resume")
graph.add_edge("review_resume", "to_pdf")
graph.add_edge("to_pdf", END)
app = graph.compile()

if __name__ == "__main__":
    resume_bank = json.load(open("resume_bank.json"))
    job_description = open("temp_job.txt").read()
    result = app.invoke({
        "job_description": job_description,
        "resume_bank": resume_bank,
        "tailored_resume": "",
        "resume_obj": None,
        "company_name": "Microsoft",
        "pdf_path": "",
        "feedback": "",
        "review_scores": {},
        "review_feedback": {},
        "review_per_agent_passed": {},
        "review_passed": False,
    })
    print(result["tailored_resume"])
    print(result["pdf_path"])
    print(result["review_scores"], result["review_passed"])
