import json, os, markdown
from weasyprint import HTML
from typing import TypedDict
from dotenv import load_dotenv
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

class ResumeState(TypedDict):
    job_description:str
    resume_bank:list
    tailored_resume:str
    company_name:str
    pdf_path:str
    feedback:str
    review_scores: dict 
    review_feedback: dict
    review_per_agent_passed: dict
    review_passed: bool

prompt=ChatPromptTemplate.from_template("""
You are a resume-tailoring assistant. You will be given:
1. A candidate's full bank of resume bullets (as JSON), each tagged with skills.
2. A job description.

Select the bullets most relevant to this job, and lightly rewrite them to mirror
the job description's language and keywords — WITHOUT inventing any claim,
metric, or skill not present in the source bullets.

You may think through your reasoning first if needed. When you are ready to give
your final answer, output it between these exact markers:

===RESUME_START===
===RESUME_END===

Between the markers, write the ACTUAL resume content — real markdown starting
with the candidate's name as a top-level heading, followed by a professional
summary paragraph tailored to this role, then each project as a subheading with
its selected bullet points underneath. Use the candidate's real name, real
project names, and real bullet text from the RESUME BANK below. Do not write
placeholder text, descriptions of what should go there, or instructions —
write the finished resume itself, ready to be read by a recruiter.

Example of the expected format (structure only, not real content):
===RESUME_START===
# Neeraj Mittal
neeraj@email.com | github.com/neerajmittal

Backend engineer with experience building scalable APIs and data pipelines,
seeking to apply Python and cloud deployment skills to this role.

## Project Name
- Built X using Y, resulting in Z
- Implemented A to achieve B
===RESUME_END===

RESUME BANK:
{resume_bank}

JOB DESCRIPTION:
{job_description}

PREVIOUS REVIEW FEEDBACK (address these issues if present):
{feedback}
""")

def generate_node(state:ResumeState)->ResumeState:
    chain=prompt|llm
    result=chain.invoke({
        "resume_bank": json.dumps(state["resume_bank"],indent=2),
        "job_description":state["job_description"],
        "feedback": state.get("feedback", "") or "None — first draft.",
    })
    content = result.content
    if "===RESUME_START===" in content and "===RESUME_END===" in content:
        content = content.split("===RESUME_START===", 1)[1].split("===RESUME_END===", 1)[0].strip()
    state["tailored_resume"]=content
    return state

def pdf_node(state:ResumeState)->ResumeState:
    html_content=markdown.markdown(state["tailored_resume"])
    styled_html=f"""
    <html><head><style>
        body {{ font-family: 'Helvetica', sans-serif; margin: 40px; }}
        h1, h2 {{ color: #1a1a1a; }}
        li {{ margin-bottom: 6px; }}
    </style></head>
    <body>{html_content}</body></html>
    """
    output_path=f"/tmp/resume_{state['company_name']}.pdf"
    HTML(string=styled_html).write_pdf(output_path)
    state["pdf_path"]=output_path
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

graph=StateGraph(ResumeState)
graph.add_node("generate",generate_node)
graph.add_node("review_resume", review_resume_node)
graph.add_node("to_pdf",pdf_node)

graph.set_entry_point("generate")
graph.add_edge("generate", "review_resume") 
graph.add_edge("review_resume", "to_pdf") 
graph.add_edge("to_pdf",END)
app=graph.compile()

if __name__=="__main__":
    resume_bank=json.load(open("resume_bank.json"))
    job_description=open("temp_job.txt").read()
    result=app.invoke({
        "job_description": job_description,
        "resume_bank": resume_bank,
        "tailored_resume": "",
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
