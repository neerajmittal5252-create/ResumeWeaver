import json, os, markdown
from weasyprint import HTML
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from crew import run_review_crew

load_dotenv() 

llm=ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
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

Output a clean resume draft in markdown, grouped by project, with a short
professional summary line at the top tailored to this role.

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
    state["tailored_resume"]=result.content
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
        tailored_markdown=state["tailored_markdown"],
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
