from generate_resume import app as resume_graph
from resume_analyzer.crew import ResumeAnalyzerCrew

def generate_and_review(job_description, company_name, resume_bank, max_attempts=3):
    feedback = ""
    result = None
    outcomes = None

    for attempt in range(max_attempts):
        result = resume_graph.invoke({
            "job_description": job_description,
            "resume_bank": resume_bank,
            "company_name": company_name,
            "tailored_resume": "",
            "pdf_path": "",
            "feedback": feedback,
        })

        review = ResumeAnalyzerCrew().crew().kickoff(inputs={
            "tailored_resume_md": result["tailored_resume"],
            "job_description": job_description,
        })
        outcomes = [t.pydantic for t in review.tasks_output]

        if all(o.pass_ for o in outcomes):
            return result, outcomes, attempt + 1

        feedback = "\n".join(f"- {issue}" for o in outcomes for issue in o.issues)

    return result, outcomes, max_attempts
