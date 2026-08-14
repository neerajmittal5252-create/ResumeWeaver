from pydantic import BaseModel
from generate_resume import app as resume_graph


class ReviewOutcome(BaseModel):
    agent: str
    score: int
    pass_: bool
    issues: list[str]

    class Config:
        populate_by_name = True


def generate_and_review(job_description, company_name, resume_bank, max_attempts=2):
    """
    Runs the LangGraph resume pipeline (generate -> review -> pdf).
    If the review doesn't pass, feeds the review feedback back in and
    retries up to max_attempts times.
    Returns: (result_state, outcomes_for_last_attempt, attempts_used)
    """
    state = {
        "job_description": job_description,
        "resume_bank": resume_bank,
        "tailored_resume": "",
        "company_name": company_name,
        "pdf_path": "",
        "feedback": "",
        "review_scores": {},
        "review_feedback": {},
        "review_per_agent_passed": {},
        "review_passed": False,
    }

    result = None
    outcomes = []

    for attempt in range(1, max_attempts + 1):
        result = resume_graph.invoke(state)

        outcomes = [
            ReviewOutcome(
                agent=agent_name,
                score=result["review_scores"].get(agent_name, 0),
                pass_=result["review_per_agent_passed"].get(agent_name, False),
                issues=result["review_feedback"].get(agent_name, []),
            )
            for agent_name in result.get("review_scores", {})
        ]

        if result["review_passed"]:
            return result, outcomes, attempt

        state = {
            **state,
            "tailored_resume": result["tailored_resume"],
            "feedback": str(result["review_feedback"]),
        }

    return result, outcomes, max_attempts
