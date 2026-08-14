import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel,Field

reviewer_llm = LLM(
    model="openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

class ReviewResult(BaseModel):
    score: int
    pass_: bool = Field(alias="pass")
    issues: list[str]

    class Config:
        populate_by_name = True

@CrewBase
class ResumeAnalyzerCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def ats_reviewer(self) -> Agent:
        return Agent(config=self.agents_config['ats_reviewer'], verbose=True)

    @agent
    def technical_accuracy_reviewer(self) -> Agent:
        return Agent(config=self.agents_config['technical_accuracy_reviewer'], verbose=True)

    @agent
    def readability_reviewer(self) -> Agent:
        return Agent(config=self.agents_config['readability_reviewer'], verbose=True)

    @task
    def ats_review_task(self) -> Task:
        return Task(config=self.tasks_config['ats_review_task'], output_pydantic=ReviewResult)

    @task
    def technical_accuracy_task(self) -> Task:
        return Task(config=self.tasks_config['technical_accuracy_task'], output_pydantic=ReviewResult)

    @task
    def readability_task(self) -> Task:
        return Task(config=self.tasks_config['readability_task'], output_pydantic=ReviewResult)

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
