from crewai import Crew, Process, LLM
from agents import blog_researcher, blog_writer
from tasks import research_task, write_task
from dotenv import load_dotenv

load_dotenv()

gemini_llm = LLM(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

crew = Crew(
    agents =[blog_researcher, blog_researcher],
    tasks =[research_task, write_task],
    process =Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True,
    manager_llm=gemini_llm
)

result =crew.kickoff(inputs={"topic": "AI vs ML vs DL vs Data Science"})

print(result)
