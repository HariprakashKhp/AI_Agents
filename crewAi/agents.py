from crewai import Agent, LLM
from tools import yt_tool
from dotenv import load_dotenv

load_dotenv()

gemini_llm = LLM(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

blog_researcher = Agent(
    role="Blog Researcher from Youtube Videos",
    goal="get the relevant video content for the topic {topic} from Yt channel",
    verbose=True,
    memory=True,
    backstory ="Expert in understanding videos in AI Data Science, Machine Learning and Gen AI and providing suggestion",
    tools=[yt_tool],
    allow_delegation=True,
    llm=gemini_llm
)

blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic} from Yt channel",
    verbose=True,
    memory=True,
    backstory = (
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing"
        "new discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=False,
    llm=gemini_llm
)