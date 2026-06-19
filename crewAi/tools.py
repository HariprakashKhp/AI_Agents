from crewai_tools import YoutubeVideoSearchTool
from dotenv import load_dotenv
import os
from crewai import LLM

load_dotenv()

# yt_tool = youtube_search_tool = YoutubeVideoSearchTool(
#     youtube_video_url='https://www.youtube.com/watch?v=k2P_pHQDlp0'
# )


# CrewAI's LLM class needs the "gemini/" provider prefix to route via LiteLLM
gemini_llm = LLM(
    model="gemini-3.1-flash-lite",
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7,
)
 
# embedchain's google-generativeai embedder reads GOOGLE_API_KEY from the env,
# not GEMINI_API_KEY. Mirror it so the tool doesn't fall back to OpenAI.
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
 
yt_tool = YoutubeVideoSearchTool(
    youtube_video_url="https://www.youtube.com/watch?v=k2P_pHQDlp0",
    config=dict(
        llm=dict(
            provider="google",
            config=dict(
                model="gemini/gemini-2.5-flash",
                api_key=os.environ.get("GEMINI_API_KEY"),
            ),
        ),
        embedder=dict(
            provider="google-generativeai",
            config=dict(
                model_name="gemini-embedding-001",
                task_type="RETRIEVAL_DOCUMENT",
                api_key=os.environ.get("GEMINI_API_KEY"),
            ),
        ),
    ),
)
