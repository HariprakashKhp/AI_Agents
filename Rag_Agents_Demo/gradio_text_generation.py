import gradio as gr
from langchain_ollama import ChatOllama
from langchain.tools import tool
from datetime import datetime
import requests
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

model_name = "qwen3"

llm = ChatOllama(
    model=model_name,
    temperature=0.5,
    num_predict=256,
)

# Tools 
@tool
def get_time():
    """Returns current system time"""
    return datetime.now().isoformat()

@tool
def get_weather(city):
    """Gets City as input and returns the weather data in that city"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=e4a81caa25577571d2897d5bbf364b1b"
    return requests.get(url).json()

tools = [get_time, get_weather]

llm = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, you can use the following tools to answer user queries: get_time, get_weather. Use get_time to get the current time and get_weather to get the weather data for a city. Always use the tools when necessary to answer user queries."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def generate_response(prompt):
    response = executor.invoke({"input": prompt})
    
    # LangChain agents return dict
    if isinstance(response, dict):
        return response.get("output", str(response))
    
    return str(response)

chat_bot = gr.Interface(
    fn= generate_response,
    inputs= gr.Textbox(label="Prompt", lines=2, placeholder="type your prompt here..."),
    outputs=gr.Textbox(label="Output"),
    title="Ollama Chatbot",
    description="Ask any question and the chatbot will answer"
)

chat_bot.launch()  
