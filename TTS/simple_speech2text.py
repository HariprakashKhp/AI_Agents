import requests
import torch
from transformers import pipeline
import gradio as gr
import sys
import ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hTqGqoC-LrW6S79HjuJUkg/trimmed-02.wav"

# response = requests.get(url)

file_path = "sample_meeting.wav"

# if response.status_code == 200:
#     with open(file_path, "wb") as file:
#         file.write(response.content)
#         print("file downloaded successfully")
# else:
#     print("file download failed")

# print(gr.__version__)
# print(sys.version)

model_name = "qwen3:latest"

llm = OllamaLLM(
    model=model_name
)

def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
     financial products and common financial terms are in the correct format. For each
     financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
     transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)' , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)', and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing financial products into their numeric representations, followed by the full name of the product in parentheses. For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'. However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to  and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted transcript and a list of the words you've changed"""

    prompt_input = system_prompt + "\n" + ascii_transcript

    response = llm.invoke(prompt_input)

    return response

template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": RunnablePassthrough()}  # Pass the transcript as context
    | prompt
    | llm
    | StrOutputParser()
)

def transcript_audio(audio_file):
    #AI part 
    pipe = pipeline(
      "automatic-speech-recognition",
      model="openai/whisper-tiny.en",
      chunk_length_s=30,
    )
    raw_transcript = pipe(audio_file, batch_size = 8)["text"]
    ascii_transcript = remove_non_ascii(raw_transcript)
    adjusted_transcript = product_assistant(ascii_transcript)
    res = chain.invoke({"context": adjusted_transcript})
    #AI Part
    return res

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn=transcript_audio, inputs=audio_input, outputs=output_text, title="Audio Transcript App", description="Upload the audio file")

iface.launch(server_name="0.0.0.0", server_port=5000)