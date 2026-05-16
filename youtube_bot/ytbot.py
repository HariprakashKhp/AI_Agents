import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.runnables import RunnableLambda
from langchain_community .vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from functions import (
    summarize_video
    , answer_question
    )

#Gradio UI

with gr.Blocks() as Interface:
    video_url =gr.Textbox(label="Youtube video url", placeholder="Enter the youtube video url")

    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    summarize_btn.click(summarize_video, inputs=[video_url],outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

Interface.launch(server_name="0.0.0.0", server_port=7860)