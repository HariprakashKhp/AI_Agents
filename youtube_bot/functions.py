import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community .vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda

# Defining the llm and the models used
llm = OllamaLLM(
    model= "qwen3.5:2b"
)

def get_video_id(url):
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(url):
    video_id = get_video_id(url)

    ytt_api = YouTubeTranscriptApi()

    transcripts = ytt_api.list(video_id)

    transcript = ""
    for t in transcripts:
        if t.language_code == "en":
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break;

    return transcript if transcript else None

def process(transcript):
    txt = ""

    for i in transcript:
        print(i)
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass

    return txt

def chunk_transcript(processed_transcript, chunk_size = 200, chunk_overlap = 20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )

    chunks = text_splitter.split_text(processed_transcript)
    return chunks

def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)

# The same method will be used to retrive context for the AI
def perform_similarity_search(query, faiss_index, k =3):
    print(type(faiss_index))

    results = faiss_index.similarity_search(query, k=k)
    return results

def create_summary_prompt(transcript):
    template = """
    <|im_start|>system:
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.
    <|im_end|>
    <|im_start|>
    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.
    <|im_end|>
    <|im_start|>
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|im_end|>
"""

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )

    return prompt

def create_qa_prompt_template():
    qa_template = """
        You are an expert assistant providing detailed answers based on the following video content.

        Relevant Video Context: {context}

        Based on the above context, please answer the following question:
        Question: {question}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    return prompt_template

def generate_answer(question, faiss_index, qa_chain, k = 7):
    relavant_context = perform_similarity_search(question, faiss_index, k = k)
    answer = qa_chain.invoke({"context":relavant_context, "question" : question})
    return answer

def summarize_video(video_url):
    if video_url:
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid url"
    
    if processed_transcript:
        summary_chain = RunnableLambda(create_summary_prompt) | llm
        summary = summary_chain.invoke(processed_transcript)
        print("summary -- ", summary)
        return summary
    else:
        return "No transcript available, Please fetch the transcript first."
    
def answer_question(video_url, user_question):
    if video_url:
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid url"
    
    if processed_transcript and user_question:
        chunks = chunk_transcript(processed_transcript)
        
        embed =HuggingFaceEmbeddings(
            model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        
        faiss_index = create_faiss_index(chunks, embed)
        qa_chain = create_qa_prompt_template() | llm
        print(type(faiss_index))
        answer = generate_answer(user_question,faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure that the trascript has been fetched."
