"""Module for interfacing with ollama gwen LLMs."""

import logging
from typing import Dict, Any, Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

def create_embedding() -> HuggingFaceEmbedding:
    """Creates an IBM Watsonx Embedding model for vector representation.
    
    Returns:
        WatsonxEmbeddings model.
    """
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embed_model

def create_llm(
    temperature: float = 0.0,
    max_new_tokens: int = 500,
    top_p: str = "0.9"
) -> Ollama:
    """Creates an IBM Watsonx LLM for generating responses.
    
    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: Decoding method to use (sample, greedy).
        
    Returns:
        WatsonxLLM model.
    """
    llm = Ollama(
        model="qwen3",   # use your exact model name from `ollama list`
        request_timeout=240,
        temperature=temperature,   # direct param supported
        additional_kwargs={
            "num_predict": max_new_tokens,        # max_new_tokens equivalent
            "top_p": 0.9               # used for sampling
        }
    )

    return llm
