import os
import google.generativeai as genai
from .base import AIPlatform

class Gemini(AIPlatform):
    def __init__(self, api_key, system_prompts: str = None):
        self.api_key = api_key
        self.system_prompt = system_prompts
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-3.1-flash-lite")

    def chat(self, prompt: str) -> str:
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"

        response = self.model.generate_content(prompt)
        return response.text
    
