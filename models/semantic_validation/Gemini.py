import logging
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from models.semantic_validation.Model import SemanticValidationModel
from utils import get_config


class GeminiValidationModel(SemanticValidationModel):
    DEFAULT_SYSTEM_PROMPT = "You are a validator. You get a statement and need to validate it. Answer with \"yes\" and \"no\" only!"
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: str, system_prompt=DEFAULT_SYSTEM_PROMPT, model_name=DEFAULT_MODEL):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        super().__init__(None, None, None)


    def __call__(self, prompt: str, system_prompt: str | None = None) -> bool:
        model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
        generation_config = genai.GenerationConfig(max_output_tokens=3, temperature=0.1)

        try:
            response = model.generate_content(prompt, generation_config=generation_config)
        except ResourceExhausted:
            logging.debug(f"Resource exhausted. Sleeping for 60s")
            time.sleep(60)
            return self(prompt)

        answer = response.text.lower().strip()
        if answer not in ["yes", "no"]:
            logging.error(f"Unexpected answer {answer}  ")


        return answer == "yes"
