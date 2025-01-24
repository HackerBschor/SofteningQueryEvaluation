import logging
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from models.semantic_validation import SemanticValidator
from utils import get_config


class Gemini_Validator(SemanticValidator):
    DEFAULT_SYSTEM_PROMPT = "You are a validator. You get a statement and need to validate it. Answer with \"yes\" and \"no\" only!"
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, config: str, system_prompt=DEFAULT_SYSTEM_PROMPT, model_name=DEFAULT_MODEL):
        config = get_config(config)
        genai.configure(api_key=config["MODEL"]["google_aistudio_api_key"])
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
        super().__init__(model, None)



    def __call__(self, prompt: str) -> bool:
        generation_config = genai.GenerationConfig(max_output_tokens=3, temperature=0.1, )

        try:
            response = self._model.generate_content(prompt, generation_config=generation_config)
        except ResourceExhausted:
            logging.debug(f"Resource exhausted. Sleeping for 60s")
            time.sleep(60)
            return self(prompt)

        answer = response.text.lower().strip()

        if answer not in ("yes", "no"):
            logging.debug(f"LLM response not in the right format (yes/ no). Response: '{answer}'")

        return answer == "yes"
