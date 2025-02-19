import logging

import torch
from ollama import chat
from ollama import ChatResponse

from models import ModelMgr
from models.semantic_validation.Model import SemanticValidationModel

class DeepSeekValidationModel(SemanticValidationModel):
    DEFAULT_SYSTEM_PROMPT = "You are a validator. You get a statement and need to validate it. Answer with \"yes\" and \"no\" only!"

    def __init__(self, model_mgr: ModelMgr, default_system_prompt=DEFAULT_SYSTEM_PROMPT):
        self._default_system_prompt = default_system_prompt

        super().__init__(model_mgr, None, None)

    def model_to_cuda(self):
        # Handle by OLLAMA
        pass

    def model_to_cpu(self):
        # Handle by OLLAMA
        pass


    def __call__(self, prompt: str, system_prompt: str | None = None) -> bool:
        if torch.cuda.is_available():
            self._model_mgr.get_gpu_resources(self)

        system_prompt = self._default_system_prompt if system_prompt is None else system_prompt

        messages = [{'role': 'system', 'content': system_prompt}] if system_prompt is not None else[]
        messages.append({'role': 'user', 'content': prompt})

        response: ChatResponse = chat(model='deepseek-r1:7b', messages=messages)
        answer = response.message.content.split("</think>")[1].lower().strip()
        if answer not in ("yes", "no"):
            logging.error(f"{answer} is not a valid answer")

        return answer == "yes"
