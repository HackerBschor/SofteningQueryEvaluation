import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.semantic_validation.Model import SemanticValidationModel
from models import ModelMgr

class LLaMAValidationModel(SemanticValidationModel):
    DEFAULT_SYSTEM_PROMPT = "You are a validator. Validate the following statement using \"no\" and \"yes\" only!"
    DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, model_mgr: ModelMgr, system_prompt=DEFAULT_SYSTEM_PROMPT, model_path=DEFAULT_MODEL, temperature: float | None = None):
        self._system_prompt = system_prompt

        model = AutoModelForCausalLM.from_pretrained(model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            return_dict_in_generate=True)

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        self.generation_config = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": 1,
            "top_p": None,
            "temperature": temperature,
            "do_sample":  False
        }

        name = model_path.split("/")[-1]

        super().__init__(model_mgr, model, tokenizer, name)



    def __call__(self, prompt: str, system_prompt: str | None = None) -> bool:
        self.model_to_cuda()

        system_prompt = system_prompt if system_prompt else self._system_prompt
        chat_template = [ {'content': system_prompt, 'role': 'system'}, {'content': prompt, 'role': 'user'} ]

        logging.debug(f"Chat template: {chat_template}")

        prompt_tok: torch.Tensor = self._tokenizer.apply_chat_template(
            chat_template, add_generation_prompt=True, return_tensors="pt")

        attention_mask = (prompt_tok != self._tokenizer.pad_token_id).long()

        if torch.cuda.is_available():
            self._model.to("cuda")
            prompt_tok = prompt_tok.cuda()
            attention_mask = attention_mask.cuda()

        answer_tok = self._model.generate(prompt_tok, attention_mask = attention_mask, **self.generation_config)
        answer = self._tokenizer.decode( answer_tok[0][len(prompt_tok[0]):], skip_special_tokens=True ).lower()

        if answer not in ("yes", "no"):
            logging.error(f"{answer} is not a valid answer")

        return answer == "yes"
