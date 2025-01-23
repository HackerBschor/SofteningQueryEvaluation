import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SemanticValidation:
    DEFAULT_SYSTEM_PROMPT = "You are a validator. You get a statement and need to validate it. Answer with \"yes\" and \"no\" only!"
    DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, system_prompt=DEFAULT_SYSTEM_PROMPT, model_path=DEFAULT_MODEL):
        self.system_prompt = system_prompt

        self.model = AutoModelForCausalLM.from_pretrained(model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            return_dict_in_generate=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, prompt: str, system_prompt: str | None = None) -> bool:
        system_prompt = system_prompt if system_prompt else self.system_prompt
        chat_template = [ {'content': system_prompt, 'role': 'system'}, {'content': prompt, 'role': 'user'} ]

        prompt_tok: torch.Tensor = self.tokenizer.apply_chat_template(
            chat_template, add_generation_prompt=True, return_tensors="pt")

        attention_mask = (prompt_tok != self.tokenizer.pad_token_id).long()

        if torch.cuda.is_available():
            self.model.to("cuda")
            prompt_tok = prompt_tok.cuda()
            attention_mask = attention_mask.cuda()

        answer_tok = self.model.generate(prompt_tok,
            attention_mask = attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=3,
        )

        answer = self.tokenizer.decode( answer_tok[0][len(prompt_tok[0]):], skip_special_tokens=True ).lower()

        if answer not in ("yes", "no"):
            logging.debug(f"LLM response not in the right format (yes/ no). Response: '{answer}'")

        return answer == "yes"



