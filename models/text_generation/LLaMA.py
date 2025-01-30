import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.text_generation.Model import TextGenerationModel
from models import ModelMgr

class LLaMATextGenerationModel(TextGenerationModel):
    DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, model_mgr: ModelMgr, model_path=DEFAULT_MODEL):
        model = AutoModelForCausalLM.from_pretrained(model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            return_dict_in_generate=True)

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        super().__init__(model_mgr, model, tokenizer)



    def __call__(self, prompt: str, system_prompt: str | None = None, max_new_tokens=100, temperature=0.7) -> str:
        self.model_to_cuda()

        chat_template = [ {'content': prompt, 'role': 'user'} ]

        if system_prompt is not None:
            chat_template.append({'content': system_prompt, 'role': 'system'})

        prompt_tok: torch.Tensor = self._tokenizer.apply_chat_template(
            chat_template, add_generation_prompt=True, return_tensors="pt")

        attention_mask = (prompt_tok != self._tokenizer.pad_token_id).long()

        if torch.cuda.is_available():
            self._model.to("cuda")
            prompt_tok = prompt_tok.cuda()
            attention_mask = attention_mask.cuda()

        answer_tok = self._model.generate(prompt_tok,
            attention_mask = attention_mask,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        return self._tokenizer.decode( answer_tok[0][len(prompt_tok[0]):], skip_special_tokens=True ).lower()
