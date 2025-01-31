import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from models.text_generation.Model import TextGenerationModel
from models import ModelMgr

class DeepSeekGenerationModel(TextGenerationModel):
    DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    def __init__(self, model_mgr: ModelMgr, model_path=DEFAULT_MODEL):
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(model_mgr, model, tokenizer)



    def __call__(self, prompt: str, system_prompt: str | None = None, max_new_tokens=100, temperature=0.7) -> str:
        self.model_to_cuda()

        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=self._tokenizer.eos_token_id)
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
