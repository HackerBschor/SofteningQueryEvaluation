from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

from models.embedding.Model import EmbeddingModel

class LLaMAEmbeddingModel(EmbeddingModel):
    DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, model_mgr, model_path=DEFAULT_MODEL):
        config_kwargs = {
            "trust_remote_code": True,
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token": None,
            "output_hidden_states": True,
            "return_dict_in_generate": True
        }

        model_config = AutoConfig.from_pretrained(model_path, **config_kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            config=model_config,
            device_map="cpu",
            torch_dtype=torch.float16)

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.pad_token = "[PAD]"
        # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        super().__init__(model_mgr, model, tokenizer)


    def embedd(self, text: str | list[str], max_len: int = 128) -> np.array:
        self.model_to_cuda()

        tokens = self._tokenizer(text, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)

        if torch.cuda.is_available():
            tokens = tokens.to("cuda")

        with torch.no_grad():
            batch_outputs: torch.Tensor = self._model(**tokens)
            embeddings = torch.mean(batch_outputs.hidden_states[-1], axis=1)

        return embeddings.cpu().detach().numpy()
