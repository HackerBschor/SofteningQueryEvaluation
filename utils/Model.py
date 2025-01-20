import configparser
import logging

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

from utils import get_config, TColor

from huggingface_hub import login

from openai import OpenAI


class Model:
    def __init__(self, config: str, model_type: str = None):
        config: configparser.ConfigParser = get_config(config)

        if "huggingface_token" in config["MODEL"]:
            login(config["MODEL"]["huggingface_token"])

        self._model_path: str = config['MODEL'][f"path_{model_type}"]
        self._use_cuda = torch.cuda.is_available() and config['MODEL'][f"use_cuda"] == "True"
        self._device = "cuda:0" if self._use_cuda else "cpu"

        self._tokenizer = self._load_tokenizer()


        logging.debug(f"Load model {self._model_path}")
        self._model = self._load_model()

    def _tokenize(self, text: str, **kwargs) -> dict[str, torch.Tensor]:
        inputs: dict[str, torch.Tensor] = self._tokenizer(text, **kwargs)

        if torch.cuda.is_available():
            inputs = {key: inputs[key].cuda() for key in inputs}

        return inputs

    def _load_model(self):
        pass

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        if "llama" in self._model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer


class EmbeddingModel(Model):
    def __init__(self, config, model_type: str = 'embeddings'):
        super().__init__(config, model_type)
        self.embeddings_size = self._model.config.hidden_size

    def _load_model(self):
        model = AutoModel.from_pretrained(
            self._model_path,
            # low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
        )

        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        return model

    def embedd(self, text: str | list[str]) -> np.array:
        with torch.no_grad():
            inputs = self._tokenize(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self._model(**inputs)

            last_hidden = outputs.last_hidden_state.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
            doc_embeds = last_hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]

            embeddings = doc_embeds.detach().cpu().numpy()

            del inputs, outputs, last_hidden, doc_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return embeddings

    def embedd_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            embeddings.extend(self.embedd(texts[i: max(len(texts), i + batch_size)]))

        return np.array(embeddings)

    def get_embedding_shape(self):
        return self._model.config.hidden_size


class LLaMAModel(EmbeddingModel):
    def __init__(self, config):
        super().__init__(config, "llama")

    def _load_model(self):
        config_kwargs = {
            "trust_remote_code": True,
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token": None,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }
        model_config = AutoConfig.from_pretrained(self._model_path, **config_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            config=model_config,
            device_map=self._device,
            torch_dtype=torch.float16)

        model.eval()

        return model

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.pad_token = "[PAD]"
        #tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def embedd(self, text: str | list[str], max_len: int = 128) -> np.array:
        tokens = self._tokenizer(text, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        tokens = tokens.to(self._device)

        with torch.no_grad():
            batch_outputs = self._model(**tokens)
            embeddings = torch.mean(batch_outputs.hidden_states[-1], axis=1)

        return embeddings.cpu().detach().numpy()


class SentenceTransformers(EmbeddingModel):
    def __init__(self, config):
        super().__init__(config, "sentence_transformers")

    def _load_model(self):
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

        if self._use_cuda:
            model = model.cuda()

        return model

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self._model_path)

    def embedd(self, text: str | list[str], max_len: int = 128) -> np.array:
        encoded_input = self._tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        if self._use_cuda:
            encoded_input = {k: v.cuda() for k,v in encoded_input.items()}

        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            return F.normalize(sentence_embeddings, p=2, dim=1).detach().cpu().numpy()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class GenerationModel(Model):
    def __init__(self, config):
        super().__init__(config, 'generation')
        # self._tokenizer.pad_token = self._tokenizer.eos_token
        # self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            # low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16
        )

        # model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        return model

    def generate(self,
                 prompt: str,
                 max_length: int | None = None,
                 max_new_tokens: int | None = None,
                 temperature: float = 0.7,
                 top_p: float = 0.0,
                 top_k: float = 40,
                 do_sample: bool = True):
        inputs = self._tokenize(prompt, return_tensors="pt")

        if max_length is None and max_new_tokens is None:
            max_length = self._tokenizer.model_max_length

        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=self._tokenizer.eos_token_id,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                do_sample=do_sample
            )

        answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer

class OpenAiModel():
    def __init__(self, config):
        self._config: configparser.ConfigParser = get_config(config)

    def generate(self):
        client = OpenAI(api_key=self._config['MODEL']['open_ai_key'])
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say this is a test"}],
            stream=True,
        )
        print(completion)
