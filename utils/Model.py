import configparser

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from utils import get_config, TColor


class Model:
    def __init__(self, config: str, model_type: str = None):
        config: configparser.ConfigParser = get_config(config)

        self._model_path: str = config['MODEL'][f"path_{model_type}"]

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        print(f"Load model {self._model_path}", end=" ")
        self._model = self._load_model()
        print(f"{TColor.OKGREEN}Done{TColor.ENDC}")

    def _tokenize(self, text: str, **kwargs) -> dict[str, torch.Tensor]:
        inputs: dict[str, torch.Tensor] = self._tokenizer(text, **kwargs)

        if torch.cuda.is_available():
            inputs = {key: inputs[key].cuda() for key in inputs}

        return inputs

    def _load_model(self):
        pass


class EmbeddingModel(Model):
    def __init__(self, config):
        super().__init__(config, 'embeddings')

    def _load_model(self):
        model = AutoModel.from_pretrained(
            self._model_path,
            # low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
        )

        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        return model

    def embedd(self, text):
        with torch.no_grad():
            inputs = self._tokenize(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self._model(**inputs)

            last_hidden = outputs.last_hidden_state.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
            doc_embeds = last_hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]

            embeddings = doc_embeds.detach().cpu().numpy()

            del inputs, outputs, last_hidden, doc_embeds
            torch.cuda.empty_cache()

        return embeddings


class GenerationModel(Model):
    def __init__(self, config):
        super().__init__(config, 'generation')
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            # low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16
        )

        model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        return model

    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0, top_p: float = 0.9):
        inputs = self._tokenize(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer


if __name__ == '__main__':
    # print(GenerationModel(config_file="../config.ini").generate("Hallo Welt"))
    print(EmbeddingModel("../config.ini").embedd(["Hallo", "Embeddings?"]).shape)