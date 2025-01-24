from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from models.embedding.Model import EmbeddingModel

class GenericEmbeddingModel(EmbeddingModel):
    DEFAULT_MODEL = "intfloat/e5-base-v2"

    def __init__(self, model_mgr, model_path=DEFAULT_MODEL):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModel.from_pretrained(
            model_path,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            # low_cpu_mem_usage=True,
        )

        super().__init__(model_mgr, model, tokenizer)

    def embedd(self, text: str | list[str]) -> np.array:
        self.model_to_cuda()

        with torch.no_grad():
            inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            outputs = self._model(**inputs)

            last_hidden = outputs.last_hidden_state.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
            doc_embeds = last_hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]

            embeddings = doc_embeds.detach().cpu().numpy()

            del inputs, outputs, last_hidden, doc_embeds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return embeddings

