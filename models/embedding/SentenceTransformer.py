from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch.nn.functional as F
import torch

from models.embedding.Model import EmbeddingModel


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model_mgr, model_path=DEFAULT_MODEL):
        model = AutoModel.from_pretrained(model_path, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        name = model_path.split("/")[-1]

        super().__init__(model_mgr, model, tokenizer, name)

    def embedd(self, text: str | list[str]) -> np.array:
        self.model_to_cuda()

        tokens = self._tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        if torch.cuda.is_available():
            tokens = tokens.to("cuda")

        with torch.no_grad():
            model_output = self._model(**tokens)
            sentence_embeddings = self._mean_pooling(model_output, tokens['attention_mask'])

        return F.normalize(sentence_embeddings, p=2, dim=1).detach().cpu().numpy()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
