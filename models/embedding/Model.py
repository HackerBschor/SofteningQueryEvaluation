import numpy as np
from models import Model

class EmbeddingModel(Model):
    def __call__(self, text: str | list[str], batch_size = 32) -> np.array:
        if isinstance(text, str):
            embedding = self.embedd(text)
            return embedding[0] if len(embedding.shape) > 0 else embedding
        else:
            embeddings = []
            for batch in range(0, len(text), batch_size):
                embeddings.append(self.embedd(text[batch : batch + batch_size]))

            return np.concatenate(embeddings)

    def embedd(self, text: str | list[str]) -> np.array:
        raise NotImplementedError()

    def get_embedding_size(self) -> int:
        return self._model.config.hidden_size
