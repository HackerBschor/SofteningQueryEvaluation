from models import Model

class TextGenerationModel(Model):
    def __call__(self, prompt: str, system_prompt: str | None = None, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        raise NotImplementedError()

