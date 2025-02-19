from models import Model

class SemanticValidationModel(Model):
    def __call__(self, prompt: str, system_prompt: str | None = None) -> bool:
        raise NotImplementedError()