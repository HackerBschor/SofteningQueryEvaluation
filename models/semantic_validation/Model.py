from models import Model

class SemanticValidationModel(Model):
    def __call__(self, prompt: str) -> bool:
        raise NotImplementedError()