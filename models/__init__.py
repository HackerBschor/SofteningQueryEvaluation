import torch

from huggingface_hub import login

class ModelMgr:
    def __init__(self):
        self.current_model_gpu: Model | None = None

    def get_gpu_resources(self, model):
        if self.current_model_gpu is None:
            self.current_model_gpu = model
        elif self.current_model_gpu != model:
            self.current_model_gpu.model_to_cpu()
            self.current_model_gpu = model
        else:
            pass


class Model:
    def __init__(self, model_mgr, model, tokenizer, huggingface_token=None):
        self._model = model
        self._tokenizer = tokenizer
        self._model_mgr: ModelMgr = model_mgr

        if huggingface_token is not None:
            login(huggingface_token)

    def model_to_cpu(self):
        self._model.cpu()

    def model_to_cuda(self):
        if torch.cuda.is_available():
            self._model_mgr.get_gpu_resources(self)
            self._model.cuda()
