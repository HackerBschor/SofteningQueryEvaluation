class ModelMgr:
    def __init__(self):
        self.current_model_gpu: Model | None = None

    def to_gpu(self, model):
        if self.current_model_gpu is None:
            self.current_model_gpu = model
        elif self.current_model_gpu != model:
            self.current_model_gpu.model_to_cpu()
            self.current_model_gpu = model
        else:
            pass


class Model:
    def __init__(self, model, model_mgr):
        self._model = model
        self._model_mgr: ModelMgr = model_mgr

    def model_to_cpu(self):
        self._model.cpu()

    def model_to_cuda(self):
        self._model_mgr.to_gpu(self)
