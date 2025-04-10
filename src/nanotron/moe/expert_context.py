import torch


class ExpertContext:
    _instance = None

    def __init__(self):
        self.aux_loss = []

    def push_aux_loss(self, aux_loss: torch.Tensor):
        self.aux_loss.append(aux_loss)

    def pop_all_aux_loss(self) -> list[torch.Tensor]:
        aux_loss, self.aux_loss = self.aux_loss, []
        return aux_loss

    @classmethod
    def get_instance(cls) -> "ExpertContext":
        if not cls._instance:
            cls._instance = ExpertContext()
        return cls._instance
