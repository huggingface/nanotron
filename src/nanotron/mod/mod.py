import torch
from torch import nn
from torchtyping import TensorType
import torch.nn.functional as F

class MixtureOfDepth(nn.Module):
    def __init__(self, capacity: int, d_model: int, block: nn.Module):
        super().__init__()
        self.router = Router(capacity, d_model)
        self.block = block
    
    def forward(self, inputs: TensorType["batch_size", "seq_len", "d_model"]) -> TensorType["batch_size", "seq_len", "d_model"]:
        selected_idxs = self.router(inputs)
        assert selected_idxs.shape == (inputs.size(0), self.router.capacity)
        selected_inputs = inputs[torch.arange(inputs.size(0)).unsqueeze(1), selected_idxs]
        
        outputs_of_selected_inputs = self.block(selected_inputs)
        # NOTE: now keep the representation of the selected inputs and replace the original inputs with the new ones
        inputs[torch.arange(inputs.size(0)).unsqueeze(1), selected_idxs] = outputs_of_selected_inputs
        return inputs
    

class Router(nn.Module):
    def __init__(self, capacity: int, d_model: int):
        super().__init__()
        self.capacity = capacity
        self.gate = nn.Linear(d_model, 1)
        
    def forward(self, inputs: TensorType["batch_size", "seq_len", "d_model"]) -> TensorType["batch_size", "seq_len"]:
        probs = F.softmax(self.gate(inputs), dim=1).view(-1, inputs.size(1))
        _, top_k_indices = torch.topk(probs, self.capacity)
        return top_k_indices
