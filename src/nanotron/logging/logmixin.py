import torch
from torch import nn
from typing import Dict, List, Any

class LogMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tbi_logs: List[Dict[str, torch.Tensor]] = []

    def tbi_logger(self, log_data: Dict[str, torch.Tensor]):
        """
        Logs a dictionary of named tensors.
        Tensors are stored by reference (i.e., on their original device, e.g., CUDA).
        If you need a snapshot at the exact moment of logging and the tensor might be
        modified in-place later, consider storing {key: tensor.clone() for key, tensor in log_data.items()}.
        """
        # Example check (optional):
        # for tensor in log_data.values():
        #     if not tensor.is_cuda:
        #         # Or raise an error, or move to CUDA, depending on desired behavior
        #         print(f"Warning: Tensor {list(log_data.keys())[list(log_data.values()).index(tensor)]} is not on CUDA.")
        self._tbi_logs.append(log_data)

    def _get_internal_logs(self) -> List[Dict[str, torch.Tensor]]:
        """
        Retrieves the logs stored by this module instance.
        """
        return self._tbi_logs

    def _clear_internal_logs(self):
        """
        Clears the logs stored by this module instance.
        Important for managing memory, should be called after logs are processed.
        """
        self._tbi_logs = []


class LoggingCollectorMixin:
    """
    A mixin class for nn.Module-based models to collect logs from submodules
    that use LogMixin.
    The class this is mixed into must be an nn.Module or its subclass.
    """
    # No __init__ is strictly necessary here if the mixin itself doesn't have
    # its own state to initialize. The methods operate on `self` which will be
    # an instance of the class it's mixed into (e.g., Qwen2ForTraining).
    # If an __init__ were added, it should also call super().__init__(*args, **kwargs).

    def get_tbi_logs(self, non_blocking: bool = False) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """
        Collects all TBI logs from modules that use LogMixin.
        Returns a dictionary where keys are fully qualified module names and 
        values are lists of log entries (each entry being a dictionary of tensors).
        Tensors remain on their original CUDA devices.
        Assumes `self` is an nn.Module instance with `named_modules()` method.
        """
        all_logs: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        # `self` refers to the instance of the class LoggingCollectorMixin is mixed into.
        # This class is expected to be an nn.Module or subclass.
        for name, module in self.named_modules(): 
            if isinstance(module, LogMixin):
                module_logs = module._get_internal_logs()
                if module_logs:  # Only add if there are logs for this module
                    for entry in module_logs:
                        for k, v in entry.items():
                            all_logs[name + "/" + k] = v.detach().to(device="cpu", non_blocking=non_blocking)
        return all_logs

    def clear_all_tbi_logs(self):
        """
        Clears TBI logs from all modules that use LogMixin.
        This should be called after processing the logs (e.g., after a forward/backward pass)
        to free up memory.
        Assumes `self` is an nn.Module instance with `modules()` method.
        """
        # `self` refers to the instance of the class LoggingCollectorMixin is mixed into.
        for module in self.modules(): 
            if isinstance(module, LogMixin):
                try:
                    module._clear_internal_logs()
                except AttributeError:
                    # Similar to get_tbi_logs, handle cases where mixin might not be fully initialized.
                    pass
