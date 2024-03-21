import collections
import contextlib

from torch import nn


class Store(collections.defaultdict):
    """
    We use the store to locally store on gpu some states so that we don't have to communicate.
    This is useful at inference if we don't want to recompute kv_cache for example, or that we don't want to communicate it through the pipeline
    """

    def __init__(self):
        super().__init__(dict)

    def flush(self):
        # TODO @thomasw21: There's probably a simpler way than doing this.
        for key in list(self.keys()):
            del self[key]


class AttachableStore:
    def _attach_store(self, store: Store):
        assert not hasattr(self, "_store"), "You can't assign a store when there's already one attached"
        self._store = store

    def _detach_store(self):
        delattr(self, "_store")

    def get_local_store(self):
        if hasattr(self, "_store"):
            if isinstance(self, nn.Module):
                assert self.training is False, "Store is used only in evaluation mode"
            return self._store[id(self)]
        else:
            return None


@contextlib.contextmanager
def attach_store(model: nn.Module, store: Store):
    list_module_containing_store = []
    for module in model.modules():
        if not isinstance(module, AttachableStore):
            continue
        module._attach_store(store)
        list_module_containing_store.append(module)

    try:
        yield
    finally:
        for module in list_module_containing_store:
            module._detach_store()
