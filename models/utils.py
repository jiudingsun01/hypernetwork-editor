import contextlib
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput


class EditorConfig(PretrainedConfig):
    edit_channel_width_factor: int = 2
    chop_editor_at_layer: int = -1
    num_editing_heads: int = 32
    use_layerwise_embeddings: bool = True
    edit_dampening_factor: float = 0.001
    kill_token_zero: bool = False


@dataclass
class EditorModelOutput(BaseModelOutput):
    logits: torch.Tensor
    target_hidden_states = Optional[torch.Tensor] = None
    edited_hidden_states = Optional[torch.Tensor] = None
    edit_vectors = Optional[torch.Tensor] = None
    editor_attention = Optional[torch.Tensor] = None


@contextlib.contextmanager
def add_fwd_hooks(module_hooks):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


def assign_layer_indices(model):
    """
    Assigns a custom attribute 'layer_index' to each transformer layer in the GPT-2 model.
    This function iterates over the transformer blocks and assigns an index to each.
    """
    model.transformer.wte.layer_index = 0
    for i, layer in enumerate(model.transformer.h):
        layer.layer_index = i + 1