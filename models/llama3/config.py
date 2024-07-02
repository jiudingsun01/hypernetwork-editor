from transformers import (
    LlamaConfig,
)

from ..utils import (
    EditorConfig,
)


class LlamaEditorConfig(LlamaConfig, EditorConfig):
    init_attn_proj_bias: bool = True
    compute_position_ids: bool = True
