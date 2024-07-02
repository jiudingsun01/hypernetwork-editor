from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2FlashAttention2,
)

from ..utils import (
    EditorConfig,
    EditorModelOutput,
    add_fwd_hooks,
    assign_layer_indices,
)
from .layers import InterpretorUnembedCrossAttention

T = TypeVar("T", bound="GPT2Interpretor")


class GPT2InterpretorConfig(GPT2Config, EditorConfig):
    compute_position_ids: bool = True
    default_intervention_layer: int = 6


class GPT2InterpretorHypernetwork(GPT2LMHeadModel):
    _tied_weights_keys = []

    def __init__(self, config: GPT2InterpretorConfig):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        
        # only LM head gets special attention
        if config._attn_implementation == "flash_attention_2":
            _attention_cls = GPT2FlashAttention2
        else:
            _attention_cls = GPT2Attention

        self.lm_head = InterpretorUnembedCrossAttention(
            config=config, layer_idx=config.chop_editor_at_layer
        )

        # prune layers and add cross attn heads
        self.transformer.h = self.transformer.h[: config.chop_editor_at_layer]
        if config.cross_attn_layers == []:
            config.cross_attn_layers = list(range(config.chop_editor_at_layer))

        for i, layer in enumerate(self.transformer.h):
            if i not in config.cross_attn_layers:
                continue
            layer.crossattention = _attention_cls(
                config=config, layer_idx=i, is_cross_attention=True
            )
            layer.ln_cross_attn = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_epsilon
            )
            original_query_weights = layer.attn.c_attn.weight[:, : config.hidden_size]
            original_keys_values = layer.attn.c_attn.weight[:, config.hidden_size :]
            original_query_bias = layer.attn.c_attn.bias[: config.hidden_size]
            original_keys_values_bias = layer.attn.c_attn.bias[config.hidden_size :]

            # with torch.no_grad():
            # Initialize the new layer with these parameters
            layer.crossattention.q_attn.weight = nn.Parameter(original_query_weights)
            layer.crossattention.q_attn.bias = nn.Parameter(original_query_bias)
            layer.crossattention.c_attn.weight = nn.Parameter(original_keys_values)
            layer.crossattention.c_attn.bias = nn.Parameter(original_keys_values_bias)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        base_encoder_hidden_states: Optional[torch.Tensor] = None,
        base_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        source_encoder_hidden_states: Optional[torch.Tensor] = None,
        source_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_intervention_ratio: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # set device for input_ids to cuda ?
        # input_ids = input_ids.to(self.lm_head.weight.device)
        if (
            attention_mask is not None
            and position_ids is None
            and self.config.compute_position_ids
        ):
            position_ids = attention_mask.cumsum(-1)
        
        encoder_hidden_states = torch.cat(
            (base_encoder_hidden_states, source_encoder_hidden_states), dim=1
        )
        
        encoder_attention_mask = torch.cat(
            (base_encoder_attention_mask, source_encoder_attention_mask), dim=1
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        reverse_attention_output = self.lm_head(
            hidden_states,
            attention_mask=attention_mask,
            base_encoder_hidden_states=base_encoder_hidden_states,
            base_encoder_attention_mask=base_encoder_attention_mask,
            source_encoder_hidden_states=source_encoder_hidden_states,
            source_encoder_attention_mask=source_encoder_attention_mask,
            output_attentions=output_attentions,
        )

        # (output, present[,attentions])
        return reverse_attention_output


class GPT2Interpretor(nn.Module):
    def __init__(self, config: GPT2InterpretorConfig):
        super().__init__()

        self.config = config
        self.hypernetwork = GPT2InterpretorHypernetwork(config)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path
        ).eval()

        # freeze target model
        for param in self.target_model.parameters():
            param.requires_grad = False

        assign_layer_indices(self.target_model)

        if config.use_layerwise_embeddings:
            # extra layer is cross-attn in the lm_head
            self.layerwise_embeddings = nn.Parameter(
                torch.zeros(config.n_layer + 1, config.n_embd), requires_grad=True
            )
            self.layerwise_embeddings.data.normal_(
                mean=0.0, std=self.target_model.config.initializer_range
            )
        else:
            self.layerwise_embeddings = None

    def train(self: T, mode: bool = True) -> T:
        return self.hypernetwork.train(mode)

    def eval(self: T) -> T:
        return self.hypernetwork.eval()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Only load weights for the trainable hypernetwork."""
        self.hypernetwork.load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def _run_target_model_for_encoded_hidden_states(
        self,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
    ):
        """Gets the hidden states from the target model, if necessary"""

        if position_ids is not None:
            outputs = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )

        else:
            outputs = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                output_hidden_states=True,
            )

        return outputs.hidden_states

    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        editor_attention_mask: torch.Tensor = None,
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        source_input_ids: torch.Tensor = None,
        source_attention_mask: torch.Tensor = None,
        base_hidden_states: torch.Tensor = None,
        base_position_ids: torch.Tensor = None,
        source_hidden_states: torch.Tensor = None,
        source_position_ids: torch.Tensor = None,
        intervention_layer: int = None,
        output_target_hidden_states: bool = False,
        output_edited_hidden_states: bool = False,
        output_intervention_ratio: bool = False,
        batch_intervention_ratio: torch.Tensor = None,
    ) -> EditorModelOutput:
        
        if intervention_layer is None:
            intervention_layer = self.config.default_intervention_layer        
        
        # Run target model for encoded hidden states
        if base_hidden_states is None:
            base_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    base_input_ids, base_attention_mask, base_position_ids
                ),  # seems to break while we are passing thru batch_size=1; the last (12th =) has different dimensions
                dim=2,
            )
        
        if source_hidden_states is None:
            source_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    source_input_ids, source_attention_mask, source_position_ids
                ),
                dim=2,
            )
            
        # dimensions of target_hidden_states:
        # batch_size, token_sequence_length, num_layers = 13, resid_width = 768
        # Normalize along the last dimension
        base_normalization_factors = base_hidden_states.norm(dim=-1, keepdim=True)
        base_hidden_states = base_hidden_states / base_normalization_factors
        
        source_normalization_factors = source_hidden_states.norm(dim=-1, keepdim=True)
        source_hidden_states = source_hidden_states / source_normalization_factors

        # Error catching:
        
        # batch_intervention_ratio = (batch_size, source_token_sequence_length, base_token_sequence_length)
        if batch_intervention_ratio is not None:
            if output_intervention_ratio:
                raise ValueError(
                    "Inputting your own batch_intervention_ratio means the model does not construct the outputs you are requesting"
                )

        # Run editor model, get edit vectors
        if batch_intervention_ratio is None:
            
            n_layer = base_hidden_states.shape[2]
            
            # collapsed_base_hidden_states (batch_size, token_sequence_length * num_layers, resid_width)
            collapsed_base_hidden_states = base_hidden_states.reshape(
                base_hidden_states.shape[0],
                base_hidden_states.shape[1] * base_hidden_states.shape[2],
                base_hidden_states.shape[3],
            )
            # collapsed_base_attention_mask (batch_size, token_sequence_length * num_layers)
            collapsed_base_attention_mask = base_attention_mask.repeat(1, n_layer)
            
            collapsed_source_hidden_states = source_hidden_states.reshape(
                source_hidden_states.shape[0],
                source_hidden_states.shape[1] * source_hidden_states.shape[2],
                source_hidden_states.shape[3],
            )
            
            collapsed_source_attention_mask = source_attention_mask.repeat(1, n_layer)

            interpretor_output = self.hypernetwork(
                input_ids=editor_input_ids,
                attention_mask=editor_attention_mask,
                base_encoder_hidden_states=collapsed_base_hidden_states,
                base_encoder_attention_mask=collapsed_base_attention_mask,
                source_encoder_hidden_states=collapsed_source_hidden_states,
                source_encoder_attention_mask=collapsed_source_attention_mask,
                output_intervention_ratio=output_intervention_ratio,
            )

            # Multiply the outputs by normalization factors
            intervention_output, _, batch_intervention_ratio = interpretor_output
            
            batch_intervention_ratio = batch_intervention_ratio.squeeze()
            
            
        source_output = self.target_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
        )
        
        source_hidden_states = source_output.hidden_states[intervention_layer]
        intervention_matrix = torch.einsum("bij,bid->bijd", batch_intervention_ratio, source_hidden_states)
        intervention_matrix = intervention_matrix.sum(dim=1)
        base_intervention_weights = torch.sum(batch_intervention_ratio, dim=1)
        
        # Run target model with edit vectors.
        # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
        def representation_swap(module, input, output):
            res_diff = torch.einsum("bid,bi->bid", output[0].clone(), base_intervention_weights)
            output[0][:] += (intervention_matrix - res_diff)
            
        def embedding_representation_swap(module, input, output):
            res_diff = torch.einsum("bid,bi->bid", output.clone(), base_intervention_weights)
            output[:] += (intervention_matrix - res_diff)          
        
        # Now editing the target model
        if intervention_layer == 0:
            hooks = [(self.target_model.transformer.wte, embedding_representation_swap)]
        else:
            hooks = [(self.target_model.transformer.h[intervention_layer - 1], representation_swap)]

        with add_fwd_hooks(hooks):
            # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
            # THE END OF `layer` AS A SIDE EFFECT)
            target_result = self.target_model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                position_ids=base_position_ids,
                output_hidden_states=output_edited_hidden_states,
            )
    
        logits = target_result.logits
        
        output = EditorModelOutput(logits=logits)
        if output_edited_hidden_states:
            output.edited_hidden_states = target_result.hidden_states
        return output
