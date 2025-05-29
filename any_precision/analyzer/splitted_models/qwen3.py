import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Union, List, Tuple
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

logger = logging.get_logger(__name__)

class SplittedQwen3Model(Qwen3Model):

    def set_devices(self):
        num_visible_devices = torch.cuda.device_count()
        assert num_visible_devices > 0, "Must use at least one GPU"
        self.split_gpus = num_visible_devices > 1
        print(f"splitting into {num_visible_devices} GPUs")
        if not self.split_gpus:
            self.cuda()
        else:
            # For larger model, we need to split the model into multiple GPUs
            # assign the embedding and norm onto the 1st devide
            self.embed_tokens.to(f"cuda:0")
            self.rotary_emb.to(f"cuda:0")
            self.norm.to(f"cuda:0")
            # layers are divided into #(num GPUs) chunks
            self.split_indices = []
            prev_device = 0
            nums = len(self.layers) // num_visible_devices
            for i, layer in enumerate(self.layers):
                device = min(num_visible_devices - 1, i // nums)
                if prev_device != device:
                    self.split_indices.append(i)
                print(f"Moving layer {i} to cuda:{device}")
                layer.to(f"cuda:{device}")
                prev_device = device


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        device = 0
        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Move activations to the next device at the split points
            if self.split_gpus and idx in self.split_indices:
                device += 1
                hidden_states = hidden_states.to(f"cuda:{device}")
                new_position_embeddings = []
                for position_embedding in position_embeddings:
                    new_position_embeddings.append(position_embedding.to(f"cuda:{device}"))
                position_embeddings = tuple(new_position_embeddings)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            # Move activations back to the 1st device at the end
            if self.split_gpus and idx == len(self.layers) - 1:
                hidden_states = hidden_states.to("cuda:0")
                new_position_embeddings = []
                for position_embedding in position_embeddings:
                    new_position_embeddings.append(position_embedding.to(f"cuda:0"))
                position_embeddings = tuple(new_position_embeddings)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
