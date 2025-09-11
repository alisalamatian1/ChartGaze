from dataclasses import dataclass
import dataclasses
from typing import List, Optional, Tuple, Union
import ast
import re
from enum import auto, Enum
import requests
from PIL import Image
from io import BytesIO
import base64
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage

import torch
import torch.utils.checkpoint
from torch import nn, Tensor
from torch.nn import functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import CLIPVisionModel, CLIPImageProcessor,SiglipVisionModel, SiglipImageProcessor
from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import TinyLlavaConfig, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# from tinyllava.utils.data_utils import get_value_from_kwargs
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."
import os
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

# this import has to be relative, otherwise, when setting trust_remote_code=True
# huggingface transformers won't be able to load the module correctly
from numbers import Number
from typing import List, Optional, Union

import numpy as np
from transformers import PretrainedConfig, AutoTokenizer
from my_config import GLOBAL_SETTINGS
from datetime import datetime

logger = logging.get_logger(__name__)

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
LOGDIR = "."


# USER-DEFINED CONSTANTS
ATTN_LOSS_SCALER = GLOBAL_SETTINGS.get("ATTN_LOSS_SCALER")
ATTN_SKIP_TOPK = GLOBAL_SETTINGS.get("ATTN_SKIP_TOPK")
ATTN_COLLECT_FIRST_N_LAYERS = GLOBAL_SETTINGS.get("ATTN_COLLECT_FIRST_N_LAYERS")
ATTN_LOSS_TEMPERATURE = GLOBAL_SETTINGS.get("ATTN_LOSS_TEMPERATURE")

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    TINY_LLAMA = auto()
    QWEN_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)




conv_phi_v0 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    It can be seen at:
    https://github.com/tensorflow/models/blob/2cfc99eff5e5eb729c6793d2f3d03aa1c9be2b15/research/slim/nets/mobilenet/mobilenet.py#L62
    Args:
        v: input value
        divisor: default to 8
        min_value: minimum divisor value
    Returns:
        new_v: new divisible value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def compute_heads(model_dim: int, head_dim: int) -> int:
    """Compute the number of heads.
    Args:
        model_dim: Model dimension.
        head_dim: Head dimension.
    Returns:
        An integer denoting number of heads in multi-head attention is returned.
    Raises:
        ValueError: if model dimension is not divisible by head dimension.
    """
    if model_dim % head_dim == 0:
        return model_dim // head_dim
    else:
        raise ValueError(
            f"Model dimension should be divisible by head dimension. Got: {model_dim} and {head_dim}."
        )


OpenELM_CONFIGS = {
    "OpenELM-270M": dict(
        num_transformer_layers=16,
        model_dim=1280,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multipliers to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-450M": dict(
        num_transformer_layers=20,
        model_dim=1536,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multipliers to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-1_1B": dict(
        num_transformer_layers=28,
        model_dim=2048,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multipliers to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-3B": dict(
        num_transformer_layers=36,
        model_dim=3072,
        head_dim=128,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multipliers to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
}


class OpenELMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OpenELMModel`]. It is used to instantiate an OpenELM model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the OpenELM model.
        max_context_length (`int`, *optional*, defaults to 2048):
            Maximum number of input tokens.
        num_transformer_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        model_dim (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        qkv_multipliers (`Union[Number, List[Number]]`, *optional*, defaults to 1.0):
            If the qkv_multipliers is a Number, then all attention layers have the same latent dimensions,
            resulting in uniform allocation of parameters.
            If the qkv_multipliers is a List of Number, then each attention layer have different latent dimensions
            assuming qkv_multipliers[0] != qkv_multipliers[1]. This results in variable allocation of parameters in attention layer.
            This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
        num_query_heads (`Union[int, None]`, *optional*, defaults to None):
            The number of query heads, computed from `compute_heads(model_dim=model_dim, head_dim=head_dim)`.
        num_gqa_groups (`int`, *optional*, defaults to 1):
            This variable allows to switch between multi-head attention, group query attention, and multi-query attention.
            When num_gqa_groups == 1, then it is multi-head attention.
            When 1 < num_gqa_groups < num_heads and num_heads is divisible by num_gqa_groups, then it is group query attention
            When num_gqa_groups == num_heads, then it is multi-query attention
        ffn_multipliers (`Union[Number, List[Number]]`, *optional*, defaults to 4.0):
            Feed-forward network (FFN) multipliers.
            If the ffn_multipliers is a Number, then all FFN layers have the same latent dimensions,
            resulting in uniform allocation of parameters.
            If the ffn_multipliers is a List of Number, then each FFN layer have different latent dimensions
            assuming ffn_multipliers[0] != ffn_multipliers[1]. This results in variable allocation of parameters in FFN layer.
            This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
        ffn_with_glu (`bool`, *optional*, defaults to True):
            Whether to use FFN with Gated Linear Unit (GLU)
        ffn_dim_divisor (`int`, *optional*, defaults to 256):
            The ffn layer dimension divisor.
        activation_fn_name (`str` or `function`, *optional*, defaults to `"swish"`):
            The non-linear activation function (function or string) in the decoder.
        normalization_layer_name (`str` or `function`, *optional*, defaults to `"rms_norm"`):
            Type of normalization layer.
        normalize_qk_projections (`bool`, *optional*, defaults to False):
            Whether to normalize queries and keys after projections
        share_input_output_layers (`bool`, *optional*, defaults to False):
            Whether to share the embedding between input and output linear layer
        rope_freq_constant (`int`, *optional*, defaults to 10000):
            The base period of the RoPE embeddings.
        rope_max_length (`int`, *optional*, defaults to 4096):
            That rope_max_length is set to twice of max_context_length.
            This allows flexibility in token lengths during training or fine-tuning.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
    """

    model_type = "openelm"

    def __init__(
        self,
        vocab_size: int = 32000,
        max_context_length: int = 2048,
        num_transformer_layers: int = 12,
        model_dim: int = 2048,
        head_dim: int = 128,
        qkv_multipliers: Union[Number, List[Number]] = 1.0,
        num_query_heads: Union[int, None] = None,
        num_gqa_groups: int = 1,
        ffn_multipliers: Union[Number, List[Number]] = 4.0,
        ffn_with_glu: bool = True,
        ffn_dim_divisor: int = 256,
        activation_fn_name: str = "swish",
        normalization_layer_name: str = "rms_norm",
        normalize_qk_projections: bool = False,
        share_input_output_layers: bool = False,
        rope_freq_constant: int = 10000,
        rope_max_length: int = 4096,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.num_transformer_layers = num_transformer_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.qkv_multipliers = qkv_multipliers
        self.num_query_heads = num_query_heads
        self.num_gqa_groups = num_gqa_groups
        self.ffn_multipliers = ffn_multipliers
        self.ffn_with_glu = ffn_with_glu
        self.ffn_dim_divisor = ffn_dim_divisor
        self.activation_fn_name = activation_fn_name
        self.normalization_layer_name = normalization_layer_name
        self.normalize_qk_projections = normalize_qk_projections
        self.share_input_output_layers = share_input_output_layers
        self.rope_freq_constant = rope_freq_constant
        self.rope_max_length = rope_max_length
        self.num_query_heads = (
            compute_heads(model_dim=model_dim, head_dim=head_dim)
            if num_query_heads is None
            else num_query_heads
        )
        self.initializer_range = initializer_range

        self.__post_init__()
        super().__init__(
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def __post_init__(self) -> None:
        if self.num_gqa_groups is not None:
            head_multiple_of = self.num_gqa_groups
        else:
            head_multiple_of = 2

        if isinstance(self.qkv_multipliers, Number):
            # All attention layers have the same latent dimensions, resulting in uniform allocation of parameters.
            qkv_dim = make_divisible(
                self.model_dim * self.qkv_multipliers,
                divisor=self.head_dim * head_multiple_of,
            )
            query_dims = [int(qkv_dim)] * self.num_transformer_layers

        elif (
            isinstance(self.qkv_multipliers, (tuple, list))
            and len(self.qkv_multipliers) == 2
        ):
            # Each attention layer have different latent dimensions assuming qkv_multipliers[0] != qkv_multipliers[1].
            # This results in variable allocation of parameters in attention layer.
            # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
            qkv_multipliers = [
                round(v, 2)
                for v in np.linspace(
                    self.qkv_multipliers[0],
                    self.qkv_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=float,
                )
            ]
            # Make sure that scaled model dimension is divisible by scaled head dimension.
            query_dims = [
                int(
                    make_divisible(
                        self.model_dim * m, divisor=self.head_dim * head_multiple_of
                    )
                )
                for m in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                f"QKV multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        # compute the number of query, key, and value heads
        # For multi-head and multi-query attention, the number of heads for query, key, and value are the same.
        # For group query attention, the number of key and value heads are the same.
        self.num_query_heads = [
            int(compute_heads(q_dim, self.head_dim)) for q_dim in query_dims
        ]
        self.num_kv_heads = [
            q_heads // self.num_gqa_groups for q_heads in self.num_query_heads
        ]

        # Feed-forward network (FFN) multipliers
        if isinstance(self.ffn_multipliers, Number):
            # All FFN layers have the same latent dimensions, resulting in uniform allocation of parameters.
            self.ffn_multipliers = [self.ffn_multipliers] * self.num_transformer_layers
        elif isinstance(self.ffn_multipliers, (tuple, list)):
            # Each FFN layer have different latent dimensions assuming ffn_multipliers[0] != ffn_multipliers[1].
            # This results in variable allocation of parameters in FFN layer.
            # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
            if len(self.ffn_multipliers) == 2:
                self.ffn_multipliers = [
                    round(v, 2)
                    for v in np.linspace(
                        self.ffn_multipliers[0],
                        self.ffn_multipliers[1],
                        num=self.num_transformer_layers,
                        dtype=float,
                    )
                ]
            else:
                assert (
                    len(self.ffn_multipliers) == self.num_transformer_layers
                ), f"{len(self.ffn_multipliers)=}!={self.num_transformer_layers=}"
        else:
            raise NotImplementedError(
                f"FFN multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        # check num_query_heads divisible by num_kv_heads for every layer
        for layer_idx in range(len(query_dims)):
            assert self.num_query_heads[layer_idx] % self.num_kv_heads[layer_idx] == 0

class OpenELMRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        Initialize the OpenELMRMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        """
        Apply the OpenELMRMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the OpenELMRMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying OpenELMRMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return (
            super().extra_repr() + f"num_features={self.num_features}, eps={self.eps}"
        )


class OpenELMPreTrainedModel(PreTrainedModel):
    config_class = OpenELMConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OpenELMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, OpenELMRMSNorm):
            module.weight.data.fill_(1.0)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
    return (x * pos_cos) + (_rotate_half(x) * pos_sin)


class OpenELMRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings (aka RoPE) from `RoFormer <https://arxiv.org/abs/2104.09864>`_.
    RoPE encodes the position information of tokens using a rotation matrix, and is able to capture
    explicit relative positional dependencies.
    Args:
        model_dim: The dimensionality of the model's hidden state.
        max_seq_length: Maximum sequence length.
        freq_constant: A constant used for computing frequencies.
    """

    def __init__(
        self, model_dim: int, max_seq_length: int, freq_constant: int = 10000
    ) -> None:
        inv_freq = 1.0 / (
            freq_constant
            ** (torch.arange(0, model_dim, 2, dtype=torch.float32) / model_dim)
        )
        super().__init__()

        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.max_seq_length = max_seq_length

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def extra_repr(self) -> str:
        return f"\tmodel_dim={self.model_dim}, max_seq_length={self.max_seq_length}, freq_constant={self.freq_constant}"

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_device: torch.device = torch.device("cpu"),
        key_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute sine and cos embeddings.
        Args:
            key_len: Number of tokens in the key embeddings in the transformer model.
            device: Device where the key embeddings are stored.
            key_dtype: Data type of the key embeddings.
        Returns:
            None
        ...note:
            We recalculate the sine and cosine embeddings if any of the following conditions are met:
                1. The number of tokens in key embeddings are greater than the cached sequence length.
                2. Sine and cosine caches are empty.
                3. The device and data type of sine and cosine embeddings does not match with the key embeddings.
        """
        if (
            key_len > self._cached_seq_length
            or self._cached_cos is None
            or (self._cached_cos is not None and self._cached_cos.device != key_device)
            or (self._cached_cos is not None and self._cached_cos.dtype != key_dtype)
            or self._cached_sin is None
            or (self._cached_sin is not None and self._cached_sin.device != key_device)
            or (self._cached_sin is not None and self._cached_sin.dtype != key_dtype)
        ):
            self._cached_seq_length = max(key_len, self._cached_seq_length)

            # The shape of 'pos_index' is [number of key tokens]
            pos_index = torch.arange(
                self._cached_seq_length,
                dtype=torch.float32,
                device=self.inv_freq.device,
            )
            # The shape of 'pos_index_theta' is [number of key tokens, model dimension]
            pos_index_theta = torch.einsum("i,j->ij", pos_index, self.inv_freq)
            # The shape of 'emb' is [number of key tokens, model dimension]
            emb = torch.cat((pos_index_theta, pos_index_theta), dim=-1)

            # the shape of cos and sin embeddings is [number of key tokens, model_dim]
            cos_emb = emb.cos().to(dtype=key_dtype, device=key_device)
            sin_emb = emb.sin().to(dtype=key_dtype, device=key_device)

            # the shape of cached cos and sin embeddings is [1, 1, number of key tokens, model_dim]
            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function of RoPE embeddings.
        Args:
            query: Query embeddings in the transformer model. The shape of query embeddings is
                [Batch, number of query heads, number of query tokens, model dimension].
            key: Key embeddings in the transformer model. The shape of key embeddings is
                [Batch, number of key heads, number of key tokens, model dimension].
        Returns:
            A tuple containing the query and key embeddings with positional information. The shape of the returned query
            and key embeddings is the same as the input query and key embeddings respectively.
        ...note:
            The RoPE embedding computation is done in full-precision. After the computation, input query and key tensors
            are casted to original input datatype.
        """
        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]

        assert dim == self.model_dim
        assert key.device == query.device
        assert key.dtype == query.dtype

        # In the context of self-attention, the lengths of keys and queries are equal.
        # However, in generation tasks, such as predicting the next token in a sequence, the lengths of keys and queries
        # can differ. For instance, when employing key-value (KV) caching for sequence prediction, the keys
        # represent embeddings of previous tokens and the current token, while the query corresponds
        # to the embedding of the current token only.
        assert (
            key_len >= query_len
        ), "Number of keys has to be greater than or equal to number of queries."

        query_float = query.float()
        key_float = key.float()

        self._compute_sin_cos_embeddings(
            key_len, key_device=key_float.device, key_dtype=key_float.dtype
        )
        query_float = _apply_rotary_pos_emb(
            x=query_float,
            pos_sin=self._cached_sin[..., key_len - query_len : key_len, :],
            pos_cos=self._cached_cos[..., key_len - query_len : key_len, :],
        )
        key_float = _apply_rotary_pos_emb(
            x=key_float,
            pos_sin=self._cached_sin[..., :key_len, :],
            pos_cos=self._cached_cos[..., :key_len, :],
        )

        return query_float.type_as(query), key_float.type_as(key)


class OpenELMMultiHeadCausalAttention(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=(q_heads + k_heads + v_heads) * head_dim,
            bias=False,
        )

        self.pos_embedding = OpenELMRotaryEmbedding(
            model_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(
            in_features=q_heads * head_dim,
            out_features=config.model_dim,
            bias=False,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"query_heads={self.num_q_heads}, key_heads={self.num_k_heads}, value_heads={self.num_v_heads}"
        )

    # Efficient implementation equivalent to the following:
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(attn_mask.size(), dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value, attn_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of multi-head self-attention.
        Args:
            hidden_states: Input tensor of the shape [batch size, sequence length, model dimension].
            past_key_value: Tensor storing the cached keys and values.
            output_attentions: output attention weights.
            use_cache: Specifies whether to use kv-cache for generation.
            cache_position: used for updating the kv-cache.
        Returns:
            The output of the same shape as the input, optionally with a tensor containing cached keys and values.
        """

        # scaled_dot_product_attention does not return attention weights, set output_attentions to False
        # output_attentions = False
        batch_size, seq_length, d_model = hidden_states.size()

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            batch_size,
            seq_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        qkv = qkv.transpose(1, 2)
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S h], [B, k_h, S, h], [B, v_h, S, h]
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=1
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            cache_kwargs = {"cache_position": cache_position}
            keys, values = past_key_value.update(
                keys, values, self.layer_idx, cache_kwargs
            )

        # Add positional embedding
        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            # GQA
            # [B, k_h, S, h] --> [B, q_h, S, h]
            keys = keys.repeat_interleave(self.num_groups, dim=1)
            # [B, v_h, S, h] --> [B, q_h, S, h]
            values = values.repeat_interleave(self.num_groups, dim=1)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

        attn_output, attn_weight = self.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=causal_mask,
            dropout_p=0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = attn_weight
        return attn_output, attn_weights, past_key_value


class OpenELMFeedForwardNetwork(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=2 * intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def extra_repr(self) -> str:
        return super().extra_repr() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of FFN layer.
        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].
        Returns:
            A tensor of the same shape as the input.
        """
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.chunk(2, dim=-1)
            y = self.act(y_1) * y_2
            return self.proj_2(y)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class OpenELMDecoderLayer(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = OpenELMMultiHeadCausalAttention(config=config, layer_idx=layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config=config, layer_idx=layer_idx)
        self.ffn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )
        self.attn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OpenELMModel(OpenELMPreTrainedModel):
    config_class = OpenELMConfig

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.config = config

        self.token_embeddings = nn.Embedding(
            embedding_dim=config.model_dim,
            num_embeddings=config.vocab_size,
        )

        self.layers = nn.ModuleList(
            OpenELMDecoderLayer(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_transformer_layers)
        )
        self.norm = OpenELMRMSNorm(num_features=config.model_dim)
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nn.Linear(
                in_features=config.model_dim,
                out_features=config.vocab_size,
                bias=False,
            )
        self.num_transformer_layers = config.num_transformer_layers
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        # NOTE: This is not friendly with TorchScript, ONNX, ExportedProgram serialization for very large `max_context_length`.
        causal_mask = torch.full(
            (config.max_context_length, config.max_context_length),
            fill_value=True,
            dtype=torch.bool,
        )
        self.register_buffer(
            "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
        )

        # Initialize weights and apply final processing
        self.post_init()
        self.reset_parameters(config=config)

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.token_embeddings = new_embeddings

    def reset_parameters(self, config: OpenELMConfig) -> None:
        """Initialize the layers in Language Model
        The initialization scheme is followed, following `OPT <https://arxiv.org/pdf/2205.01068.pdf>`_.
        Args:
            use_megatron_std: Use standard deviation as described in Megatron-LM.
        Returns:
            None
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = module.in_features**-0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                std = module.embedding_dim**-0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, OpenELMRMSNorm):
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        model_dim = config.model_dim
        n_layers = config.num_transformer_layers
        std = (model_dim**-0.5) * ((2 * n_layers) ** -0.5)
        for param_name, param in self.named_parameters():
            if param_name.endswith("out_proj.weight") or param_name.endswith(
                "ffn.proj_2.weight"
            ):
                torch.nn.init.normal_(param, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full(
                (2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]),
                fill_value=1,
            )
            self.register_buffer(
                "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
            )

        # We use the current dtype to avoid any overflows
        min_dtype = torch.finfo(dtype).min
        causal_mask = (
            self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype)
            * min_dtype
        )

        causal_mask = causal_mask.to(dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                :, None, None, :
            ].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        if self.config._attn_implementation == "sdpa" and attention_mask is not None:
            # For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in masked rows from the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = causal_mask.mul(
                    ~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True)
                ).to(dtype)

        return causal_mask


####################################################################################################

def probability_to_logit(prob, eps=1e-7):
    """
    Convert a probability (or a probability map) to its logit.

    Parameters:
        prob (float or torch.Tensor): Probability value(s) in the range (0, 1).
        eps (float): A small epsilon value to avoid division by zero.

    Returns:
        logit (float or torch.Tensor): The logit corresponding to the input probability.
    """
    # Clip probabilities to avoid log(0) issues
    prob = torch.clamp(prob, min=eps, max=1 - eps)
    logit = torch.log(prob / (1 - prob))
    return logit


def soft_label_cross_entropy_loss_for_prob(input, target, temperature=1.0, reduction='mean'):
    """
    Compute the cross-entropy loss between the predicted probabilities and the soft labels.
    """
    epsilon = 1e-7

    assert len(input.shape) == 2, f"Input shape must be (batch_size, num_classes), got {input.shape}"

    input_logit = probability_to_logit(input)
    logprobs = torch.log(nn.functional.softmax(input_logit / temperature, dim=1) + epsilon)
    if reduction == 'mean':
        return - (target * logprobs).sum() / input.size(0) / input.size(1)
    elif reduction == 'sum':
        return - (target * logprobs).sum() / input.size(0) 

def visualize_attn_map_0(attn_maps, path_prefix, image_size=384, image_ratio=1):
    # Detach the tensor from computation graph before operations
    attn_maps = attn_maps.detach()
    
    # Resize attention map to match image dimensions
    attn_resized = F.interpolate(
        attn_maps.unsqueeze(0).unsqueeze(0),  # Add batch + channel dims
        size=image_size,  # Target size (H, W)
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()  # Remove added dims and convert to numpy
    
    fig, ax = plt.subplots(figsize=(1, image_ratio), dpi=150)
    
    # Use OpenCV-based blending
    im = ax.imshow(attn_resized, cmap='hot')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Add colorbar for reference
    ax.axis("off")
    
    # Use a more efficient timestamp approach
    formatted_time = datetime.now().strftime("%H%M%S")
    # Make the directory if it doesn't exist
    os.makedirs("train_temp4", exist_ok=True)
    # Save figure
    fig.savefig(f"train_temp4/test_save_kl_v2_{path_prefix}_{formatted_time}.png")  
    # Close the figure to free memory
    plt.close(fig)
    
def visualize_attn_map(attn_maps1, attn_maps2, path_prefix, image_size=384, image_ratio=1):
    # Detach the tensors from computation graph before operations
    attn_maps1 = attn_maps1.detach()
    attn_maps2 = attn_maps2.detach()
    
    # Resize attention maps to match image dimensions
    attn_resized1 = F.interpolate(
        attn_maps1.unsqueeze(0).unsqueeze(0),  # Add batch + channel dims
        size=image_size,  # Target size (H, W)
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()  # Remove added dims and convert to numpy
    
    attn_resized2 = F.interpolate(
        attn_maps2.unsqueeze(0).unsqueeze(0),  # Add batch + channel dims
        size=image_size,  # Target size (H, W)
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()  # Remove added dims and convert to numpy
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * image_ratio, image_ratio), dpi=150)
    
    # Plot the first attention map
    im1 = ax1.imshow(attn_resized1, cmap='hot')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)  # Add colorbar for reference
    ax1.axis("off")
    ax1.set_title("Model Output")
    
    # Plot the second attention map
    im2 = ax2.imshow(attn_resized2, cmap='hot')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)  # Add colorbar for reference
    ax2.axis("off")
    ax2.set_title("Target")
    
    # Use a more efficient timestamp approach
    formatted_time = datetime.now().strftime("%H%M%S")
    # Make the directory if it doesn't exist
    os.makedirs("train_vis", exist_ok=True)
    # Save figure
    fig.savefig(f"train_vis/test_save_kl_v2_{path_prefix}_{formatted_time}.png")  
    # Close the figure to free memory
    plt.close(fig)
'''
p = target
q = input
'''
def calculate_bce_loss(q, p, threshold=0.7, eps=1e-8):
    p = (p > threshold).float()  
    ce = p * torch.log(q + eps) + (1 - p) * torch.log(1 - q + eps)
    return -ce.sum() 
    
def calculate_l1_loss(q, p, threshold=0.5):
    # p = (p > threshold).float()  
    return torch.abs(p - q).mean()

def weighted_mse_loss_customized(input, target, lamda=0.1):
    weighted_mse = target * (target - input)**2 + lamda*(1-target) * (target - input)**2
    return weighted_mse.sum()

'''
wi = 1/ (alpha - G_i)
W_MSE = sum(wi * (G_i - Ai)**2)
'''
def weighted_mse_loss(input, target, alpha=1.1):
    # note that target is at max 1, so this operation is safe to do
    w = 1 / (alpha - target)
    w_mse = torch.sum(w * (target - input)**2)
    return w_mse

def focal_loss(pred, target, alpha=0.7, gamma=2.0):
     """Computes Focal Loss."""
     bce = F.binary_cross_entropy(pred, target, reduction='none')  # Compute BCE per pixel
     p_t = target * pred + (1 - target) * (1 - pred)  # p_t = prob assigned to true class
     loss = alpha * (1 - p_t) ** gamma * bce  # Apply focal weight
     return loss.mean()

def dice_loss(input, target, epsilon=1e-8):
    intersection = torch.sum(input * target)
    total = torch.sum(input) + torch.sum(target)
    dice_coeff = (2. * intersection + epsilon) / (total + epsilon)
    loss = 1 - dice_coeff

    return loss


class FocalLossForAttention(nn.Module):
    def __init__(self, gamma=2.0, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        input = input.double()
        target = target.double()
        
        # print(f"Input before min/max: {input.min().item()}, {input.max().item()}")
        # print(f"Target before min/max: {target.min().item()}, {target.max().item()}")
        
        # Ensure numerical stability (avoid log(0))
        input = input.clamp(min=1e-8, max=1.0 - 1e-8)  # More aggressive clamping
        target = target.clamp(min=0.0, max=1.0)  # Ensure target is valid
        
        # Check for invalid values in input and target
        if torch.isnan(input).any() or torch.isinf(input).any():
            raise ValueError("Input contains NaN or inf values")
        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError("Target contains NaN or inf values")

        loss = - ((1 - (input ** self.gamma)) * (target * torch.log(input)) 
                + (input ** self.gamma) * (1 - target) * torch.log(1 - input))
        
        # Check for NaN values in loss
        if torch.isnan(loss).any():
            raise ValueError("NaN values detected in loss")
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

####################################################################################################
class OpenELMForCausalLM_wAttn(OpenELMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.transformer = OpenELMModel(config)
        self.vocab_size = config.vocab_size
        if config.share_input_output_layers:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def collect_attention_maps(self, outputs, crossmodal_labels, n_layer_to_collect=None):

        # collecting attention map (detail: avg all heads, all layers, and all tokens)
        attn_maps_all_layers = []
        for layer_idx, layer_attn in enumerate(outputs.attentions[:n_layer_to_collect]):
            # layer_attn shape: (batch_size, num_heads, seq_len, seq_len)
            # the seq_len, seq_len part is representing the softmax(QK^T)
            layer_attn_avg_head = layer_attn.mean(dim=1)
            # layer_attn_avg_head = layer_attn.max(dim=1)[0]
            # get the first and last "1" in crossmodal_labels where 1 is in sequence
            start_idx = (crossmodal_labels[0] == 1).nonzero()[0][0]
            end_idx = (crossmodal_labels[0] == 1).nonzero()[-1][0]

            # fixed attn query -> vision: batch_size, text_token_section_size, image_token_section_size
            layer_attn_avg_head_vis = layer_attn_avg_head[:, end_idx+1:, start_idx:end_idx+1]
            # averaging over all the queries: batch_size, image_token_section_size
            layer_attn_avg_head_vis = layer_attn_avg_head_vis.mean(dim=1)
            grid_size = math.ceil(math.sqrt(layer_attn_avg_head_vis.shape[-1]))
            # append 0 in the front if the number of tokens is not a perfect square
            # this can happen when using different vision token strategies
            if grid_size * grid_size != layer_attn_avg_head_vis.shape[-1]:
                zero_pad = torch.zeros((layer_attn_avg_head_vis.shape[0], grid_size * grid_size - layer_attn_avg_head_vis.shape[-1]), dtype=layer_attn_avg_head_vis.dtype, device=layer_attn_avg_head_vis.device)
                layer_attn_avg_head_vis = torch.cat((zero_pad, layer_attn_avg_head_vis), dim=-1)

            attn_maps_all_layers.append(layer_attn_avg_head_vis)

        # avg all layers: shape batch_size, padded seq len
        attn_maps_all_layers = torch.stack(attn_maps_all_layers, dim=0).mean(dim=0)
        return attn_maps_all_layers, grid_size

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        crossmodal_labels: Optional[torch.LongTensor] = None,
        g_attn_maps: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if self.training:
            print("Training step")
        else:
            print("Evaluation step")

        # Force output_attentions to True for the attention maps loss
        output_attentions = True

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        
        # check all elements in crossmodal_labels are the same
        if crossmodal_labels is not None:
            if not torch.all(crossmodal_labels == crossmodal_labels[0]):
                raise ValueError("All elements in crossmodal_labels should be the same.")

        # collecting logits
        hidden_states = outputs[0]
        if self.lm_head is None:
            # shared
            logits = F.linear(
                hidden_states, weight=self.transformer.token_embeddings.weight
            )
        else:
            logits = self.lm_head(hidden_states)

        
        logits = logits[:, : self.config.vocab_size]

        loss = None
        lan_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lan_loss = loss_fct(shift_logits, shift_labels)
            if self.training:
                print(f"Training language loss: {lan_loss}")
            else:
                print(f"Evaluation language loss: {lan_loss}")
            loss = lan_loss
        # TODO: Check this function
        if g_attn_maps is not None:

            S = ATTN_LOSS_SCALER
            skip_K = ATTN_SKIP_TOPK
            N = ATTN_COLLECT_FIRST_N_LAYERS

            # collecting attention map (detail: avg all heads, all layers, and all tokens)
            attn_maps_all_layers, grid_size = self.collect_attention_maps(outputs, crossmodal_labels, n_layer_to_collect=N)
            g_attn_maps = g_attn_maps / g_attn_maps.max()
            # print(f"ground attention map shape/min/max/mean:{g_attn_maps.shape}, {g_attn_maps.min().item()}, {g_attn_maps.max().item()}, {g_attn_maps.mean().item()}")
            # Needed for the interpolation, used for all the losses except dice + bce
            g_attn_maps = g_attn_maps.unsqueeze(1)
            g_attn_maps = F.interpolate(g_attn_maps, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
            g_attn_maps = g_attn_maps.squeeze(1).view(-1, grid_size * grid_size)
            
            # for dice + bce:
            # gt_res=  g_attn_maps.shape[-1]
            # g_attn_maps = g_attn_maps.view(-1, gt_res * gt_res)
            # g_attn_maps[g_attn_maps>=0.5] = 1
            # g_attn_maps[g_attn_maps<0.5] = 0
            
            # print(f"Ground after processing shape: {g_attn_maps.shape}")
            # Each pixel should be between 0 and 1, which it already is because when created, we divided by the max
            # but the overall attention doesn't have to add up to one
            # g_attn_maps = g_attn_maps / g_attn_maps.sum(1, keepdim=True)

            # apply loss on attention maps # set lambda to 10.
            # breakpoint()
            # loss_attn_maps = F.mse_loss(attn_maps_all_layers, g_attn_maps)
            # loss = 1. * loss_attn_maps * float(grid_size * grid_size) # scale the loss by grid_size ** 2


            _, topk_indices = torch.topk(attn_maps_all_layers, skip_K, dim=-1)
            # remove elements with top K for each batch
            attn_maps_all_layers = attn_maps_all_layers.scatter(dim=-1, index=topk_indices, value=0)
            # print(f"prediction attention map before normalization shape/min/max/mean/sum:\
            #     {attn_maps_all_layers.shape}, {attn_maps_all_layers.min().item()},\
            #     {attn_maps_all_layers.max().item()}, {attn_maps_all_layers.mean().item()},\
            #     {attn_maps_all_layers.sum().item()}")

            
            # for w_mse and focal loss it makes sense to use max for kl divergence it doesn't, bce + dice does it seperately below
            attn_maps_all_layers = attn_maps_all_layers / attn_maps_all_layers.max(1, keepdim=True)[0]

            # for dice + bce:
            # batch_size, seq_len_square = attn_maps_all_layers.shape
            # attn_maps_all_layers = attn_maps_all_layers.view(batch_size, grid_size, grid_size)
            # attn_maps_all_layers = F.interpolate(attn_maps_all_layers.unsqueeze(1), size=(gt_res, gt_res), mode='bilinear', align_corners=True)
            # attn_maps_all_layers = attn_maps_all_layers.view(batch_size, gt_res * gt_res)
            # attn_maps_all_layers = attn_maps_all_layers / attn_maps_all_layers.max()

            # visualize_attn_map(attn_maps_all_layers.view(grid_size, grid_size), "model_output")
            # visualize_attn_map(g_attn_maps.view(grid_size, grid_size), "target")
            # visualize_attn_map(attn_maps_all_layers.view(grid_size, grid_size), g_attn_maps.view(grid_size, grid_size), "combined_output")
            
            # print(f"prediction attention map after normalization shape/min/max/mean:{attn_maps_all_layers.shape}, 
            # {attn_maps_all_layers.min().item()}, {attn_maps_all_layers.max().item()}, 
            # {attn_maps_all_layers.mean().item()}")
            
            # focal loss:
            # gamma = 2.0  
            # focal_loss = FocalLossForAttention(gamma=gamma, reduction="sum")
            # # Flatten the attention maps for compatibility with the focal loss input
            # attn_maps_all_layers = attn_maps_all_layers.view(-1)
            # g_attn_maps = g_attn_maps.view(-1)
            # loss_attn_maps = focal_loss(attn_maps_all_layers, g_attn_maps)
            
    
            # cross entropy
            # IMPORTANT: Mean here is mean over all visual tokens in the batch
            # loss_attn_maps = soft_label_cross_entropy_loss_for_prob(attn_maps_all_layers, 
            #                                                         g_attn_maps, temperature=ATTN_LOSS_TEMPERATURE,
            #                                                         reduction='mean')

            
            # KL divergence
            # kl_loss = nn.KLDivLoss(reduction="batchmean")
            # attn_maps_all_layers_soft_dist = torch.nn.functional.log_softmax(attn_maps_all_layers, dim=-1)
            # g_attn_maps_softmaxed = torch.nn.functional.softmax(g_attn_maps, dim=-1)
            # loss_attn_maps = kl_loss(attn_maps_all_layers_soft_dist, g_attn_maps_softmaxed)

            # simple MSE loss code:     
            # loss_attn_maps = F.mse_loss(attn_maps_all_layers, g_attn_maps, reduction='sum')
            # loss += S * loss_attn_maps / len(attn_maps_all_layers)
            
            # l1 loss
            # attn_maps_all_layers = attn_maps_all_layers.view(-1)
            # g_attn_maps = g_attn_maps.view(-1)
            # loss_attn_maps = calculate_l1_loss(attn_maps_all_layers, g_attn_maps)
            
            # weighted MSE loss
            loss_attn_maps = weighted_mse_loss(attn_maps_all_layers, g_attn_maps)
            
            # dice + bce loss
            # d_loss = dice_loss(attn_maps_all_layers, g_attn_maps)
            # bce_loss =  F.binary_cross_entropy(attn_maps_all_layers, g_attn_maps, reduction='mean')
            # loss_attn_maps = 100 * d_loss +  1.0 * bce_loss
            # print(f"Dice Loss {d_loss}, BCE Loss {bce_loss}")

            if self.training:
                print(f"Training Attention Loss {loss_attn_maps}")
            else:
                print(f"Evaluation Attention Loss {loss_attn_maps}")
            loss += S * loss_attn_maps
            print(f"total loss: {loss}")


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if self.generation_config.cache_implementation == "static":
            # generation with static cache
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = torch.arange(
            past_length,
            past_length + position_ids.shape[-1],
            device=position_ids.device,
        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # We could use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids.contiguous(),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}

class Connector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config.connector_type)
        act_type = config.connector_type.split('_')[-1]
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.vision_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(ACT_TYPE[act_type]())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            
        self._connector = nn.Sequential(*modules)
    
    def forward(self, x):
        return self._connector(x)

class VisionTower(nn.Module):
    def __init__(self, cfg, model_name_or_path = 'clip'):
        super().__init__()
        if 'clip' in model_name_or_path:
            self._vision_tower = CLIPVisionModel(cfg)
            self._image_processor = CLIPImageProcessor.from_pretrained(cfg.model_name_or_path)
        else:
            self._vision_tower = SiglipVisionModel(cfg)
            self._image_processor = SiglipImageProcessor.from_pretrained(cfg.model_name_or_path)
            
        self.config = cfg
        


    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]

        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        return image_features
        

    
    @property
    def vision_tower(self):
        return self._vision_tower
        
    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None
    


class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        
        super().__init__(config)

        self.language_model = OpenELMForCausalLM_wAttn(config.text_config)
        self.vision_tower = VisionTower(config.vision_config, config.vision_model_name_or_path)
        self.connector = Connector(config)
        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        g_attn_maps: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                crossmodal_labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        output_dict = self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            g_attn_maps=g_attn_maps,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            crossmodal_labels=crossmodal_labels
        )

        ############################################################
        # visualize the images and the attention maps
        # attn_maps_all_layers, grid_size = self.language_model.collect_attention_maps(output_dict, crossmodal_labels)
        # visualization_dir = 'OUTPUTS/lora-finetune-TinyLLaVA-OpenELM-450M-SigLIP-0.89B_wAttn_s50' #'OUTPUTS/TinyLLaVA-OpenELM-450M-SigLIP-0.89B'

        # # attn_maps_all_layers = attn_maps_all_layers.reshape(-1, grid_size, grid_size)

        # # reset top K high values for each image to 0
        # k_list = [0, 5, 10, 20]
        # for k in k_list:
        #     attn_maps_all_layers_skip_topk = attn_maps_all_layers.clone()
        #     if k > 0:
        #         topk_values, _ = torch.topk(attn_maps_all_layers.view(attn_maps_all_layers.shape[0], -1), k, dim=-1)
        #         topk_values = topk_values[:, -1].unsqueeze(-1)
        #         attn_maps_all_layers_skip_topk[attn_maps_all_layers_skip_topk >= topk_values] = 0

        #     attn_maps_all_layers_skip_topk = attn_maps_all_layers_skip_topk.reshape(-1, grid_size, grid_size)

        #     # interpolate to image size
        #     attn_maps_all_images = F.interpolate(attn_maps_all_layers_skip_topk.unsqueeze(1), size=images.shape[2:], mode='bilinear', align_corners=False)

        #     # visualizing the attention map and overlap with the image
        #     # images: (batch_size, num_channels, height, width)
        #     # attn_maps_all_layers: (batch_size, grid_size, grid_size)

        #     to_pil = ToPILImage()
        #     for i in range(images.shape[0]):
        #         image = to_pil(images[i])
        #         plt.figure()
        #         plt.imshow(np.array(image))
        #         plt.imshow(attn_maps_all_images[i][0].detach().cpu().numpy(), alpha=0.5)
        #         plt.savefig(os.path.join(visualization_dir, f'attn_map_{i}_k{k}.png'))
        # exit()
        ############################################################

        return output_dict
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features
    
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        
        image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_crossmodal_labels = [] # 1 for image, 0 for the rest, 2 for outputs
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_crossmodal_labels.append(torch.zeros(cur_input_embeds.shape[0], dtype=torch.long, device=cur_input_embeds.device))
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # new_crossmodal_label will be 0 * (image_token_indices[1]) + 1 * num_images * image_features.shape[1] + 0 * (image_token_indices[2] - 1 - image_token_indices[1])
            assert num_images == 1, "Only support one image per input for now"
            new_crossmodal_label = torch.cat([
                torch.zeros(image_token_indices[1], dtype=torch.long, device=cur_input_ids.device),
                torch.ones(num_images * image_features.shape[1], dtype=torch.long, device=cur_input_ids.device),
                torch.zeros(image_token_indices[2] - 1 - image_token_indices[1], dtype=torch.long, device=cur_input_ids.device)
            ])
            new_crossmodal_labels.append(new_crossmodal_label)
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_crossmodal_labels = [x[:tokenizer_model_max_length] for x in new_crossmodal_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_crossmodal_labels_padded = torch.zeros((batch_size, max_len), dtype=new_crossmodal_labels[0].dtype, device=new_crossmodal_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_crossmodal_labels) in enumerate(zip(new_input_embeds, new_labels, new_crossmodal_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_crossmodal_labels_padded[i, -cur_len:] = cur_new_crossmodal_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_crossmodal_labels_padded[i, :cur_len] = cur_new_crossmodal_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        new_crossmodal_labels = new_crossmodal_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_crossmodal_labels
        
    def chat(
        self,
        prompt: str,
        tokenizer = None,
        image: str = None,
        max_new_tokens: int = 512,
        num_beams = 1,
        top_p=None,
        temperature=0
    ):
        image_processor = self.vision_tower._image_processor

        if image is not None:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt 
        conv = conv_phi_v0.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if image is not None:
            image = load_image(image)
            image_tensor = process_images(image, image_processor, self.config).to(self.device)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0).to(self.device)
        )
        # Generate
        stime = time.time()

        with torch.inference_mode():
            output_ids = self.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                # stopping_criteria=[stopping_criteria],
            )

        # print('inference over')
        generation_time = time.time() - stime
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        outputs = outputs.strip()

        return outputs, generation_time

    

            

AutoConfig.register("tinyllava", TinyLlavaConfig)        
AutoModelForCausalLM.register(TinyLlavaConfig, TinyLlavaForConditionalGeneration)        
