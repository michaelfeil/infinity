# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings

from .decoder_models import (
    BarkAttentionLayerNestedTransformer,
    BartAttentionLayerNestedTransformer,
    BlenderbotAttentionLayerNestedTransformer,
    BloomAttentionLayerNestedTransformer,
    CodegenAttentionLayerNestedTransformer,
    GPT2AttentionLayerNestedTransformer,
    GPTJAttentionLayerNestedTransformer,
    GPTNeoAttentionLayerNestedTransformer,
    GPTNeoXAttentionLayerNestedTransformer,
    M2M100AttentionLayerNestedTransformer,
    MarianAttentionLayerNestedTransformer,
    OPTAttentionLayerNestedTransformer,
    PegasusAttentionLayerNestedTransformer,
    T5AttentionLayerNestedTransformer,
)
from .encoder_models import (
    AlbertLayerNestedTransformer,
    BartEncoderLayerNestedTransformer,
    BertLayerNestedTransformer,
    CLIPLayerNestedTransformer,
    DistilBertLayerNestedTransformer,
    FSMTEncoderLayerNestedTransformer,
    MBartEncoderLayerNestedTransformer,
    ProphetNetEncoderLayerNestedTransformer,
    ViltLayerNestedTransformer,
    ViTLayerNestedTransformer,
    Wav2Vec2EncoderLayerNestedTransformer,
)


class NestedTransformerManager:
    MODEL_MAPPING = {
        "albert": {"AlbertLayer": AlbertLayerNestedTransformer},
        "bark": {"BarkSelfAttention": BarkAttentionLayerNestedTransformer},
        "bart": {
            "BartEncoderLayer": BartEncoderLayerNestedTransformer,
            "BartAttention": BartAttentionLayerNestedTransformer,
        },
        "bert": {"BertLayer": BertLayerNestedTransformer},
        "bert-generation": {"BertGenerationLayer": BertLayerNestedTransformer},
        "blenderbot": {
            "BlenderbotAttention": BlenderbotAttentionLayerNestedTransformer
        },
        "bloom": {"BloomAttention": BloomAttentionLayerNestedTransformer},
        "camembert": {"CamembertLayer": BertLayerNestedTransformer},
        "blip-2": {"T5Attention": T5AttentionLayerNestedTransformer},
        "clip": {"CLIPEncoderLayer": CLIPLayerNestedTransformer},
        "codegen": {"CodeGenAttention": CodegenAttentionLayerNestedTransformer},
        "data2vec-text": {"Data2VecTextLayer": BertLayerNestedTransformer},
        "deit": {"DeiTLayer": ViTLayerNestedTransformer},
        "distilbert": {"TransformerBlock": DistilBertLayerNestedTransformer},
        "electra": {"ElectraLayer": BertLayerNestedTransformer},
        "ernie": {"ErnieLayer": BertLayerNestedTransformer},
        "fsmt": {"EncoderLayer": FSMTEncoderLayerNestedTransformer},
        "gpt2": {"GPT2Attention": GPT2AttentionLayerNestedTransformer},
        "gptj": {"GPTJAttention": GPTJAttentionLayerNestedTransformer},
        "gpt_neo": {"GPTNeoSelfAttention": GPTNeoAttentionLayerNestedTransformer},
        "gpt_neox": {"GPTNeoXAttention": GPTNeoXAttentionLayerNestedTransformer},
        "hubert": {"HubertEncoderLayer": Wav2Vec2EncoderLayerNestedTransformer},
        "layoutlm": {"LayoutLMLayer": BertLayerNestedTransformer},
        "m2m_100": {
            "M2M100EncoderLayer": MBartEncoderLayerNestedTransformer,
            "M2M100Attention": M2M100AttentionLayerNestedTransformer,
        },
        "marian": {
            "MarianEncoderLayer": BartEncoderLayerNestedTransformer,
            "MarianAttention": MarianAttentionLayerNestedTransformer,
        },
        "markuplm": {"MarkupLMLayer": BertLayerNestedTransformer},
        "mbart": {"MBartEncoderLayer": MBartEncoderLayerNestedTransformer},
        "opt": {"OPTAttention": OPTAttentionLayerNestedTransformer},
        "pegasus": {"PegasusAttention": PegasusAttentionLayerNestedTransformer},
        "rembert": {"RemBertLayer": BertLayerNestedTransformer},
        "prophetnet": {
            "ProphetNetEncoderLayer": ProphetNetEncoderLayerNestedTransformer
        },
        "roberta": {"RobertaLayer": BertLayerNestedTransformer},
        "roc_bert": {"RoCBertLayer": BertLayerNestedTransformer},
        "roformer": {"RoFormerLayer": BertLayerNestedTransformer},
        "splinter": {"SplinterLayer": BertLayerNestedTransformer},
        "tapas": {"TapasLayer": BertLayerNestedTransformer},
        "t5": {"T5Attention": T5AttentionLayerNestedTransformer},
        "vilt": {"ViltLayer": ViltLayerNestedTransformer},
        "vit": {"ViTLayer": ViTLayerNestedTransformer},
        "vit_mae": {"ViTMAELayer": ViTLayerNestedTransformer},
        "vit_msn": {"ViTMSNLayer": ViTLayerNestedTransformer},
        "wav2vec2": {
            "Wav2Vec2EncoderLayer": Wav2Vec2EncoderLayerNestedTransformer,
            "Wav2Vec2EncoderLayerStableLayerNorm": Wav2Vec2EncoderLayerNestedTransformer,
        },
        "xlm-roberta": {"XLMRobertaLayer": BertLayerNestedTransformer},
        "yolos": {"YolosLayer": ViTLayerNestedTransformer},
    }

    OVERWRITE_METHODS = {
        # "llama": {"LlamaModel": ("_prepare_decoder_attention_mask", _llama_prepare_decoder_attention_mask)}
    }

    EXCLUDE_FROM_TRANSFORM = {
        # clip's text model uses causal attention, that is most likely not supported in NestedTransformer
        "clip": ["text_model"],
        # blip-2's Q-former and vision model should not be identified as the last layers of the model
        "blip-2": ["qformer.encoder.layer", "vision_model.encoder.layers"],
        # bark.codec_model.encoder is not supported in NestedTransformer
        "bark": ["codec_model.encoder.layers"],
    }

    CAN_NOT_BE_SUPPORTED = {
        "deberta-v2": "DeBERTa v2 does not use a regular attention mechanism, which is not supported in PyTorch's NestedTransformer.",
        "glpn": "GLPN has a convolutional layer present in the FFN network, which is not supported in PyTorch's NestedTransformer.",
    }

    NOT_REQUIRES_NESTED_TENSOR = {
        "bark",
        "blenderbot",
        "bloom",
        "codegen",
        "gpt2",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "opt",
        "pegasus",
        "t5",
    }

    NOT_REQUIRES_STRICT_VALIDATION = {
        "blenderbot",
        "blip-2",
        "bloom",
        "codegen",
        "gpt2",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "opt",
        "pegasus",
        "t5",
    }

    @staticmethod
    def cannot_support(model_type: str) -> bool:
        """
        Returns True if a given model type can not be supported by PyTorch's Better Transformer.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type in NestedTransformerManager.CAN_NOT_BE_SUPPORTED

    @staticmethod
    def supports(model_type: str) -> bool:
        """
        Returns True if a given model type is supported by PyTorch's Better Transformer, and integrated in Optimum.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type in NestedTransformerManager.MODEL_MAPPING

    @staticmethod
    def requires_nested_tensor(model_type: str) -> bool:
        """
        Returns True if the NestedTransformer implementation for a given architecture uses nested tensors, False otherwise.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type not in NestedTransformerManager.NOT_REQUIRES_NESTED_TENSOR

    @staticmethod
    def requires_strict_validation(model_type: str) -> bool:
        """
        Returns True if the architecture requires to make sure all conditions of `validate_NestedTransformer` are met.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type not in NestedTransformerManager.NOT_REQUIRES_STRICT_VALIDATION


class warn_uncompatible_save(object):
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, ex_typ, ex_val, traceback):
        return True

    def __call__(self, *args, **kwargs):
        warnings.warn(
            "You are calling `save_pretrained` to a `NestedTransformer` converted model you may likely encounter unexpected behaviors. ",
            UserWarning,
        )
        return self.callback(*args, **kwargs)
