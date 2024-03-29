# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)


@dataclass
class Wav2Vec2AsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )
    conv_feature_layers: Optional[str] = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": (
                "string describing convolutional feature extraction "
                "layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
            ),
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    viewmaker_pretrain_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )

    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None
    offload_activations: bool = field(
        default=False, metadata={"help": "offload_activations"}
    )
    min_params_to_wrap: int = field(
        default=int(1e8),
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )
    ddp_backend: str = II("distributed_training.ddp_backend")

    branch_ctc_v1: bool = field(
        default=False,
        metadata={"help": "make ctc layer after each encoder layer, train iteratively"},
    )
    
    branch_ctc_v2: bool = field(
        default=False,
        metadata={"help": "make ctc layer after each encoder layer, loss average train"},
    )

    branch_ctc_v3: bool = field(
        default=False,
        metadata={"help": "make ctc layer after each encoder layer, PCGrad train"},
    )

    ctc_num: int = field(
        default=12,
        metadata={"help": "number of ctc layer after each encoder layer"},
    )

    init_transformer: bool = field(
        default=False,
        metadata={"help": "newly initilize transofmer weight"},
    )
    
    del_spk_info: bool = field(
        default=False,
        metadata={"help": "add spk informatin"},
    )
    
    viewmaker: bool = field(
        default=False,
        metadata={"help": "viewmaker"},
    )
    init_viewmaker: bool = field(
        default=False,
        metadata={"help": "initialize new viewmaker"},
    )
    viewmaker_num: int = field(
        default=1,
        metadata={"help": "number of viewmaker"},
    )
    ch_prune_idx: int = field(
        default=-1,
        metadata={"help": "number of viewmaker"},
    )
    wavlm: bool = field(
        default=False,
        metadata={"help": "use wavlm"},
    )
    MTL: bool = field(
        default=False,
        metadata={"help": "use multiple ctc loss"},
    )
    
    fgsm: bool = field(
        default=False,
    )



@dataclass
class Wav2Vec2CtcConfig(Wav2Vec2AsrConfig):
    blank_weight: float = 0
    blank_mode: str = "add"


@register_model("wav2vec_ctc", dataclass=Wav2Vec2CtcConfig)
class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        """Build a new model instance."""
        #w2v_encoder = Wav2VecEncoderBranchCtcV1(cfg, len(task.target_dictionary)) if cfg.branch_ctc_v1 \
        #                    else Wav2VecEncoder(cfg, len(task.target_dictionary))
        if cfg.branch_ctc_v1:
            w2v_encoder = Wav2VecEncoderBranchCtcV1(cfg, len(task.target_dictionary))
        elif cfg.branch_ctc_v2 or cfg.branch_ctc_v3:
            w2v_encoder = Wav2VecEncoderBranchCtcV2(cfg, len(task.target_dictionary))
        elif cfg.del_spk_info:
            w2v_encoder = Wav2VecEncoderSpkClf(cfg, len(task.target_dictionary), 251)
        elif cfg.viewmaker:
            w2v_encoder = Wav2VecEncoderViewMaker(cfg, len(task.target_dictionary))
        elif cfg.wavlm:
            w2v_encoder = WavLMEncoder(cfg, len(task.target_dictionary))
        elif cfg.MTL:
            w2v_encoder = Wav2VecEncoderMTL(cfg, len(task.target_dictionary))
        else:
            w2v_encoder = Wav2VecEncoder(cfg, 32)
            #w2v_encoder = Wav2VecEncoder(cfg, len(task.target_dictionary))

        return cls(cfg, w2v_encoder)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, tgt_layer=False, **kwargs):
        x = self.w2v_encoder(**kwargs) if not tgt_layer else self.w2v_encoder(tgt_layer=tgt_layer, **kwargs)
        return x


@dataclass
class Wav2Vec2Seq2SeqConfig(Wav2Vec2AsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    autoregressive: bool = II("task.autoregressive")


@register_model("wav2vec_seq2seq", dataclass=Wav2Vec2Seq2SeqConfig)
class Wav2Vec2Seq2SeqModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        return Wav2Vec2Seq2SeqModel(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2AsrConfig):
        return Wav2VecEncoder(cfg)

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqConfig, tgt_dict, embed_tokens):
        return TransformerDecoder(cfg, tgt_dict, embed_tokens)

    def forward(self, **kwargs):
        encoder_out = self.encoder(**kwargs)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
        }
        
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations
               
        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.viewmaker_pretrain_updates = cfg.viewmaker_pretrain_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim

        if targ_d is not None:
            self.proj = Linear(d, targ_d)

    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=False)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=True)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            if cfg.init_transformer:
                keys = list(state["model"].keys())
                for key in keys:
                    if 'encoder.layer' in key:
                        del state["model"][key]
                model.load_state_dict(state["model"], strict=False)
            else:
                model.load_state_dict(state["model"], strict=False)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):
        cnn_fgsm = kwargs['cnn_fgsm'] if 'cnn_fgsm' in kwargs else None
        cnn_feat = kwargs['cnn_feat'] if 'cnn_feat' in kwargs else None
        
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
            "cnn_fgsm": cnn_fgsm if cnn_fgsm is not None else None,
            "cnn_feat": cnn_feat if cnn_feat is not None else None,
        } 

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)
            
            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            
        x = self.final_dropout(x)
        
        if self.proj:
            x = self.proj(x)
        
        return {
            "encoder_out": x,  # T x B x C
            #"conv_feat": res["conv_feat"] if cnn_fgsm is not None else None,
            "conv_feat": res["conv_feat"],
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
            "pac_output": res["pac_output"],
        }

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class Wav2VecEncoderMTL(Wav2VecEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            "branch_ctc_v1": cfg.branch_ctc_v1,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        
        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)
        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj1 = None
        self.proj2 = None
        self.proj3 = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim
        
        if targ_d is not None:
            self.proj1 = Linear(d, targ_d)
            self.proj2 = Linear(d, targ_d)
            self.proj3 = Linear(d, targ_d)
        
        self.proj = [self.proj1, self.proj2, self.proj3]

        self.blank_mode= "add"
        self.blank_weight = 0.

    def forward(self, source, padding_mask, **kwargs):
        cnn_fgsm = kwargs['cnn_fgsm'] if 'cnn_fgsm' in kwargs else None
        cnn_feat = kwargs['cnn_feat'] if 'cnn_feat' in kwargs else None
        
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
            "cnn_fgsm": cnn_fgsm if cnn_fgsm is not None else None,
            "cnn_feat": cnn_feat if cnn_feat is not None else None,
        } 

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)
            
            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            
        x = self.final_dropout(x)
        
        x_list = []
        if self.proj:
            for proj in self.proj:
                x_list.append(proj(x))
        
        return {
            "encoder_out": x,  # T x B x C
            #"conv_feat": res["conv_feat"] if cnn_fgsm is not None else None,
            "conv_feat": res["conv_feat"],
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
            "encoder_out_list": x_list,
        }
    
    def get_logits(self, net_output, logits_list, normalize=False):
        output = []
        for logits in logits_list:
            if self.blank_weight != 0:
                if self.blank_mode == "add":
                    logits[..., 0] += self.blank_weight
                elif self.blank_mode == "set":
                    logits[..., 0] = self.blank_weight
                else:
                    raise Exception(f"invalid blank mode {self.blank_mode}")

            if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
                number_of_classes = logits.size(-1)
                masking_tensor = torch.ones(
                    number_of_classes, device=logits.device
                ) * float("-inf")
                masking_tensor[0] = 0
                logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

            if normalize:
                logits = utils.log_softmax(logits.float(), dim=-1)

            output.append(logits)

        return output

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits_list = self.get_logits(net_output, net_output["encoder_out_list"])

        if log_probs:
            return [utils.log_softmax(logits.float(), dim=-1) for logits in logits_list]
        else:
            return [utils.softmax(logits.float(), dim=-1) for logits in logits_list]


class WavLMEncoder(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            "layer_type": "transformerpos",
        }
        
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations
               
        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.viewmaker_pretrain_updates = cfg.viewmaker_pretrain_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim

        if targ_d is not None:
            self.proj = Linear(d, targ_d)

    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            if cfg.init_transformer:
                keys = list(state["model"].keys())
                for key in keys:
                    if 'encoder.layer' in key:
                        del state["model"][key]
                model.load_state_dict(state["model"], strict=False)
            else:
                model.load_state_dict(state["model"], strict=False)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):
        cnn_fgsm = kwargs['cnn_fgsm'] if 'cnn_fgsm' in kwargs else None
        cnn_feat = kwargs['cnn_feat'] if 'cnn_feat' in kwargs else None
        
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
            "cnn_fgsm": cnn_fgsm if cnn_fgsm is not None else None,
            "cnn_feat": cnn_feat if cnn_feat is not None else None,
        } 

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)
            
            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            
        x = self.final_dropout(x)
        
        if self.proj:
            x = self.proj(x)
        
        return {
            "encoder_out": x,  # T x B x C
            #"conv_feat": res["conv_feat"] if cnn_fgsm is not None else None,
            "conv_feat": res["conv_feat"],
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
        }

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class Wav2VecEncoderBranchCtcV1(Wav2VecEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            "branch_ctc_v1": cfg.branch_ctc_v1,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        
        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)
        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj1 = None
        self.proj2 = None
        self.proj3 = None
        self.proj4 = None
        self.proj5 = None
        self.proj6 = None
        self.proj7 = None
        self.proj8 = None
        self.proj9 = None
        self.proj10 = None
        self.proj11 = None
        self.proj12 = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim
        
        if targ_d is not None:
            self.proj1 = Linear(d, targ_d)
            self.proj2 = Linear(d, targ_d)
            self.proj3 = Linear(d, targ_d)
            self.proj4 = Linear(d, targ_d)
            self.proj5 = Linear(d, targ_d)
            self.proj6 = Linear(d, targ_d)
            self.proj7 = Linear(d, targ_d)
            self.proj8 = Linear(d, targ_d)
            self.proj9 = Linear(d, targ_d)
            self.proj10 = Linear(d, targ_d)
            self.proj11 = Linear(d, targ_d)
            self.proj12 = Linear(d, targ_d)
        
        self.proj = [self.proj1, self.proj2, self.proj3, self.proj4, self.proj5, self.proj6, self.proj7, self.proj8, self.proj9, self.proj10, self.proj11, self.proj12]

    def forward(self, tgt_layer, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args, tgt_layer=tgt_layer, branch_ctc=True)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj[tgt_layer-1](x)

        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
        }


class Wav2VecEncoderBranchCtcV2(Wav2VecEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None):
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            "branch_ctc_v2": cfg.branch_ctc_v2,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        
        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)
        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj1 = None
        self.proj2 = None
        self.proj3 = None
        self.proj4 = None
        self.proj5 = None
        self.proj6 = None
        self.proj7 = None
        self.proj8 = None
        self.proj9 = None
        self.proj10 = None
        self.proj11 = None
        self.proj12 = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim
        
        if targ_d is not None:
            self.proj1 = Linear(d, targ_d)
            self.proj2 = Linear(d, targ_d)
            self.proj3 = Linear(d, targ_d)
            self.proj4 = Linear(d, targ_d)
            self.proj5 = Linear(d, targ_d)
            self.proj6 = Linear(d, targ_d)
            self.proj7 = Linear(d, targ_d)
            self.proj8 = Linear(d, targ_d)
            self.proj9 = Linear(d, targ_d)
            self.proj10 = Linear(d, targ_d)
            self.proj11 = Linear(d, targ_d)
            self.proj12 = Linear(d, targ_d)
        
        self.proj = [self.proj1, self.proj2, self.proj3, self.proj4, self.proj5, self.proj6, self.proj7, self.proj8, self.proj9, self.proj10, self.proj11, self.proj12]

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)
            
            x = res["x"]

            if type(res["layer_results"][6]) != int:
                x7 = res["layer_results"][6][0]
            else:
                x7 = 0

            if type(res["layer_results"][7]) != int:
                x8 = res["layer_results"][7][0]
            else:
                x8 = 0

            if type(res["layer_results"][8]) != int:
                x9 = res["layer_results"][8][0]
            else:
                x9 = 0

            if type(res["layer_results"][9]) != int:
                x10 = res["layer_results"][9][0]
            else:
                x10 = 0

            if type(res["layer_results"][10]) != int:
                x11 = res["layer_results"][10][0]
            else:
                x11 = 0

            if type(res["layer_results"][11]) != int:
                x12 = res["layer_results"][11][0] 
            else:
                x12 = 0
            
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)
        
        result = []
        if self.proj:
            if type(x7) != int:
                x7 = self.final_dropout(x7)
                x7 = self.proj[6](x7)
                result.append(x7)
            if type(x8) != int:
                x8 = self.final_dropout(x8)
                x8 = self.proj[7](x8)
                result.append(x8)
            if type(x9) != int:
                x9 = self.final_dropout(x9)
                x9 = self.proj[8](x9)
                result.append(x9)
            if type(x10) != int:
                x10 = self.final_dropout(x10)
                x10 = self.proj[9](x10)
                result.append(x10)
            if type(x11) != int:
                x11 = self.final_dropout(x11)
                x11 = self.proj[10](x11)
                result.append(x11)
            if type(x12) != int:
                x12 = self.final_dropout(x12)
                x12 = self.proj[11](x12)
                result.append(x12)
        
        return {
            "encoder_out": result,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
            "dropped_layer": res["dropped_layer"],
        }
    
    def get_logits(self, net_output, normalize=False):
        logits_ = net_output["encoder_out"]

        logits_list = []

        for logits in logits_:
            if self.blank_weight != 0:
                if self.blank_mode == "add":
                    logits[..., 0] += self.blank_weight
                elif self.blank_mode == "set":
                    logits[..., 0] = self.blank_weight
                else:
                    raise Exception(f"invalid blank mode {self.blank_mode}")

            if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
                number_of_classes = logits.size(-1)
                masking_tensor = torch.ones(
                    number_of_classes, device=logits.device
                ) * float("-inf")
                masking_tensor[0] = 0
                logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

            if normalize:
                logits = utils.log_softmax(logits.float(), dim=-1)

            logits_list.append(logits)

        return logits_list

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return [utils.log_softmax(logit.float(), dim=-1) for logit in logits]
        else:
            return [utils.softmax(logit.float(), dim=-1) for logit in logits]


class Wav2VecEncoderSpkClf(Wav2VecEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None, spk_num=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            "branch_ctc_v1": cfg.branch_ctc_v1,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        
        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)
        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None
        self.proj_ctc = None
        self.proj_spk = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim
        
        if targ_d is not None:
            self.proj_ctc = Linear(d, targ_d)
            self.proj_spk = Linear(d, spk_num)
            self.proj = [self.proj_ctc, self.proj_spk]
            self.softmax = nn.Softmax(dim=1)

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)
        spk_prob = None

        if self.proj:
            x = self.proj_ctc(x)
            #print(res["dropped_layer"]) 
            #if len(res["layer_results"]) == 12:
            count = 0
            for drop in res["dropped_layer"]:
                if drop < 6:
                    count += 1
            
            if not (5 in res["dropped_layer"]):
                #in_layer_results = res["layer_results"][2][0].mean(0)
                mid1_layer_results = res["layer_results"][5-count][0].mean(0)
                #mid2_layer_results = res["layer_results"][8][0].mean(0)
                #out_layer_results = res["layer_results"][11][0].mean(0)

                spk_logits = self.proj_spk(mid1_layer_results)
                spk_prob = self.softmax(spk_logits)
        
        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
            "spk_prob": spk_prob,
        }


class Wav2VecEncoderSpkClf(Wav2VecEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None, spk_num=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            "branch_ctc_v1": cfg.branch_ctc_v1,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        
        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)
        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None
        self.proj_ctc = None
        self.proj_spk = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim
        
        if targ_d is not None:
            self.proj_ctc = Linear(d, targ_d)
            self.proj_spk = Linear(d, spk_num)
            self.proj = [self.proj_ctc, self.proj_spk]
            self.softmax = nn.Softmax(dim=1)

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)
        spk_prob = None

        if self.proj:
            x = self.proj_ctc(x)
            #print(res["dropped_layer"]) 
            #if len(res["layer_results"]) == 12:
            count = 0
            for drop in res["dropped_layer"]:
                if drop < 6:
                    count += 1
            
            if not (5 in res["dropped_layer"]):
                #in_layer_results = res["layer_results"][2][0].mean(0)
                mid1_layer_results = res["layer_results"][5-count][0].mean(0)
                #mid2_layer_results = res["layer_results"][8][0].mean(0)
                #out_layer_results = res["layer_results"][11][0].mean(0)

                spk_logits = self.proj_spk(mid1_layer_results)
                spk_prob = self.softmax(spk_logits)
        
        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
            "spk_prob": spk_prob,
        }


class Wav2VecEncoderViewMaker(Wav2VecEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
            "branch_ctc_v1": cfg.branch_ctc_v1,
            "layer_type": 'transformerpos' if cfg.wavlm else 'transformer', 
            #"ch_prune_idx": cfg.ch_prune_idx,
        }
        
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        
        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            self.load_model_weights(state, model, cfg)

        FairseqEncoder.__init__(self, task.source_dictionary)
        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model
        self.viewmaker = ViewMaker1()

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.viewmaker_pretrain_updates = cfg.viewmaker_pretrain_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim
        
        if targ_d is not None:
            self.proj = Linear(d, targ_d)

        self.blank_mode= "add"
        self.blank_weight = 0.
     
    def load_model_weights(self, state, model, cfg):
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            if cfg.init_transformer:
                keys = list(state["model"].keys())
                for key in keys:
                    if 'encoder.layer' in key:
                        del state["model"][key]
                model.load_state_dict(state["model"], strict=False)
            else:
                model.load_state_dict(state["model"], strict=False)
   
    def get_logits(self, net_output, logits, normalize=False):
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output, net_output["encoder_out"])
        logits_new = self.get_logits(net_output, net_output["encoder_out_new"])

        if log_probs:
            return [utils.log_softmax(logits.float(), dim=-1), utils.log_softmax(logits_new.float(), dim=-1)]
        else:
            return [utils.softmax(logits.float(), dim=-1), utils.softmax(logits_new.float(), dim=-1)]

    def forward(self, source, padding_mask, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
            "viewmaker": self.viewmaker,
        }
        
        ft = (self.freeze_finetune_updates <= self.num_updates) or (self.num_updates <= self.viewmaker_pretrain_updates)
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(**w2v_args)

            x = res["x"]
            x_new = res["x_new"]
            padding_mask = res["padding_mask"]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            x_new = x_new.transpose(0, 1)

        x = self.final_dropout(x)
        x_new = self.final_dropout(x_new)
        spk_prob = None

        if self.proj:
            x = self.proj(x)
            x_new = self.proj(x_new)
                    
        return {
            "encoder_out": x,  # T x B x C
            "encoder_out_new": x_new,   # T x B x C 
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
            "loss": res["loss"],
            "conv_feat": None,
            "pac_output": res["pac_output"],
            #"conv_feat": res["conv_feat"],
        }


ACTIVATIONS = { 
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'gelu': torch.nn.GELU,
}


class ViewMaker1(BaseFairseqModel):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=512, distortion_budget=0.02, activation='gelu',
                clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=0, num_noise=0):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()

        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.activation = activation
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.downsample_to = downsample_to
        self.distortion_budget = distortion_budget
        self.num_noise = num_noise
        self.act = ACTIVATIONS[activation]()

        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = ConvLayer2(self.num_channels + self.num_noise, \
                self.num_channels, kernel_size=2, stride=1)
        self.in1 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv2 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in2 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv3 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in3 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv4 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)
        self.in4 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)

        # Residual layers have +N for added random channels
        if not self.num_noise:
            self.res1 = ResidualBlock2(self.num_channels + 1)
            self.res2 = ResidualBlock2(self.num_channels + 2)
            self.res3 = ResidualBlock2(self.num_channels + 3)
            self.res4 = ResidualBlock2(self.num_channels + 4)
            self.res5 = ResidualBlock2(self.num_channels + 5)
            self.conv5 = ConvLayer2(self.num_channels+self.num_res_blocks, \
                self.num_channels, kernel_size=2, stride=1)
        else:
            self.res1 = ResidualBlock2(self.num_channels)
            self.res2 = ResidualBlock2(self.num_channels)
            self.res3 = ResidualBlock2(self.num_channels)
            self.res4 = ResidualBlock2(self.num_channels)
            self.res5 = ResidualBlock2(self.num_channels)
            self.conv5 = ConvLayer2(self.num_channels, \
                self.num_channels, kernel_size=2, stride=1)

        self.ins5 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        self.conv6 = ConvLayer2(self.num_channels, self.num_channels, kernel_size=2, stride=1)

    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # actual 0 has symmetry problems
            init.normal_(m.weight.data, mean=0, std=1e-4)
            # init.constant_(m.weight.data, 0)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1)
        #if x.dtype == 'torch.cuda.float16':
        #    print('fuck'*100)
        #noise.type(torch.cuda.float16)
        noise = noise.half()
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, num_res_blocks=3, bound_multiplier=1):
        if self.num_noise:
            y = self.add_noise_channel(y, num=self.num_noise, bound_multiplier=bound_multiplier)
        y = self.act(self.in1(self.conv1(y)))
        y = self.act(self.in2(self.conv2(y, pad=True)))
        y = self.act(self.in3(self.conv3(y)))
        y = self.act(self.in4(self.conv4(y, pad=True)))

        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = y.clone().mean([-1, -2])

        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                if not self.num_noise:
                    y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))
                else:
                    y = res(y)

        y = self.act(self.ins5(self.conv5(y, pad=True)))
        y = self.conv6(y)

        return y, features

    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta
    
    def get_delta2(self, y_pixels, padding_mask, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        if padding_mask is not None:
            padding_mask_ = torch.logical_not(padding_mask)
            padding_mask_ = padding_mask_.long().unsqueeze(2)
            y_pixels = y_pixels.transpose(1,2)
            y_pixels *= padding_mask_
            y_pixels = y_pixels.transpose(1,2)
        
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, padding_mask):
        x = x.transpose(1,2)
        if self.downsample_to:
            # Downsample.
            x_orig = x
            x = torch.nn.functional.interpolate(
                x, size=(self.downsample_to, self.downsample_to), mode='bilinear')
        y = x

        if self.frequency_domain and 0:
            # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
            # Uses the Discrete Cosine Transform.
            # shape still [batch_size, C, W, H]
            y = dct.dct_2d(y)

        y_pixels, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)
        #delta = self.get_delta(y_pixels.clone())
        delta = self.get_delta2(y_pixels.clone(), padding_mask)
        
        # Additive perturbation
        #result = x + delta
        result = y_pixels

        delta = delta.transpose(1,2)
        result = result.transpose(1,2)

        return result, delta


class ViewMaker2(BaseFairseqModel):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=512, distortion_budget=0.05, activation='gelu', clamp=False, num_noise=5):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()
        
        self.num_channels = num_channels
        self.activation = activation
        self.clamp = clamp
        self.distortion_budget = distortion_budget
        self.num_noise = num_noise
        self.act = ACTIVATIONS[activation]()

        # Initial convolution layers (+ 1 for noise filter)
        self.enc1 = FCLayer(self.num_channels+self.num_noise, self.num_channels)    ## 512 + noise -> 512
        self.enc2 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc3 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        
        self.enc4 = FCLayer(self.num_channels, self.num_channels/2)                 ## 512 -> 256
        self.enc5 = FCLayer(self.num_channels/2, self.num_channels/2)               ## 256 -> 256
        self.enc6 = FCLayer(self.num_channels/2, self.num_channels/2)               ## 256 -> 256
        
        self.enc7 = FCLayer(self.num_channels/2, self.num_channels/4)               ## 256 -> 128
        self.enc8 = FCLayer(self.num_channels/4, self.num_channels/4)               ## 128 -> 128
        self.enc9 = FCLayer(self.num_channels/4, self.num_channels/4)               ## 128 -> 128

        self.mean = FCLayer(self.num_channels/4, self.num_channels/16)              ## 128 -> 32
        self.var = FCLayer(self.num_channels/4, self.num_channels/16)               ## 128 -> 32
        
        self.dec1 = FCLayer(self.num_channels/16, self.num_channels/4)              ## 32 -> 128
        self.dec2 = FCLayer(self.num_channels/4, self.num_channels/4)               ## 128 -> 128
        self.dec3 = FCLayer(self.num_channels/4, self.num_channels/4)               ## 128 -> 128
        
        self.dec4 = FCLayer(self.num_channels/4, self.num_channels/2)               ## 128 -> 256 
        self.dec5 = FCLayer(self.num_channels/2, self.num_channels/2)               ## 256 -> 256 
        self.dec6 = FCLayer(self.num_channels/2, self.num_channels/2)               ## 256 -> 256 
        
        self.dec7 = FCLayer(self.num_channels/2, self.num_channels)                 ## 256 -> 512
        self.dec8 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.dec9 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_().half()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(1)
        shp = (batch_size, filter_size, num)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1)
        noise = noise.half()
        return torch.cat((x, noise), dim=2)

    def encoder(self, y):
        y_residual1 = self.enc1(y)
        y = self.enc2(y_residual1)
        y = self.enc3(y)
        y = y + y_residual1
        
        y_residual2 = self.enc4(y)
        y = self.enc5(y_residual2)
        y = self.enc6(y)
        y = y + y_residual2

        y_residual3 = self.enc7(y)
        y = self.enc8(y_residual3)
        y = self.enc9(y)
        y = y + y_residual3
        return y
    
    def decoder(self, z):
        z_residual1 = self.dec1(z)
        z = self.dec2(z_residual1)
        z = self.dec3(z)
        z = z + z_residual1
        
        z_residual2 = self.dec4(z)
        z = self.dec5(z_residual2)
        z = self.dec6(z)
        z = z + z_residual2

        z_residual3 = self.dec7(z)
        z = self.dec8(z_residual3)
        z = self.dec9(z)
        z = z + z_residual3
        return z

    def basic_net(self, y, bound_multiplier=1):
        y = self.add_noise_channel(y, num=self.num_noise, bound_multiplier=bound_multiplier)
        y = self.encoder(y)
        
        mu, logvar = self.mean(y), self.var(y)
        z = self.reparametrize(mu, logvar)
        
        out = self.decoder(z)
        return out
    
    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, padding_mask):
        out = self.basic_net(x, bound_multiplier=1)
        delta = self.get_delta(out)
        
        # Additive perturbation
        result = x + delta
        if self.clamp:
            result = torch.clamp(result, 0, 1.0)
        
        return result, delta


class ViewMaker3(BaseFairseqModel):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=512, distortion_budget=0.05, activation='gelu', clamp=False, num_noise=0):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()
        
        self.num_channels = num_channels
        self.activation = activation
        self.clamp = clamp
        self.distortion_budget = distortion_budget
        self.num_noise = num_noise
        self.act = ACTIVATIONS[activation]()

        # Initial convolution layers (+ 1 for noise filter)
        self.enc1 = FCLayer(self.num_channels+self.num_noise, self.num_channels)    ## 512 + noise -> 512
        self.enc2 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc3 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        
        self.enc4 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc5 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc6 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        
        self.enc7 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc8 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc9 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        
        self.enc10 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc11 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512
        self.enc12 = FCLayer(self.num_channels, self.num_channels)                   ## 512 -> 512

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(1)
        shp = (batch_size, filter_size, num)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1)
        noise = noise.half()
        return torch.cat((x, noise), dim=2)

    def encoder(self, y):
        y = self.enc1(y)
        y = self.enc2(y)
        y = self.enc3(y)
        y = self.enc4(y)
        y = self.enc5(y)
        y = self.enc6(y)
        y = self.enc7(y)
        y = self.enc8(y)
        y = self.enc9(y)
        y = self.enc10(y)
        y = self.enc11(y)
        y = self.enc12(y)
        return y
    
    def basic_net(self, y, bound_multiplier=1):
        y = self.add_noise_channel(y, num=self.num_noise, bound_multiplier=bound_multiplier)
        y = self.encoder(y)
        out = y
        return out
    
    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, padding_mask):
        result = self.basic_net(x, bound_multiplier=1)
        delta = None
        
        return result, delta


class ViewMaker4(BaseFairseqModel):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=512, distortion_budget=0.01, activation='gelu',
                clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=0, num_noise=0):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()

        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.activation = activation
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.downsample_to = downsample_to
        self.distortion_budget = distortion_budget
        self.num_noise = num_noise
        self.act = ACTIVATIONS[activation]()
        self.group_size = int(self.num_channels/4)

        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = ConvLayer2(self.num_channels + self.num_noise, \
                self.num_channels, kernel_size=2, stride=1, groups=self.group_size)
        self.in1 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv2 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in2 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv3 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in3 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv4 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in4 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv5 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in5 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv6 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in6 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv7 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in7 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv8 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in8 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv9 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in9 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv10 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in10 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv11 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in11 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv12 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in12 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv13 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in13 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv14 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in14 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv15 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in15 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)
        
        self.conv16 = ConvLayer2(self.num_channels, self.num_channels, \
                kernel_size=2, stride=1, groups=self.group_size)
        self.in16 = torch.nn.InstanceNorm1d(self.num_channels, affine=True)

        # Residual layers have +N for added random channels
        if not self.num_noise:
            self.res1 = ResidualBlock2(self.num_channels + 1)
            self.res2 = ResidualBlock2(self.num_channels + 2)
            self.res3 = ResidualBlock2(self.num_channels + 3)
            self.res4 = ResidualBlock2(self.num_channels + 4)
            self.res5 = ResidualBlock2(self.num_channels + 5)
            self.conv5 = ConvLayer2(self.num_channels+self.num_res_blocks, \
                self.num_channels, kernel_size=2, stride=1)
        else:
            self.res1 = ResidualBlock2(self.num_channels)
            self.res2 = ResidualBlock2(self.num_channels)
            self.res3 = ResidualBlock2(self.num_channels)
            self.res4 = ResidualBlock2(self.num_channels)
            self.res5 = ResidualBlock2(self.num_channels)
            self.conv5 = ConvLayer2(self.num_channels, \
                self.num_channels, kernel_size=2, stride=1)

    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # actual 0 has symmetry problems
            init.normal_(m.weight.data, mean=0, std=1e-4)
            # init.constant_(m.weight.data, 0)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1)
        #if x.dtype == 'torch.cuda.float16':
        #    print('fuck'*100)
        #noise.type(torch.cuda.float16)
        noise = noise.half()
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, num_res_blocks=3, bound_multiplier=1):
        if self.num_noise:
            y = self.add_noise_channel(y, num=self.num_noise, bound_multiplier=bound_multiplier)
        y = self.act(self.in1(self.conv1(y)))
        y = self.act(self.in2(self.conv2(y, pad=True)))
        y = self.act(self.in3(self.conv3(y)))
        y = self.act(self.in4(self.conv4(y, pad=True)))

        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = y.clone().mean([-1, -2])

        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                if not self.num_noise:
                    y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))
                else:
                    y = res(y)

        y = self.act(self.in5(self.conv5(y)))
        y = self.act(self.in6(self.conv6(y, pad=True)))
        
        y = self.act(self.in7(self.conv7(y)))
        y = self.act(self.in8(self.conv8(y, pad=True)))
        
        '''
        y = self.act(self.in9(self.conv9(y)))
        y = self.act(self.in10(self.conv10(y, pad=True)))
        
        y = self.act(self.in11(self.conv11(y)))
        y = self.act(self.in12(self.conv12(y, pad=True)))
        
        y = self.act(self.in13(self.conv13(y)))
        y = self.act(self.in14(self.conv14(y, pad=True)))
        '''

        y = self.act(self.in15(self.conv15(y, pad=True)))
        y = self.conv16(y)

        return y, features

    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta
    
    def get_delta2(self, y_pixels, padding_mask, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        if padding_mask is not None:
            padding_mask_ = padding_mask.long().unsqueeze(2)
            y_pixels = y_pixels.transpose(1,2)
            y_pixels *= padding_mask_
            y_pixels = y_pixels.transpose(1,2)
        
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, padding_mask):
        x = x.transpose(1,2)
        if self.downsample_to:
            # Downsample.
            x_orig = x
            x = torch.nn.functional.interpolate(
                x, size=(self.downsample_to, self.downsample_to), mode='bilinear')
        y = x

        if self.frequency_domain and 0:
            # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
            # Uses the Discrete Cosine Transform.
            # shape still [batch_size, C, W, H]
            y = dct.dct_2d(y)

        y_pixels, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)
        #delta = self.get_delta(y_pixels.clone())
        delta = self.get_delta2(y_pixels.clone(), padding_mask)
        
        # Additive perturbation
        #result = x + delta
        result = y_pixels

        delta = delta.transpose(1,2)
        result = result.transpose(1,2)

        return result, delta

# ---

class FCLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation='gelu'):
        super(FCLayer, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.ins = torch.nn.InstanceNorm1d(out_channels, affine=True)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        out = self.linear(x)
        out = out.transpose(1,2)
        out = self.ins(out)
        out = self.act(out)
        out = out.transpose(1,2)

        return out


class ConvLayer2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super(ConvLayer2, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
        self.conv1d = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, groups=groups)

    def forward(self, x, pad=False):
        if pad:
            out = self.reflection_pad(x)
        else:
            out = x
        out = self.conv1d(out)
        return out


class ResidualBlock2(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='gelu'):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=2)
        self.in1 = nn.InstanceNorm1d(channels, affine=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=2)
        self.in2 = nn.InstanceNorm1d(channels, affine=True)
        self.reflection_pad = torch.nn.ReflectionPad1d(1)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(self.reflection_pad(x))))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg: Wav2Vec2Seq2SeqConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
    ):
        super().__init__(dictionary)

        self.dropout = cfg.decoder_dropout
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.output_embed_dim = cfg.decoder_embed_dim

        self.layerdrop = cfg.decoder_layerdrop

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        # TODO: update this when transformer gets converted to dataclass configs
        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(transformer_cfg, no_encoder_attn)
                for _ in range(transformer_cfg.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim**-0.5)

        if transformer_cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["padding_mask"] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                    self_attn_padding_mask=self_attn_padding_mask,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m



class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg: Wav2Vec2Seq2SeqConfig,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
    ):
        super().__init__(dictionary)

        self.dropout = cfg.decoder_dropout
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.output_embed_dim = cfg.decoder_embed_dim

        self.layerdrop = cfg.decoder_layerdrop

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        # TODO: update this when transformer gets converted to dataclass configs
        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(transformer_cfg, no_encoder_attn)
                for _ in range(transformer_cfg.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim**-0.5)

        if transformer_cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["padding_mask"] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                    self_attn_padding_mask=self_attn_padding_mask,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


