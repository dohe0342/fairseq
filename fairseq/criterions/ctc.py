# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("ctc_fgsm", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg
    
    def forward_and_get_fgsm(self, model, sample, optimizer, ignore_grad=False):
        origin = sample["net_input"]["source"].clone()
        diff_able = torch.autograd.Variable(sample["net_input"]["source"].data, requires_grad=True)
        sample["net_input"]["source"] = diff_able
        
        net_output = model(**sample["net_input"])
        
        #for n, p in model.named_parameters():
        #    if 'feature_extractor' in n:
        #        p.requires_grad = False
        
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
        
        if sample["net_input"]["source"].grad is not None:
            sample["net_input"]["source"].grad.data.fill_(0)
        
        if ignore_grad:
            loss *= 0

        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss, retain_graph=True)
        
        '''
        for n, p in model.named_parameters():
            if 'feature_extractor' in n:
                print(p.grad)
            else:
                print('gradient = ', p.grad)
        '''

        eps = 0.02
        sample["net_input"]["source"].grad.sign_()

        origin = torch.norm(origin, dim=1)
        noise = torch.norm(eps*sample["net_input"]["source"].grad.clone(), dim=1)
        
        snr = torch.log10(20*(origin/noise))
        
        sample["net_input"]["source"] = sample["net_input"]["source"] + eps*sample["net_input"]["source"].grad
        
        snr_avg = snr.sum() / origin.size()[0]
        
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "snr": snr_avg,
        }
        del loss
        return None, sample_size, logging_output
    
    def forward_fgsm(self, model, sample, logging_output, reduce=True):
        #print(sample["net_input"]["source"])
        #set_grad = torch.autograd.Variable(sample["net_input"]["source"].data, requires_grad=True)
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output["loss fgsm"] = utils.item(loss.data)

        return loss, sample_size, logging_output
    
    def forward_and_get_cnn_fgsm(self, model, sample, optimizer, ignore_grad=False):
        sample["net_input"]["cnn_fgsm"] = True
        
        net_output = model(**sample["net_input"])
        
        #for n, p in model.named_parameters():
        #    if 'feature_extractor' in n:
        #        p.requires_grad = False
        
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
        
        if ignore_grad:
            loss *= 0

        conv_feat = net_output["conv_feat"]
        if conv_feat.grad is not None:
            conv_feat.grad.data.fill_(0)

        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss, retain_graph=True)
        
        eps = 0.02

        origin = conv_feat.data.clone()

        conv_feat.grad.sign_()
        conv_feat = conv_feat + eps*conv_feat.grad 
        
        if 0:
            origin = origin.reshape(-1, 512)
            origin = origin/origin.norm(dim=1).unsqueeze(1)
            
            conv_feat_ = conv_feat.reshape(-1, 512)
            conv_feat_ = conv_feat_ / conv_feat_.norm(dim=1).unsqueeze(1)

            sim = torch.mm(origin, conv_feat_.T)
            sim = sim.diagonal().sum()/sim.size()[0]
        
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "snr": 0.,
        }
        del loss
        
        return conv_feat, sample_size, logging_output
    
    def forward_cnn_fgsm(self, model, sample, logging_output, conv_feat, reduce=True):
        del sample["net_input"]["cnn_fgsm"]
        sample["net_input"]["cnn_feat"] = conv_feat

        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output["loss fgsm"] = utils.item(loss.data)

        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        #print(sample["net_input"]["source"])
        #set_grad = torch.autograd.Variable(sample["net_input"]["source"].data, requires_grad=True)
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "snr": 0.,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        loss_fgsm_sum = utils.item(sum(log.get("loss fgsm", 0) for log in logging_outputs))
        snr_sum = utils.item(sum(log.get("snr", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss fgsm", loss_fgsm_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "snr", snr_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

@register_criterion("viewmaker", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.w2v_encoder.get_normalized_probs(
            net_output, log_probs=True
        )#.contiguous()  # (T, B, C) from the encoder

        lprobs2 = lprobs[1].contiguous()
        lprobs = lprobs[0].contiguous()

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        if lprobs.size()[1] != input_lengths.size()[0]:
            input_lengths = input_lengths[:int(input_lengths.size()[0]/2)]
        
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = []
            if 0:
                print('lprobs size = ', torch.cuda.current_device(), lprobs.size())
                print('target flat size = ', torch.cuda.current_device(), targets_flat.size())
                print('input lengths size = ', torch.cuda.current_device(), input_lengths.size())
            loss.append(F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            ))
            
            loss.append(F.ctc_loss(
                lprobs2,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            ))

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss[0].data),  # * sample['ntokens'],
            "loss viewmaker ctc": utils.item(loss[1].data),  # * sample['ntokens'],
            "loss mse": utils.item(net_output["loss"].data) if net_output["loss"] is not None else 0.,
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs = [lprobs, lprobs2]
                for i, lprob in enumerate(lprobs):
                    lprobs_t = lprob.transpose(0, 1).float().contiguous().cpu()

                    c_err = 0
                    c_len = 0
                    w_errs = 0
                    w_len = 0
                    wv_errs = 0
                    for lp, t, inp_l in zip(
                        lprobs_t,
                        sample["target_label"]
                        if "target_label" in sample
                        else sample["target"],
                        input_lengths,
                    ):
                        lp = lp[:inp_l].unsqueeze(0)

                        decoded = None
                        if self.w2l_decoder is not None:
                            decoded = self.w2l_decoder.decode(lp)
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]
                                if len(decoded) < 1:
                                    decoded = None
                                else:
                                    decoded = decoded[0]

                        p = (t != self.task.target_dictionary.pad()) & (
                            t != self.task.target_dictionary.eos()
                        )
                        targ = t[p]
                        targ_units = self.task.target_dictionary.string(targ)
                        targ_units_arr = targ.tolist()

                        toks = lp.argmax(dim=-1).unique_consecutive()
                        pred_units_arr = toks[toks != self.blank_idx].tolist()

                        c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                        c_len += len(targ_units_arr)

                        targ_words = post_process(targ_units, self.post_process).split()

                        pred_units = self.task.target_dictionary.string(pred_units_arr)
                        pred_words_raw = post_process(pred_units, self.post_process).split()

                        if decoded is not None and "words" in decoded:
                            pred_words = decoded["words"]
                            w_errs += editdistance.eval(pred_words, targ_words)
                            wv_errs += editdistance.eval(pred_words_raw, targ_words)
                        else:
                            dist = editdistance.eval(pred_words_raw, targ_words)
                            w_errs += dist
                            wv_errs += dist

                        w_len += len(targ_words)
                    
                    if i == 0:
                        logging_output["wv_errors"] = wv_errs
                        logging_output["w_errors"] = w_errs
                        logging_output["w_total"] = w_len
                        logging_output["c_errors"] = c_err
                        logging_output["c_total"] = c_len
                    elif i == 1:
                        logging_output["wv_errors viewmaker"] = wv_errs
                        logging_output["w_errors viewmaker"] = w_errs
                        logging_output["w_total viewmaker"] = w_len
                        logging_output["c_errors viewmaker"] = c_err
                        logging_output["c_total viewmaker"] = c_len
 
        return [loss, net_output["loss"]], sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        loss_viewmaker_ctc_sum = utils.item(sum(log.get("loss viewmaker ctc", 0) for log in logging_outputs))
        loss_mse_sum = utils.item(sum(log.get("loss mse", 0) for log in logging_outputs))
        
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss viewmaker", loss_viewmaker_ctc_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss mse", loss_mse_sum / sample_size / math.log(2), sample_size, round=8
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)
        
        c_errors_new = sum(log.get("c_errors viewmaker", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors viewmaker", c_errors)
        c_total_new = sum(log.get("c_total viewmaker", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total viewmaker", c_total)
        w_errors_new = sum(log.get("w_errors viewmaker", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors viewmaker", w_errors)
        wv_errors_new = sum(log.get("wv_errors viewmaker", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors viewmaker", wv_errors)
        w_total_new = sum(log.get("w_total viewmaker", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total viewmaker", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
            
            metrics.log_derived(
                "uer viewmaker",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )

        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("branch_ctc_v1", dataclass=CtcCriterionConfig)
class BranchCtcCriterionV1(CtcCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(CtcCriterionConfig, task)
    def forward(self, model, sample, reduce=True, tgt_layer=0):
        net_output = model(tgt_layer=tgt_layer, **sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            f"loss_{tgt_layer}": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output[f"wv_errors_{tgt_layer}"] = wv_errs
                logging_output[f"w_errors_{tgt_layer}"] = w_errs
                logging_output[f"w_total_{tgt_layer}"] = w_len
                logging_output[f"c_errors_{tgt_layer}"] = c_err
                logging_output[f"c_total_{tgt_layer}"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, layer_num=12) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get(f"loss_{layer_num}", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            f"loss_{layer_num}", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get(f"c_errors_{layer_num}", 0) for log in logging_outputs)
        metrics.log_scalar(f"_c_errors_{layer_num}", c_errors)
        c_total = sum(log.get(f"c_total_{layer_num}", 0) for log in logging_outputs)
        metrics.log_scalar(f"_c_total_{layer_num}", c_total)
        w_errors = sum(log.get(f"w_errors_{layer_num}", 0) for log in logging_outputs)
        metrics.log_scalar(f"_w_errors_{layer_num}", w_errors)
        wv_errors = sum(log.get(f"wv_errors_{layer_num}", 0) for log in logging_outputs)
        metrics.log_scalar(f"_wv_errors_{layer_num}", wv_errors)
        w_total = sum(log.get(f"w_total_{layer_num}", 0) for log in logging_outputs)
        metrics.log_scalar(f"_w_total_{layer_num}", w_total)
        
        if c_total > 0:
            metrics.log_derived(
                f"uer_{layer_num}",
                lambda meters: safe_round(
                    meters[f"_c_errors_{layer_num}"].sum * 100.0 / meters[f"_c_total_{layer_num}"].sum, 3
                )
                if meters[f"_c_total_{layer_num}"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                f"wer_{layer_num}",
                lambda meters: safe_round(
                    meters[f"_w_errors_{layer_num}"].sum * 100.0 / meters[f"_w_total_{layer_num}"].sum, 3
                )
                if meters[f"_w_total_{layer_num}"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                f"raw_wer_{layer_num}",
                lambda meters: safe_round(
                    meters[f"_wv_errors_{layer_num}"].sum * 100.0 / meters[f"_w_total_{layer_num}"].sum, 3
                )
                if meters[f"_w_total_{layer_num}"].sum > 0
                else float("nan"),
            )


@register_criterion("branch_ctc_v2", dataclass=CtcCriterionConfig)
class BranchCtcCriterionV2(CtcCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(CtcCriterionConfig, task)
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs_list = model.w2v_encoder.get_normalized_probs(
            net_output, log_probs=True
        )#.contiguous()  # (T, B, C) from the encoder

        lprobs_list = [lprobs.contiguous() for lprobs in lprobs_list]

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs_list[0].new_full(
                    (lprobs_list[0].size(1),), lprobs_list[0].size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)
        
        with torch.backends.cudnn.flags(enabled=False):
            loss_list = [F.ctc_loss(
                    lprobs,
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                ) for lprobs in lprobs_list]

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        
        logging_output = {}
        
        loss_log_list = [loss.data for loss in loss_list]

        for drop in net_output['dropped_layer']:
            lprobs_list.insert(drop, 0)
            loss_log_list.insert(drop, 0)
        
        logging_output = {
                f"loss_12": utils.item(loss_list[-1].data),
                "ntokens": ntokens,
                "nsentences": sample["id"].numel(),
                "sample_size": sample_size,
        }
        
        '''
        for i, loss in enumerate(loss_log_list):
            if loss == 0:
                logging_output[f"loss_{i+7}"] = torch.tensor(0.)
            else:
                logging_output[f"loss_{i+7}"] = utils.item(loss)
        '''

        if not model.training:
            import editdistance

            with torch.no_grad():
                for enum, lprobs in enumerate(lprobs_list[::-1]):
                    if type(lprobs) == int:
                        tgt_layer = 12-enum
                        logging_output[f"wv_errors_{tgt_layer}"] = 0
                        logging_output[f"w_errors_{tgt_layer}"] = 0
                        logging_output[f"w_total_{tgt_layer}"] = 0
                        logging_output[f"c_errors_{tgt_layer}"] = 0
                        logging_output[f"c_total_{tgt_layer}"] = 0
                        continue
                    
                    lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                    c_err = 0
                    c_len = 0
                    w_errs = 0
                    w_len = 0
                    wv_errs = 0
                    for lp, t, inp_l in zip(
                        lprobs_t,
                        sample["target_label"]
                        if "target_label" in sample
                        else sample["target"],
                        input_lengths,
                    ):
                        lp = lp[:inp_l].unsqueeze(0)

                        decoded = None
                        if self.w2l_decoder is not None:
                            decoded = self.w2l_decoder.decode(lp)
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]
                                if len(decoded) < 1:
                                    decoded = None
                                else:
                                    decoded = decoded[0]

                        p = (t != self.task.target_dictionary.pad()) & (
                            t != self.task.target_dictionary.eos()
                        )
                        targ = t[p]
                        targ_units = self.task.target_dictionary.string(targ)
                        targ_units_arr = targ.tolist()

                        toks = lp.argmax(dim=-1).unique_consecutive()
                        pred_units_arr = toks[toks != self.blank_idx].tolist()

                        c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                        c_len += len(targ_units_arr)

                        targ_words = post_process(targ_units, self.post_process).split()

                        pred_units = self.task.target_dictionary.string(pred_units_arr)
                        pred_words_raw = post_process(pred_units, self.post_process).split()

                        if decoded is not None and "words" in decoded:
                            pred_words = decoded["words"]
                            w_errs += editdistance.eval(pred_words, targ_words)
                            wv_errs += editdistance.eval(pred_words_raw, targ_words)
                        else:
                            dist = editdistance.eval(pred_words_raw, targ_words)
                            w_errs += dist
                            wv_errs += dist

                        w_len += len(targ_words)
                    
                    tgt_layer = 12-enum
                    logging_output[f"wv_errors_{tgt_layer}"] = wv_errs
                    logging_output[f"w_errors_{tgt_layer}"] = w_errs
                    logging_output[f"w_total_{tgt_layer}"] = w_len
                    logging_output[f"c_errors_{tgt_layer}"] = c_err
                    logging_output[f"c_total_{tgt_layer}"] = c_len

        return loss_list, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, layer_num=12) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get(f"loss_{layer_num}", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            f"loss_{layer_num}", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        
        for layer_num in range(7, 13):
            c_errors = sum(log.get(f"c_errors_{layer_num}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_c_errors_{layer_num}", c_errors)
            c_total = sum(log.get(f"c_total_{layer_num}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_c_total_{layer_num}", c_total)
            w_errors = sum(log.get(f"w_errors_{layer_num}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_w_errors_{layer_num}", w_errors)
            wv_errors = sum(log.get(f"wv_errors_{layer_num}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_wv_errors_{layer_num}", wv_errors)
            w_total = sum(log.get(f"w_total_{layer_num}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_w_total_{layer_num}", w_total)
            
            if c_total > 0:
                metrics.log_derived(
                    f"uer_{layer_num}",
                    lambda meters: safe_round(
                        meters[f"_c_errors_{layer_num}"].sum * 100.0 / meters[f"_c_total_{layer_num}"].sum, 3
                    )
                    if meters[f"_c_total_{layer_num}"].sum > 0
                    else float("nan"),
                )
            if w_total > 0:
                metrics.log_derived(
                    f"wer_{layer_num}",
                    lambda meters: safe_round(
                        meters[f"_w_errors_{layer_num}"].sum * 100.0 / meters[f"_w_total_{layer_num}"].sum, 3
                    )
                    if meters[f"_w_total_{layer_num}"].sum > 0
                    else float("nan"),
                )
                metrics.log_derived(
                    f"raw_wer_{layer_num}",
                    lambda meters: safe_round(
                        meters[f"_wv_errors_{layer_num}"].sum * 100.0 / meters[f"_w_total_{layer_num}"].sum, 3
                    )
                    if meters[f"_w_total_{layer_num}"].sum > 0
                    else float("nan"),
                )


@register_criterion("spk_clf", dataclass=CtcCriterionConfig)
class SpeakerClassification(CtcCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(CtcCriterionConfig, task)
        self.tsv = open('/home/work/workspace/LibriSpeech/manifests/train-100.tsv', 'r').readlines()
        self.spk = open(f'/home/work/workspace/LibriSpeech/manifests/train-100.spk', 'r').readlines()
        self.spk = [int(i.split('\n')[0]) for i in self.spk]
        self.spk_idx = {}
        for i, spk in enumerate(self.spk):
            self.spk_idx[spk] = i 

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        
        features = None
        target = []
    
        for id in sample['id']:
            target.append(self.spk_idx[int(self.tsv[id+1].split('/')[0])])
    
        target = torch.LongTensor(target).to('cuda')

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)
        
        loss_spk = None
        with torch.backends.cudnn.flags(enabled=False):
            loss_ctc = F.ctc_loss(
                    lprobs,
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                ) 
            if net_output['spk_prob'] != None:
                loss_spk = self.criterion(net_output['spk_prob'], target)
                loss = loss_ctc - 10.*loss_spk
            else:
                loss = loss_ctc
        
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "loss_ctc": utils.item(loss_ctc.data),
            "loss_spk": utils.item(loss_spk.data) if loss_spk != None else 0.,
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len
        
        if 1:
            return [loss_ctc, loss_spk], sample_size, logging_output

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        loss_sum_ctc = utils.item(sum(log.get("loss_ctc", 0) for log in logging_outputs))
        loss_sum_spk = utils.item(sum(log.get("loss_spk", 0) for log in logging_outputs))
        
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_ctc", loss_sum_ctc / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_spk", loss_sum_spk / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )


@register_criterion("spk_clf_v2", dataclass=CtcCriterionConfig)
class SpeakerClassification(CtcCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask):
        super().__init__(CtcCriterionConfig, task)
        self.tsv = open('/home/work/workspace/LibriSpeech/manifests/train-100.tsv', 'r').readlines()
        self.spk = open(f'/home/work/workspace/LibriSpeech/manifests/train-100.spk', 'r').readlines()
        self.spk = [int(i.split('\n')[0]) for i in self.spk]
        self.spk_idx = {}
        for i, spk in enumerate(self.spk):
            self.spk_idx[spk] = i 

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        
        features = None
        target = []
    
        for id in sample['id']:
            target.append(self.spk_idx[int(self.tsv[id+1].split('/')[0])])
    
        target = torch.LongTensor(target).to('cuda')

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)
        
        loss_spk = None
        with torch.backends.cudnn.flags(enabled=False):
            loss_ctc = F.ctc_loss(
                    lprobs,
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                ) 
            if net_output['spk_prob'] != None:
                loss_spk = self.criterion(net_output['spk_prob'], target)
                loss = loss_ctc - 10.*loss_spk
            else:
                loss = loss_ctc
        
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "loss_ctc": utils.item(loss_ctc.data),
            "loss_spk": utils.item(loss_spk.data) if loss_spk != None else 0.,
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len
        
        if 1:
            return [loss_ctc, loss_spk], sample_size, logging_output

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        loss_sum_ctc = utils.item(sum(log.get("loss_ctc", 0) for log in logging_outputs))
        loss_sum_spk = utils.item(sum(log.get("loss_spk", 0) for log in logging_outputs))
        
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_ctc", loss_sum_ctc / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_spk", loss_sum_spk / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

