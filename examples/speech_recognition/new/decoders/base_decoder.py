# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools as it
from typing import Any, Dict, List

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.fairseq_model import FairseqModel
import numpy as np


class BaseDecoder:
    def __init__(self, tgt_dict: Dictionary) -> None:
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)

        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        if "<sep>" in tgt_dict.indices:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict.indices:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.eos()

    def generate(
        self, models: List[FairseqModel], sample: Dict[str, Any], **unused
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        #return (self.decode(emissions[0]), emissions[1], emissions[2])
        return self.decode(emissions)

    def get_emissions(
        self,
        models: List[FairseqModel],
        encoder_input: Dict[str, Any],
    ) -> torch.FloatTensor:
        
        if len(models) == 1:
            model = models[0]
            encoder_out = model(**encoder_input)
                        
            emissions = model.get_normalized_probs(encoder_out, log_probs=False)
            
            temp_out = emissions.cpu().numpy()
            #np.save('/workspace/jieun/temp.npy', temp_out)

            #emissions_numpy = emissions.cpu().numpy()
            #np.save(f'/home/work/workspace/fairseq/scripts/whale/test-clean-part_emissions/0_{emissions.size()[0]}.npy', emissions_numpy)
            '''
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(encoder_out)
            else:
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
                #emissions = model.get_normalized_probs(encoder_out)
            '''
        else:
            encoder_out_all = []
            for model in models:
                encoder_out_all.append(model(**encoder_input))

            encoder_out = encoder_out_all[0]
            encoder_out['encoder_out'] = sum([encoder_out_all[i]['encoder_out'] for i in range(1,len(models))])/float(len(models))
            if hasattr(model, "get_logits"):
                emissions = models[0].get_logits(encoder_out)
            else:
                emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)

        #return (emissions.transpose(0, 1).float().cpu().contiguous(), encoder_out["padding_mask"], encoder_out["conv_feat"])
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        raise NotImplementedError
