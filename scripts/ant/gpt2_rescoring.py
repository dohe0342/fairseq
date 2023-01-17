from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm


class GPT2Decoder():
    def __init__(self, model_id="gpt2", device="cuda", lm_weight=2.0, ins_p=-0.5):
        self.device = device
        self.model_id = model_id
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

        self.lm_weight = lm_weight
        self.ins_p = ins_p
        self.eos = 50256
    
    def score(self, sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        encodings = encodings.to(device)
        input_ids = [self.eos] + encodings["input_ids"]
        
        score_list = []
        
        with torch.no_grad():
            for i in range(1, len(input_ids)):
                output = self.model(input_ids[:i])


print(encodings)
encodings = encodings.to(device)

with torch.no_grad():
    outputs = model(encodings["input_ids"])

print(outputs["logits"].size())
