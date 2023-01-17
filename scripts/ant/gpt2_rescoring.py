from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm


class GPT2Decoder():
    def __init__(self, model_id, device="cuda"):
        device = "cuda"
        model_id = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

text = 'I am a student.'
encodings = tokenizer(text, return_tensors="pt")

print(encodings)
encodings = encodings.to(device)

with torch.no_grad():
    outputs = model(encodings["input_ids"])

print(outputs["logits"].size())
