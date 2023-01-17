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
        self.eos = torch.tensor([50256]).to(self.device).unsqueeze(0)
    
    def score(self, sentence):
        sentence = sentence.lower()
        sentence[0] = sentence[0].upper()

        encodings = self.tokenizer(sentence, return_tensors="pt")
        encodings = encodings.to(self.device)
        #print(self.eos.size())
        #print(encodings["input_ids"].size())
        input_ids = torch.cat([self.eos, encodings["input_ids"]], dim=1)
        
        score_list = []
        
        with torch.no_grad():
            for i in range(0, len(input_ids[0])-1):
                output = self.model(input_ids[0][:i+1].unsqueeze(0))
                #print(output["logits"].size())
                score_list.append(output["logits"][-1][-1][input_ids[0][i+1]])
                #print(score_list[-1])
        print(sum(score_list))

        return None


if __name__ == "__main__":
    decoder = GPT2Decoder()
    
    f = open('./dev-clean_w2v-b-100h_hypo.txt', 'r').readlines()
    score_dict = {}
    for enum, line in enumerate(f):
        line = line.strip()
        try: score = float(line)
        except:
            if len(line) != 0:
                score_dict[line] = float(f[enum+1].strip())
            else:
                #print(score_dict)
                for s, am_score in score_dict.items():
                    decoder.score(s)
                exit()
    
    
