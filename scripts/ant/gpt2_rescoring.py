from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm


class GPT2Decoder():
    def __init__(self, model_id="gpt2", device="cuda", lm_weight=2.0, ins_p=-0.1):
        self.device = device
        self.model_id = model_id
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

        self.lm_weight = lm_weight
        self.ins_p = ins_p
        self.eos = torch.tensor([50256]).to(self.device).unsqueeze(0)
    
    def lm_score(self, sentence):
        sentence = sentence.lower()

        encodings = self.tokenizer(sentence, return_tensors="pt")
        encodings = encodings.to(self.device)
        input_ids = torch.cat([self.eos, encodings["input_ids"]], dim=1)
        
        score_list = []
        
        with torch.no_grad():
            for i in range(0, len(input_ids[0])-1):
                output = self.model(input_ids[0][:i+1].unsqueeze(0))
                score_list.append(output["logits"][-1][-1][input_ids[0][i+1]])

        return sum(score_list).item()
    
    def update_score(self, sentence, am_score, lm_score):
        final_score = am_score + self.lm_weight*lm_score + self.ins_p*len(sentence.split())
        
        return final_score

    def update_dict(self, score_dict):
        for s, am_score in score_dict.items():
            lm_score = self.lm_score(s)
            final_score = self.update_score(s, am_score, lm_score)
            score_dict[s] = final_score
        score_dict = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)
        #for s, final_score in score_dict:
        #    print(s, final_score)
        #exit()
        return score_dict

if __name__ == "__main__":
    decoder = GPT2Decoder(lm_weight=0.05)
    
    hyps = open('./dev-clean_w2v-b-100h_hypo.txt', 'r').readlines()
    refs = open('./dev-clean.tgt', 'r').readlines()

    score_dict = {}
    for enum, line in enumerate(hyps):
        line = line.strip()
        try: score = float(line)
        except:
            if len(line) != 0:
                score_dict[line] = float(f[enum+1].strip())
            else:
                for s, am_score in score_dict.items():
                    lm_score = decoder.lm_score(s)
                    final_score = decoder.update_score(s, am_score, lm_score)
                    score_dict[s] = final_score
                score_dict = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)
                for s, final_score in score_dict:
                    print(s, final_score)
                exit()
