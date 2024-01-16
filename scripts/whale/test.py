import numpy as np
import torch
import matplotlib as plt 
plt.matshow(lm_am_sim_cp.numpy())
plt.colorbar()
if not os.path.exists(f'/home/work/workspace/fairseq/scripts/whale/png/cross_attn'):
    try: os.makedirs(f'/home/work/workspace/fairseq/scripts/whale/png/cross_attn')
    except: pass
plt.savefig(f'/home/work/workspace/fairseq/scripts/whale/png/aross_attn/cross_attn.png')
