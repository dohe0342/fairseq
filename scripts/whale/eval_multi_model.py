import glob
import os

model_list = sorted(glob.glob('/home/work/workspace/exp/viewmaker_try23_labmda_cosine_annealing_progressive_linear_growing/model/*.pt'))

for model in model_list[::-1]:
    os.system(f"./eval_multimodel.sh {model} 0")
