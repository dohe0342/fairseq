import glob
import os

model_list = sorted(glob.glob('/home/work/workspace/exp/viewmaker_try23_lambda_cosine_annealing_progressive_lienar_growing/model/*.pt'))

for model in model_list:
    os.system(f"./eval_multimodel.sh {model} 0")
