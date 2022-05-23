import glob
import os

model_list = sorted(glob.glob('/home/work/workspace/exp/viewmaker_try23_lambda_cosine_annealing_progressive_linear_growing/model/*.pt'))
print(model_list)

for model in model_list:
    print(model)
    os.system(f"./eval_multimodel.sh {model} 0")
