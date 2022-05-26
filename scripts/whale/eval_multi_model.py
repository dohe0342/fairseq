import glob
import os

model_list = sorted(glob.glob('/home/work/workspace/exp/viewmaker_try26_labmda_cosine_annealing_slow/model/*.pt'))[2:]
print(model_list)
wer_list = []
for enum, model in enumerate(model_list):
    if enum < 10:
        continue
    wer = model.split('_')[-1].replace('.pt','')
    
    if (enum-10)%20 == 0:
        wer_list.append([(enum, float(wer))])
    else:
        wer_list[-1].append((enum, float(wer)))
    print(wer_list)
    #print(wer)
    #os.system(f"./eval_multimodel.sh {model} 0")

print(wer_list)
#wer_list = sorted(wer_list, key=lambda x:x[1])

#for wer in wer_list:
#    print(wer)
#    model = model_list[wer[0]]
#    os.system(f"./eval_multimodel.sh {model} 0")
