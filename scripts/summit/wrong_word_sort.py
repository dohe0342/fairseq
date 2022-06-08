import os
import glob
from wer import editDistance, getStepList

infer_log = open('./None/test-clean/infer.log', 'r').readlines()

hypo_list = []
ref_list = []

for line in infer_log:
    if 'HYPO:' in line:
        hypo_list.append(line[50:].replace('\n', ''))
    if 'REF:' in line:
        ref_list.append(line[49:].replace('\n', ''))


count = 0
aligned_list = []
wrong_dict = {}
for hypo, ref in zip(hypo_list, ref_list):
    hypo = hypo.split(' ')
    ref = ref.split(' ')
    #print(hypo)
    #print(ref)
    #print('')
    if len(hypo) != len(ref):
        d = editDistance(hypo, ref)
        aligned_list.append(getStepList(hypo, ref))
    else:
        for h, r in zip(hypo, ref):
            if h != r:
                try: wrong_dict[(h, r)] += 1
                except: wrong_dict[(h, r)] = 0

print(aligned_list)
exit()
wrong_dict = sorted(wrong_dict.items(), key=lambda x:x[1], reverse=False)
for pair, count in wrong_dict:
    print(pair, count)
