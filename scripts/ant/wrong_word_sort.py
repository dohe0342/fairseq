import os
import glob
import sys
from wer import editDistance, getStepList, alignedPrint

infer_log1 = open(f'./None/{sys.argv[1]}/test-other/infer.log', 'r').readlines()
infer_log2 = open(f'./None/{sys.argv[2]}/test-other/infer.log', 'r').readlines()

hypo1_list = []
hypo2_list = []
ref_list = []

for line in infer_log1:
    if 'HYPO:' in line:
        hypo1_list.append(line[50:].replace('\n', ''))
    if 'REF:' in line:
        ref_list.append(line[49:].replace('\n', ''))

for line in infer_log2:
    if 'HYPO:' in line:
        hypo2_list.append(line[50:].replace('\n', ''))

count1 = 0
aligned_list1 = []
wrong_dict1 = {}

count2 = 0
aligned_list2 = []
wrong_dict2 = {}

for hypo, ref in zip(hypo1_list, ref_list):
    hypo = hypo.split()
    ref = ref.split()
    
    d = editDistance(ref, hypo)
    aligned = getStepList(ref, hypo, d)
    aligned_list1.append(aligned)
    
    ref_s, hypo_s = alignedPrint(aligned, ref, hypo)
    for r, h in zip(ref_s, hypo_s):
        try: wrong_dict1[r].append(('vanilla', h))
        except: wrong_dict1[r] = [h]

for hypo, ref in zip(hypo2_list, ref_list):
    hypo = hypo.split()
    ref = ref.split()
    
    d = editDistance(ref, hypo)
    aligned = getStepList(ref, hypo, d)
    aligned_list2.append(aligned)
    
    ref_s, hypo_s = alignedPrint(aligned, ref, hypo)
    for r, h in zip(ref_s, hypo_s):
        try: wrong_dict1[r].append(h)
        except: wrong_dict1[r] = [h]

#wrong_dict1 = sorted(wrong_dict1.items(), key=lambda x:x[1], reverse=True)
#wrong_dict2 = sorted(wrong_dict2.items(), key=lambda x:x[1], reverse=True)

for enum, (r, h) in enumerate(wrong_dict1.items()):
    #print(pair[0], pair[1], count)
    print(r, h)
