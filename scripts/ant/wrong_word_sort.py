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

count = 0
aligned_list = []
wrong_dict1 = {}

for hypo, ref in zip(hypo1_list, ref_list):
    hypo = hypo.split()
    ref = ref.split()
    
    d = editDistance(ref, hypo)
    aligned = getStepList(ref, hypo, d)
    aligned_list.append(aligned)
    
    ref_s, hypo_s = alignedPrint(aligned, ref, hypo)
    for r, h in zip(ref_s, hypo_s):
        try: wrong_dict[(r, h)] += 1
        except: wrong_dict[(r, h)] = 1

for hypo, ref in zip(hypo2_list, ref_list):
    hypo = hypo.split()
    ref = ref.split()
    
    d = editDistance(ref, hypo)
    aligned = getStepList(ref, hypo, d)
    aligned_list.append(aligned)
    
    ref_s, hypo_s = alignedPrint(aligned, ref, hypo)
    for r, h in zip(ref_s, hypo_s):
        try: wrong_dict[(r, h)] += 1
        except: wrong_dict[(r, h)] = 1

wrong_dict = sorted(wrong_dict.items(), key=lambda x:x[1], reverse=True)
for pair, count in wrong_dict:
    print(pair[0], pair[1], count)
