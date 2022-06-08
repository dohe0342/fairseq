import os
import glob

infer_log = open('./None/test-clean/infer.log', 'r').readlines()

hypo_list = []
ref_list = []

for line in infer_log:
    if 'HYPO' in line:
        hypo_list.append(line)
    if 'REF' in line:
        ref_list.append(line)

print(hypo_list)
