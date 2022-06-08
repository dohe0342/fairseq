import os
import glob

infer_log = open('./None/test-clean/infer.log', 'r').readlines()

hypo_list = []
ref_list = []

for line in infer_log:
    if 'HYPO' in line:
        hypo_list.append(line[50:].replace('\n', ''))
    if 'REF' in line:
        ref_list.append(line[49:].replace('\n', ''))

print(hypo_list)
print('\n\n\n')
#print(ref_list)
