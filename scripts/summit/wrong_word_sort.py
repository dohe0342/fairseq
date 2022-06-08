import os
import glob

infer_log = open('./None/test-clean/infer.log', 'r').readlines()

hypo_list = []
ref_list = []

for line in infer_log:
    if 'HYPO:' in line:
        hypo_list.append(line[50:].replace('\n', ''))
    if 'REF:' in line:
        ref_list.append(line[49:].replace('\n', ''))

count = 0
for hypo, ref in zip(hypo_list, ref_list):
    #print(hypo)
    #print(ref)
    #print('')
    if len(hypo.split(' ')) != len(ref.split(' ')):
        print(hypo)
        print(ref)
        print('')
        count += 1

print(count)
#print(len(hypo_list))
#print(len(ref_list))
