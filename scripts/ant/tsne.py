import os
import numpy as np
import matplotlib.pyplot as plt
from tsnecuda import TSNE
import argparse
import glob
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib
matplotlib.rc('xtick', labelsize=28)
matplotlib.rc('ytick', labelsize=28)

np_list = None
class_list = sorted(glob.glob('./vanilla_t100_cnnfeat/*'))
first = True

num_list = []
for cls in tqdm(class_list):
    file_list = sorted(glob.glob(f'{cls}/*'))
    enum = 0
    for enum in tqdm(range(len(file_list)), leave=False):
        file = np.load(file_list[enum])
        if first:
            np_list = file.reshape(1, 512)
            first = False
        else:
            np_list = np.concatenate((np_list, file.reshape(1, 512)), axis=0)
        
        if enum >= 100:
            break

    num_list.append(enum+1 if enum != 0 else 0)
print(num_list)

tsne = TSNE(n_components=2, init='random')

output = tsne.fit_transform(np_list)
print(np_list.shape)
print(output.shape)
print(sum(num_list))

colors = cm.rainbow(np.linspace(0, 1, len(num_list)))

for enum in range(output.shape[0]):
    plt.scatter(output[enum][0], output[enum][1])

plt.savefig('./vanilla_decision.png', bbox_inches='tight', dpi=300)
plt.close()
