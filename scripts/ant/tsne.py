import os
import numpy as np
import matplotlib.pyplot as plt
from tsnecuda import TSNE
import argparse
import glob

import matplotlib
matplotlib.rc('xtick', labelsize=28)
matplotlib.rc('ytick', labelsize=28)

np_list = None
class_list = sorted(glob.glob('./vanilla_t100_cnnfeat/*'))
first = True

num_list = []
for cls in class_list:
    file_list = sorted(glob.glob(f'{cls}/*'))
    enum = 0
    for enum, file in enumerate(file_list):
        file = np.load(file)
        if first:
            np_list = file.reshape(1, 512)
            first = False
        else:
            np_list = np.concatenate((np_list, file.reshape(1, 512)), axis=0)
    num_list.append(enum-1 if enum != 0 else 0)
print(num_list)

tsne = TSNE(n_components=2, init='random')

output = tsne.fit_transform(np_list)
print(output.shape)
print(sum(num_list))
fig, ax_list = plt.subplots(1,dataset_num, figsize=(11*dataset_num,10))
ax_list = [ax_list]

for ax in ax_list:
    ax.set_xlim(-lim-1, lim+1)
    ax.set_ylim(-lim-1, lim+1)
    
scatter_list = []
#color_list = 
#color_list = ['tab:green', 'tab:purple', 'tab:orange']

for i, ax in enumerate(ax_list):
    scatter_list.append(ax.scatter(output[:, 0], 
                            output[:, 1], 
                            #c = color_list[i],
                            cmap='rainbow',
                            s=2.5, label=labels[i]))

plt.savefig(f'./tsne_dh_res/{title_}/{datanum_}/{trial}/'+title+f'_{key}'+'.png', bbox_inches='tight', dpi=600)
plt.close()

