import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
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

for cls in class_list:
    file_list = sorted(glob.glob(f'{cls}/*'))
    for enum, file in enumerate(file_list):
        if enum > 500:
            break

        file = np.load(file)
        #if np_list.shape[0] == 1:
        if first:
            np_list = file.reshape(1, 512)
            first = False
        else:
            np_list = np.concatenate((np_list, file.reshape(1, 512)), axis=0)
        print('np shape = ', np_list.shape)

tsne = TSNE(n_components=2, init='random')

output = tsne.fit_transform(np_list)

min_lim = np.around(out_output[key].min())
max_lim = np.around(out_output[key].max())

lim = max([-min_lim, max_lim])
print(f'limit = {lim}')
print('')

if dataset_num == 1 or 1:
    fig, ax_list = plt.subplots(1,dataset_num, figsize=(11*dataset_num,10))
    ax_list = [ax_list]
else:
    fig, ax_list = plt.subplots(1,dataset_num, figsize=(11*dataset_num,10))

print('ax list length = ', len(ax_list))

for ax in ax_list:
    ax.set_xlim(-lim-1, lim+1)
    ax.set_ylim(-lim-1, lim+1)
    
output_array = out_output[key]

scatter_list = []
labels = ['Original CIFAR-10', 'DeepInversion', 'Ours']
color_list = ['tab:green', 'tab:purple', 'tab:orange']

for i, ax in enumerate(ax_list):
    scatter_list.append(ax.scatter(output_array[datanum*i:datanum*(i+1), 0], 
                            output_array[datanum*i:datanum*(i+1), 1], 
                            #c = target_array[datanum*i:datanum*(i+1),0],
                            c = color_list[i],
                            s=2.5, label=labels[i]))
                            #cmap = plt.cm.get_cmap('rainbow', 10), label=labels))

if enum >= 5:  
    for i, ax in enumerate(ax_list):
        #ax.set_title(title_list[i])
        #ax.legend(scatter_list[i].legend_elements()[0], labels)
        ax.legend(fontsize=32, markerscale=10, loc='lower right')

#plt.suptitle(f'{key} {datanum} T-SNE result', fontsize=30)
plt.savefig(f'./tsne_dh_res/{title_}/{datanum_}/{trial}/'+title+f'_{key}'+'.png', bbox_inches='tight', dpi=600)
plt.close()

