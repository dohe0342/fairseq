import os
import numpy as np
import matplotlib.pyplot as plt
from tsnecuda import TSNE
import argparse
import glob
from tqdm import tqdm

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
    for enum, file in tqdm(enumerate(file_list), leave=False):
        file = np.load(file)
        if first:
            np_list = file.reshape(1, 512)
            first = False
        else:
            np_list = np.concatenate((np_list, file.reshape(1, 512)), axis=0)
    num_list.append(enum)
print(num_list)

tsne = TSNE(n_components=2, init='random')

output = tsne.fit_transform(np_list)
print(output.shape)
print(sum(num_list))

colors = cm.rainbow(np.linspace(0, 1, len(num_list)))

for o, c in zip(output, colors):
    plt.scatter(o[0], o[1], color=
scatter_list.append(ax.scatter(output[:, 0], 
                        output[:, 1], 
                        #c = color_list[i],
                        cmap='rainbow',
                        s=2.5, label=labels[i]))

plt.savefig(f'./tsne_dh_res/{title_}/{datanum_}/{trial}/'+title+f'_{key}'+'.png', bbox_inches='tight', dpi=600)
plt.close()

