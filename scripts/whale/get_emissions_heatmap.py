import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm

filelist = glob.glob('./test-clean-part_emissions/*.npy')
for filename in tqdm(filelist):
    emissions = np.load(filename)
    emissions = emissions.transpose((1, 2, 0))
    model_number = filename.split('/')[-1].split('_')[0].strip()
    for b, emission in tqdm(enumerate(emissions), leave=False):
        plt.matshow(emission)
        plt.colorbar()
        plt.savefig(f'./test-clean-part_emissions/{model_number}_{str(b).zfill(2)}.png', dpi=300)
        plt.close()
