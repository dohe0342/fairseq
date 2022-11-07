import matplotlib.pyplot as plt
import numpy as np
import glob

filelist = glob.glob('./test-clean-part_emissions/*.npy')
for filname in filelist:
    emissions = np.load(filename)
    model_number = filename.split('/')[-1].split('_')[0].strip()
    for b, emission in enumerate(emissions):
        plt.matshow(emission)
        plt.colorbar()
        plt.savefig(f'{model_number}_{str(b).zfill(2)}', dpi=300)
