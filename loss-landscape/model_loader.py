import os
import cifar10.model_loader
from model import Wav2Vec2Ctc

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)

    return net

if __name__ == '__main__':
    net = Wav2Vec2Ctc()
    net.overwrite_param('/home/work/workspace/models/wav2vec_model/wav2vec_small_100h.pt')
    print(net)
