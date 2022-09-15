import os
import cifar10.model_loader
from model import w2v_load


def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'LibriSpeech':
        net = w2v_load('/home/work/workspace/models/wav2vec_model/wav2vec_small_100h.pt', data_parallel) 
    return net

if __name__ == '__main__':
    load('LibriSpeech'
