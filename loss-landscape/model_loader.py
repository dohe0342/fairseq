import os
import cifar10.model_loader
from fairseq.models.wav2vec.wav2vec2_asr import wav2vec_ctc

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)

    elif dataset == 'LibriSpeech':
        net = 
    return net
