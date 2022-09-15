import os
import cifar10.model_loader
import 

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)

    return net

if __name__ == '__main__':
    
