import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from tsnecuda import TSNE
import argparse
#from resnet_cifar import ResNet34, ResNet18
from resnet_cifar3 import ResNet34
from torch_resnet import resnet18, resnet34, resnet50
import vgg

import matplotlib
matplotlib.rc('xtick', labelsize=28)
matplotlib.rc('ytick', labelsize=28)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--ce', action='store_true', help='Cross Entropy use')
args = parser.parse_args()

#model = ResNet34()
#model.load_state_dict(torch.load('../models/pretrained/resnet34.pt'))
#model = torchvision.models.resnet50(pretrained=True)
#model.linear = nn.Flatten()

mode = 'save'
datanum = 50000
datanum_ = '50k'
dataset_num = 3
#labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
title = 'origin_di_ours_wgan_cifar100_200k'
title_ = 'origin_di_ours_wgan_cifar100'
#title_ = 'CIFAR10_25k'
#title_list = [f'CIFAR10 original {datanum}', f'DeepInversion 9557 {datanum}', \
#                    f'Large Scale 9557 {datanum}', f'ours 9557 {datanum}']
#title_list = [f'CIFAR10 original {datanum}', f'ours 9557 {datanum}']
#title_list = [f'CIFAR10 original {datanum}', f'wgan {datanum}']
#title_list = ['CIFAR10 original 25k part1', 'CIFAR10 original 25k part2']
title_list = [f'CIFAR100 original {datanum}', f'DeepInversion 7802 {datanum}', f'ours 7802 {datanum}']

trial = 'tsne_v7'
arch = 'resnet50'

dest = f'./tsne_dh_res/{title_}/{datanum_}/{trial}'
if not os.path.exists(dest):
    os.makedirs(dest)

out_target = []
out_output = {'logit':[],
                'emb':[],
                's4_out':[],
                's3_out':[],
                's2_out':[],
                's1_out':[],
                's0_out':[]
                }

if mode == 'save':
    print('here')
    transform_test = transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    #extract = ResNet34(num_classes=100)
    #extract = vgg.__dict__['vgg16_bn'](num_classes=100)
    #extract = extract.to('cuda')
    #checkpoint = torch.load('./pretrained/cifar100_vgg16_7375.pt')
    #checkpoint = torch.load('./pretrained/cifar100_resnet34_7802.pth')
    #extract.load_state_dict(checkpoint)
    #print(extract)
    extract = resnet50(pretrained=True, progress=True).to('cuda')
    #extract = torchvision.models.vgg16(pretrained = True, progress = True)
    #del extract.features[30]
    '''
    del extract.classifier[-1]
    del extract.classifier[-1]
    del extract.classifier[-1]
    del extract.classifier[-1]
    del extract.classifier[-1]
    '''
    print(extract)
    
    extract.cuda()
    extract.eval()

    testset = torchvision.datasets.ImageFolder(root='./dataset/'+title, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logit, emb, s4_out, s3_out, s2_out, s1_out, s0_out = extract(inputs, out_feature=True)
        #emb, _ = extract(inputs)
        #emb = extract(inputs)

        if 0:
            print('logit shape', logit.size())
            print('emb shape', emb.size())
            print('s4_out shape', s4_out.size())
            print('s3_out shape', s3_out.size())
            print('s2_out shape', s2_out.size())
            print('s1_out shape', s1_out.size())
            print('s0_out shape', s0_out.size())
        #print("output shape : ",outputs.shape)
        
        if 0:
            s4_out, s3_out, s2_out, s1_out, s0_out = pooling(s4_out), pooling(s3_out), pooling(s2_out), pooling(s1_out), pooling(s0_out)

        '''
        s4_out = torch.flatten(s4_out, start_dim=1)
        s3_out = torch.flatten(s3_out, start_dim=1)
        s2_out = torch.flatten(s2_out, start_dim=1)
        s1_out = torch.flatten(s1_out, start_dim=1)
        s0_out = torch.flatten(s0_out, start_dim=1)
        '''
        #logit_np = logit.data.cpu().numpy()
        emb_np = emb.data.cpu().numpy()
        #s4_out_np = s4_out.data.cpu().numpy()
        #s3_out_np = s3_out.data.cpu().numpy()
        #s2_out_np = s2_out.data.cpu().numpy()
        #s1_out_np = s1_out.data.cpu().numpy()
        #s0_out_np = s0_out.data.cpu().numpy()

        #out_output['logit'].append(logit_np)
        out_output['emb'].append(emb_np)
        #out_output['s4_out'].append(s4_out_np)
        #out_output['s3_out'].append(s3_out_np)
        #out_output['s2_out'].append(s2_out_np)
        #out_output['s1_out'].append(s1_out_np)
        #out_output['s0_out'].append(s0_out_np)
        
        target_np = targets.data.cpu().numpy()
        out_target.append(target_np[:,np.newaxis])

    target_array = np.concatenate(out_target, axis=0)
    
    for key in out_output:
        if key != 'emb':
            continue
        out_output[key] = np.concatenate(out_output[key], axis=0)
        print(f'{key} shape', out_output[key].shape)
        np.save(f'npy_fid/{arch}_cifar_{title}_{datanum}_target.npy', target_array, allow_pickle=False)
        np.save(f'npy_fid/{arch}_cifar_{title}_{key}_{datanum}_feat.npy', out_output[key], allow_pickle=False)
    print('feature save done')
    exit()

tsne = TSNE(n_components=2, init='random')

for enum, key in enumerate(out_output):
    if enum < 4:
        continue
    #if 's2' in key:
    #    break
    if mode == 'load':
        target_array = np.load(f'./{title}_{datanum}_target.npy')
        out_output[key] = np.load(f'./{title}_{key}_{datanum}_feat.npy').astype(np.float64)
        print(f'{key} feature load done')
        print(f'{key} feature shape {out_output[key].shape}')

    out_output[key] = tsne.fit_transform(out_output[key])

    min_lim = np.around(out_output[key].min())
    max_lim = np.around(out_output[key].max())

    lim = max([-min_lim, max_lim])
    print(f'limit = {lim}')
    print('')

    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(44,10))
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(44,10))
    if dataset_num == 1:
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

