#!/usr/bin/env python

'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Trains a PCA model and adds a WPCA layer to an existing checkpoint.
'''


from __future__ import print_function

import argparse
import configparser
import os
import random
from os.path import join, isfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np

from patchnetvlad.training_tools.tools import pca
from patchnetvlad.tools.datasets import input_transform
from patchnetvlad.models.models_generic import get_backend, get_model, Flatten, L2Norm
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from tqdm.auto import tqdm

from patchnetvlad.training_tools.msls import MSLS, ImagesFromList
from patchnetvlad.tools.datasets import PlaceDataset
from pittsburgh import prefix_data_in

if __name__ == "__main__":

    config_path = join(PATCHNETVLAD_ROOT_DIR, 'configs/train.ini')
    configfile = config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    dataset_root_dir = prefix_data_in + config['train']['dataset_root_dir']
    threads = int(config['global_params']['threads'])
    nocuda = config['train'].getboolean('nocuda')
    dataset_choice = config['train']['dataset_choice']

    cuda = not nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    if config['global_params']['pooling'] == 'netvlad':
        if int(config['global_params']['density_L']) == 10:
            scaleSpace = [2,3,4,5,6,7,8,9,10]
        elif int(config['global_params']['density_L']) ==  3:
            scaleSpace = [2,4]

    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(int(config['train']['seed']))

    print('===> Building model')

    encoder_dim, encoder = get_backend(config['train']['arch'],int(config['train']['trim']))
    resume_path = join(config['global_params']['resumepath'],"checkpoint.pth.tar")
    if resume_path: # must resume for PCA
        if isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
            config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

            model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=False)

            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}'".format(resume_path, ))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_path))
    else:
        raise ValueError("Need an existing checkpoint in order to run PCA")

    isParallel = False
    if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    model = model.to(device)

    pool_size = encoder_dim
    if config['global_params']['pooling'].lower() == 'netvlad':
        pool_size *= int(config['global_params']['num_clusters'])

    print('===> Loading PCA dataset(s)')

    nFeatures = 10000
    if dataset_choice == 'pitts':
        dataset_file_path = join(PATCHNETVLAD_ROOT_DIR, 'dataset_imagenames', 'pitts30k_imageNames_index.txt')
        pca_train_set = PlaceDataset(None, dataset_file_path, dataset_root_dir, None, config['train'])
        pca_train_images = pca_train_set.images
    else:
        raise ValueError('Unknown dataset choice: ' + dataset_choice)

    if nFeatures > len(pca_train_images):
        nFeatures = len(pca_train_images)

    sampler = SubsetRandomSampler(np.random.choice(len(pca_train_images), nFeatures, replace=False))

    data_loader = DataLoader(
        dataset=ImagesFromList(pca_train_images, transform=input_transform()),
        num_workers=threads, batch_size=int(config['train']['cachebatchsize']), shuffle=False,
        pin_memory=cuda,
        sampler=sampler)

    print('===> Do inference to extract features and save them.')

    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')

        dbFeat = np.empty((len(data_loader.sampler), pool_size))
        print('Compute', len(dbFeat), 'features')

        for iteration, (input_data, indices) in enumerate(tqdm(data_loader)):
            input_data = input_data.to(device)
            image_encoding = model.encoder(input_data)

            image_encoding = image_encoding.view(image_encoding.size(0),image_encoding.size(1),-1)
            for l in scaleSpace:
                input_scaled = input_data[:,:,::l,::l]
                image_scaled_encoding = model.encoder(input_scaled)
                image_encoding = torch.cat([image_encoding, image_scaled_encoding.view(image_scaled_encoding.size(0),\
                        image_scaled_encoding.size(1),-1)], dim=2)
            image_encoding=image_encoding.unsqueeze(3)
            vlad_encoding = model.pool(image_encoding)

            out_vectors = vlad_encoding.detach().cpu().numpy()
            # this allows for randomly shuffled inputs
            for idx, out_vector in enumerate(out_vectors):
                dbFeat[iteration * data_loader.batch_size + idx, :] = out_vector

            del input_data, image_encoding, vlad_encoding

    print('===> Compute PCA, takes a while')
    num_pcs = int(config['global_params']['num_pcs'])
    u, lams, mu = pca(dbFeat, num_pcs)

    u = u[:, :num_pcs]
    lams = lams[:num_pcs]

    print('===> Add PCA Whiten')
    u = np.matmul(u, np.diag(np.divide(1., np.sqrt(lams + 1e-9))))
    pca_str = 'WPCA'

    utmu = np.matmul(u.T, mu)

    pca_conv = nn.Conv2d(pool_size, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
    # noinspection PyArgumentList
    pca_conv.weight = nn.Parameter(torch.from_numpy(np.expand_dims(np.expand_dims(u.T, -1), -1)))
    # noinspection PyArgumentList
    pca_conv.bias = nn.Parameter(torch.from_numpy(-utmu))

    model.add_module(pca_str, nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))

    save_path = resume_path.replace(".pth.tar", "_WPCA_" + str(num_pcs) + ".pth.tar")

    torch.save({'num_pcs': num_pcs, 'epoch': start_epoch, 'state_dict': model.state_dict()}, save_path)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    print('Done')