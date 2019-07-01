# -*- coding: utf-8 -*-

########################################################################################################
#   Dataloader of the radicoms dataset which is in the type of .npy, for the network to read data
#   Can be ultilize after dicom_transfer.py
#   
#   By Adam Mo
#   2019/4/3
########################################################################################################

from __future__ import print_function, division

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import transform
# from skimage import io, transform
# import matplotlib.pyplot as plt
import pandas as pd
import os

import pdb

from math import sqrt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MVIClassifyLoader(Dataset):
    """All images with labels dataset."""

    def __init__(self, list_file, dir_name='', mode='A', transform=None):
        """
        Args:
            list_file (string): Path to the csv file with annotations.
            dir_name (string): Path for directory of data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_lines = open(list_file).readlines()
        self.dir_name = dir_name
        self.transform = transform
        self.mode = mode
        self.meanstd = {'mean': [-0.003778537057316728, 2.449452987387403e-06, -0.0037714363744210085, 2.3108608870447406e-06, -0.0037930106360891617, 6.663891183478171e-07], 'std': [0.0003240755441649854, 5.220774878867002e-06, 0.00032346655376817564, 4.137437955074534e-06, 0.0003253171964191297, 6.077219391644975e-08]}

        # print(len(self.image_lines))

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        data_line = self.data_lines[idx]

        data_line = data_line.replace('\n', '').replace('\r', '').split(',')
        patientid = int(data_line[0])
        mvi_sta = int(data_line[1])

#         at = transforms.Normalize(self.meanstd['mean'][0], self.meanstd['std'][0])
#         al = transforms.Normalize(self.meanstd['mean'][1], self.meanstd['std'][1])
#         dt = transforms.Normalize(self.meanstd['mean'][2], self.meanstd['std'][2])
#         al = transforms.Normalize(self.meanstd['mean'][3], self.meanstd['std'][3])
#         pt = transforms.Normalize(self.meanstd['mean'][4], self.meanstd['std'][4])
#         pl = transforms.Normalize(self.meanstd['mean'][5], self.meanstd['std'][5])

        # print(data_line[2])
        ap_img = self.__normal_img(np.load(self.dir_name + data_line[2]))
#         ap_lbl = self.__normal_img(np.load(self.dir_name + data_line[3]))
        dp_img = self.__normal_img(np.load(self.dir_name + data_line[4]))
#         dp_lbl = self.__normal_img(np.load(self.dir_name + data_line[5]))
        pvp_img = self.__normal_img(np.load(self.dir_name + data_line[6]))
#         pvp_lbl = self.__normal_img(np.load(self.dir_name + data_line[7]))
        groupfeature = torch.tensor(np.array(data_line[8:]).astype(np.float32))

#        print(ap_img.min(), ap_img.max())

#        sst = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#            transforms.Normalize(-0.003778537057316728, 0.0003240755441649854),  # debug: now only for A
#        ])

#        print(sst(ap_img).min(), sst(ap_img).max())

        # print(patientid, groupfeature.shape)
#        image=transform.resize(image,(128, 128, 16))

        if self.transform:
#                 print('dododo!')
                ap_img = self.transform(ap_img)[np.newaxis, :]
#                 ap_lbl = self.transform(ap_lbl)[np.newaxis, :]
                dp_img = self.transform(dp_img)[np.newaxis, :]
#                 dp_lbl = self.transform(dp_lbl)[np.newaxis, :]
                pvp_img = self.transform(pvp_img)[np.newaxis, :]
#                 pvp_lbl = self.transform(pvp_lbl)[np.newaxis, :]

#         pdb.set_trace()
        if self.mode == 'ALL':
            data_dict = {
                'id': patientid, 'mvi': mvi_sta,
                'apimg': ap_img, 'aplbl': 0,
                'dpimg': dp_img, 'dplbl': 0,
                'pvpimg': pvp_img, 'pvplbl': 0,
                'groupfeature': groupfeature
            }
        elif self.mode == 'A':
            data_dict = {
                'id': patientid, 'mvi': mvi_sta,
                'img': ap_img, 'lbl': 0
            }
        elif self.mode == 'D':
            data_dict = {
                'id': patientid, 'mvi': mvi_sta,
                'img': dp_img, 'lbl': 0
            }
        elif self.mode == 'P':
            data_dict = {
                'id': patientid, 'mvi': mvi_sta,
                'img': pvp_img, 'lbl': 0
            }
        elif self.mode == 'G':
            data_dict = {
                'id': patientid, 'mvi': mvi_sta,
                'img': pvp_img, 'lbl': 0,
                'groupfeature': groupfeature
            }
        else:
            data_dict = {}

        return data_dict

    def __normal_img(self, img):
#         return img
        image = (img - img.min()) / (img.max() - img.min())
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image
    
    def compute_mean_std(self):
        total = len(self.data_lines) * 255 * 255 * 16

        sixmean = [0, 0, 0, 0, 0, 0]
        sixstd = [0, 0, 0, 0, 0, 0]

        # arr_stn = 0
        # arr_mean = 0

        for data_line in self.data_lines:
            data_line = data_line.replace('\n', '').replace('\r', '').split(',')

            ap_img = np.load(self.dir_name + data_line[2])
            ap_lbl = np.load(self.dir_name + data_line[3])
            dp_img = np.load(self.dir_name + data_line[4])
            dp_lbl = np.load(self.dir_name + data_line[5])
            pvp_img = np.load(self.dir_name + data_line[6])
            pvp_lbl = np.load(self.dir_name + data_line[7])

            imglist = [ap_img, ap_lbl, dp_img, dp_lbl, pvp_img, pvp_lbl]

            for imgi in range(len(imglist)):
                for i in imglist[imgi]:
                    sixmean[imgi] += float(i.sum()) / total

                for i in imglist[imgi]:
                    sixstd[imgi] += (((i - sixmean[imgi]) ** 2).sum())
                sixstd[imgi] = sqrt(sixstd[imgi] / total)

        # [-0.003778537057316728, 2.449452987387403e-06, -0.0037714363744210085, 2.3108608870447406e-06, -0.0037930106360891617, 6.663891183478171e-07], [0.0003240755441649854, 5.220774878867002e-06, 0.00032346655376817564, 4.137437955074534e-06, 0.0003253171964191297, 6.077219391644975e-08]
        return sixmean, sixstd


if __name__ == '__main__':
    file_nail = ''
    batch_size = 5
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(-0.003778537057316728, 0.0003240755441649854),     # debug: now only for A
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(-0.003778537057316728, 0.0003240755441649854),     # debug: now only for A
    ])

    readpath = 'full_datafile.csv'

    trainset = MVIClassifyLoader(readpath, transform=transform_train, dir_name='', mode='A')
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=1)

    print(trainset.compute_mean_std())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # testset = MVIClassifyLoader(os.path.join(readpath, 'test_datafile' + file_nail + '.csv'), transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=1)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # print(trainloader)

    import matplotlib.pyplot as plt

    for batch_idx, datai in enumerate(trainloader):
    #     plt.imshow(datai['img'][0][0][11].numpy())
    #     plt.show()
        input()

