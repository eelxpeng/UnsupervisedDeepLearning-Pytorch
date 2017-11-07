'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from PIL import Image

import logging

def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)

class Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def readData(filename, data_num, vocab_size, randgen=None):
    dataX = torch.FloatTensor(data_num, vocab_size) *0
    dataY = torch.LongTensor(data_num) *0
    if randgen != None:
        print('Reading data with permutation from %s\n' % filename)
        idx = randgen.permutation(data_num)
    else:
        print('Reading data without permutation from %s\n' % filename)
        idx = range(data_num)

    infile = open(filename)
    count = 0
    for line in infile:
        line = line.strip('\n').split(',')
        dataY[ idx[count] ] = int(line[0])

        entry_list = [[int(listed_pair[0]), int(listed_pair[1])] for listed_pair in [pair.split(':') for pair in line[1:]]]
        entry_tensor = torch.LongTensor(entry_list)
        if len(entry_list)!=0:
            dataX[ idx[count] ][entry_tensor[:,0]] = entry_tensor[:,1].type(torch.FloatTensor)
        count += 1
    infile.close()
    assert count == data_num, (count, data_num)
    print('Read %d\t datacases\t Done!\n' % count)

    return dataX, dataY