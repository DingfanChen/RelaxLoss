import os
import sys
import time
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms as transforms

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR, '../../../data')
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
from dataset import prepare_texas, prepare_purchase
from utils import BaseTrainer


class NonImageTrainer(BaseTrainer):
    def set_dataloader(self):
        """The function to set the dataset parameters"""
        if self.args.dataset == 'Texas':
            target_train, target_test, shadow_train, shadow_test, pseudo_attack = prepare_texas(self.args.random_seed)
            self.num_classes = 100
        else:
            assert self.args.dataset == 'Purchase'
            target_train, target_test, shadow_train, shadow_test, pseudo_attack = prepare_purchase(self.args.random_seed)
            self.num_classes = 100
        self.target_train = target_train
        self.target_test = target_test
        self.shadow_train = shadow_train
        self.shadow_test = shadow_test
        self.pseudo_attack = pseudo_attack

        ### Set partition
        if self.args.partition == 'target':
            trainset = target_train
            testset = target_test
        elif self.args.partition == 'shadow':
            trainset = shadow_train
            testset = shadow_test

        ## Set dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.train_batchsize, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.test_batchsize, shuffle=False)
        self.trainset = trainset
        self.testset = testset
        self.trainloader = trainloader
        self.testloader = testloader
