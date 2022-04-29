import os
import sys
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
DATA_ROOT = os.path.join(FILE_DIR, '../../data')
from dataset import CIFAR10, CIFAR100
import models as models
from utils import mkdir, load_yaml, write_yaml, BaseAttacker, Partition


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--target_path', '-target', type=str, help='path to the target model')
    parser.add_argument('--shadow_path', '-shadow', type=str, help='path to the shadow model')
    return parser


def check_args(parser):
    '''check and store the arguments as well as set up the save_dir'''
    ## set up save_dir
    args = parser.parse_args()
    save_dir = os.path.join(args.target_path, 'attack')
    mkdir(save_dir)
    mkdir(os.path.join(args.shadow_path, 'attack'))

    ## load configs and store the parameters
    preload_configs = load_yaml(os.path.join(args.target_path, 'params.yml'))
    parser.set_defaults(**preload_configs)
    args = parser.parse_args()
    write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))
    return args, save_dir


#############################################################################################################
# helper functions
#############################################################################################################
class Attacker(BaseAttacker):
    def set_dataloader(self):
        """The function to set the dataset parameters"""
        self.data_root = DATA_ROOT
        if self.args.dataset == 'CIFAR10':
            self.dataset = CIFAR10
            self.num_classes = 10
            self.dataset_size = 60000
        elif self.args.dataset == 'CIFAR100':
            self.dataset = CIFAR100
            self.num_classes = 100
            self.dataset_size = 60000
        transform_train = transform_test = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010))])
        self.transform_train = transform_train
        self.transform_test = transform_test

        ## Set the partition and datloader
        indices = np.load(os.path.join(self.args.target_path, 'full_idx.npy'))
        if os.path.exists(os.path.join(self.args.shadow_path, 'full_idx.npy')):
            shadow_indices = np.load(os.path.join(self.args.shadow_path, 'full_idx.npy'))
            assert np.array_equiv(indices, shadow_indices)
        self.partition = Partition(dataset_size=self.dataset_size, indices=indices)
        target_train_idx, target_test_idx = self.partition.get_target_indices()
        shadow_train_idx, shadow_test_idx = self.partition.get_shadow_indices()

        target_trainset = self.dataset(root=self.data_root, indices=target_train_idx,
                                       download=True, transform=self.transform_train)
        target_testset = self.dataset(root=self.data_root, indices=target_test_idx,
                                      download=True, transform=self.transform_test)
        shadow_trainset = self.dataset(root=self.data_root, indices=shadow_train_idx,
                                       download=True, transform=self.transform_train)
        shadow_testset = self.dataset(root=self.data_root, indices=shadow_test_idx,
                                      download=True, transform=self.transform_test)
        self.target_trainloader = torch.utils.data.DataLoader(target_trainset, batch_size=self.args.test_batchsize, shuffle=False)
        self.target_testloader = torch.utils.data.DataLoader(target_testset, batch_size=self.args.test_batchsize, shuffle=False)
        self.shadow_trainloader = torch.utils.data.DataLoader(shadow_trainset, batch_size=self.args.test_batchsize, shuffle=False)
        self.shadow_testloader = torch.utils.data.DataLoader(shadow_testset, batch_size=self.args.test_batchsize, shuffle=False)
        self.loader_dict = {'s_pos': self.shadow_trainloader, 's_neg': self.shadow_testloader,
                            't_pos': self.target_trainloader, 't_neg': self.target_testloader}


#############################################################################################################
# main function
#############################################################################################################
def main():
    args, save_dir = check_args(parse_arguments())
    attacker = Attacker(args, save_dir)
    attacker.run_blackbox_attacks()
    attacker.run_whitebox_attacks()
    attacker.save_results()


if __name__ == '__main__':
    main()
