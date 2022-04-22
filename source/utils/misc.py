''' Common helper functions
'''
import os
import sys
import errno
import pickle
import argparse
import yaml
import csv
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

__all__ = ['mkdir', 'savefig', 'plot_hist', 'str2bool', 'get_all_losses',
           'load_yaml', 'write_yaml', 'savepickle', 'unpickle', 'write_csv', 'Partition']


def mkdir(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi, format='png')


def plot_hist(values, names, save_file):
    plt.figure()
    bins = np.histogram(np.hstack(values), bins=50)[1]
    for val, name in zip(values, names):
        plt.hist(val, bins=bins, alpha=0.5, label=name)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_file, dpi=150, format='png')
    plt.close()


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data


def write_yaml(data, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(data, f)


def savepickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_csv(file_path, entry_title, entry_data, index_names=None):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col='name')
        df[entry_title] = entry_data
    else:
        df = pd.DataFrame({'name': index_names,
                           entry_title: entry_data}).set_index('name')
    df.to_csv(file_path)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_all_losses(dataloader, model, criterion, device):
    model.eval()

    losses = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            ### Forward
            outputs = model(inputs)

            ### Evaluate
            loss = criterion(outputs, targets)
            losses.append(loss.cpu().numpy())

    losses = np.concatenate(losses)
    return losses


class Parser(dict):
    def __init__(self, *args):
        super(Parser, self).__init__()
        for d in args:
            if isinstance(d, argparse.Namespace):
                d = vars(d)
            for k, v in d.items():
                if k in self.keys():
                    if v is not None and self[k] is not None:
                        print(f'{k} is already defined to be {self[k]}. Overwriting to {v}')
                    if v is None:
                        continue
                k = k.replace('-', '_')
                self[k] = v

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, val):
        self[key] = val


class Partition():
    def __init__(self,
                 dataset_size,
                 indices,
                 target_train_size=None,
                 target_test_size=None,
                 shadow_train_size=None,
                 shadow_test_size=None,
                 pseudo_test_size=None
                 ):
        ## Config
        self.dataset_size = dataset_size
        self.indices = indices
        self.num_slices = 5
        self.slice_size = self.dataset_size // self.num_slices

        ## Size of each subset
        self.target_train_size = target_train_size if target_train_size is not None else self.slice_size
        self.target_test_size = target_test_size if target_test_size is not None else self.slice_size
        self.shadow_train_size = shadow_train_size if shadow_train_size is not None else self.slice_size
        self.shadow_test_size = shadow_test_size if shadow_test_size is not None else self.slice_size
        self.pseudo_test_size = pseudo_test_size if pseudo_test_size is not None else self.slice_size

        ## Construct indices subsets
        self.names = ['target_train', 'target_test',
                      'shadow_train', 'shadow_test',
                      'pseudo_test']
        self.size_list = [self.target_train_size, self.target_test_size,
                          self.shadow_train_size, self.shadow_test_size,
                          self.pseudo_test_size]
        self.indices_dict = {}
        start_idx = 0
        for i in range(len(self.names)):
            self.indices_dict[self.names[i]] = self.indices[start_idx:start_idx + self.size_list[i]]
            start_idx += self.size_list[i]

    def get_target_indices(self):
        return self.indices_dict['target_train'], self.indices_dict['target_test']

    def get_shadow_indices(self):
        return self.indices_dict['shadow_train'], self.indices_dict['shadow_test']

    def get_pseudoattack_train_indices(self):
        return self.indices_dict['target_train'], self.indices_dict['pseudo_test']

    def get_attack_eval_indices(self):
        return self.indices_dict['target_train'], self.indices_dict['target_test']
