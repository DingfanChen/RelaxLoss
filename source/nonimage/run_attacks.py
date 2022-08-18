import os
import sys
import argparse
import torch.utils.data as data

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
DATA_ROOT = os.path.join(FILE_DIR, '../../data')
from dataset import prepare_texas, prepare_purchase
import models as models
from utils import mkdir, load_yaml, write_yaml, BaseAttacker


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
        if self.args.dataset == 'Texas':
            target_train, target_test, shadow_train, shadow_test, pseudo_attack = prepare_texas(self.args.random_seed)
            self.num_classes = 100
        else:
            assert self.args.dataset == 'Purchase'
            target_train, target_test, shadow_train, shadow_test, pseudo_attack = prepare_purchase(
                self.args.random_seed)
            self.num_classes = 100
        self.target_train = target_train
        self.target_test = target_test
        self.shadow_train = shadow_train
        self.shadow_test = shadow_test
        self.pseudo_attack = pseudo_attack

        ## Set dataloader
        self.target_trainloader = data.DataLoader(target_train, batch_size=self.args.test_batchsize, shuffle=False)
        self.target_testloader = data.DataLoader(target_test, batch_size=self.args.test_batchsize, shuffle=False)
        self.shadow_trainloader = data.DataLoader(shadow_train, batch_size=self.args.test_batchsize, shuffle=False)
        self.shadow_testloader = data.DataLoader(shadow_test, batch_size=self.args.test_batchsize, shuffle=False)
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
