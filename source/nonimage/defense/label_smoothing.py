import os
import sys
import argparse
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
SAVE_ROOT = os.path.join(FILE_DIR, '../../../results/%s/label_smoothing')
import models as models
from base import NonImageTrainer
from utils import mkdir, str2bool, load_yaml, write_yaml, adjust_learning_rate


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, help='experiment name, used for set up save_dir')
    parser.add_argument('--dataset', type=str, choices=['Texas', 'Purchase'], help='dataset name')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--schedule_milestone', type=int, nargs='+', help='when to decrease the learning rate')
    parser.add_argument('--gamma', type=float, help='learning rate step gamma')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--train_batchsize', type=int, help='training batch size')
    parser.add_argument('--test_batchsize', type=int, help='testing batch size')
    parser.add_argument('--num_workers', type=int, help='number of workers')
    parser.add_argument('--num_epochs', '-ep', type=int, help='number of epochs')
    parser.add_argument('--partition', type=str, choices=['target', 'shadow'], help='training partition')
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon for label smoothing')
    parser.add_argument('--if_resume', type=str2bool, help='If resume from checkpoint')
    parser.add_argument('--if_onlyeval', type=str2bool, help='If only evaluate the pre-trained model')
    return parser


def check_args(parser):
    """Check and store the arguments as well as set up the save_dir"""
    ## set up save_dir
    args = parser.parse_args()
    save_dir = os.path.join(SAVE_ROOT % args.dataset, args.exp_name)
    if args.partition == 'shadow':
        save_dir = os.path.join(save_dir, 'shadow')
    mkdir(save_dir)
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10 ^ 5)

    ## load configs and store the parameters
    if args.if_onlyeval:
        preload_configs = load_yaml(os.path.join(save_dir, 'params.yml'))
        parser.set_defaults(**preload_configs)
        args = parser.parse_args()
    else:
        default_configs = load_yaml(FILE_DIR + '/configs/default.yml')
        parser.set_defaults(**default_configs)
        args = parser.parse_args()
        write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


################################################################################
# helper functions
################################################################################
class LabelSmoothingCrossEntropy(nn.Module):
    '''
    epsilon * 1/K \sum_{i=1}^K -\log p(yi|x) + (1-epsilon)* \sum_{i=1}^K -ti \log p(yi|x)
    =  epsilon * KL (u || p(y|x)) + const + (1-epsilon)* CE(p(y|x), target)
    '''

    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)

    @staticmethod
    def linear_combination(x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class Trainer(NonImageTrainer):
    def set_criterion(self):
        """Set the criterion to be label smoothing"""
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.criterion = LabelSmoothingCrossEntropy(epsilon=self.args.epsilon)
        self.crossentropy = nn.CrossEntropyLoss()


#############################################################################################################
# main function
#############################################################################################################
def main():
    args, save_dir = check_args(parse_arguments())

    ### Set up trainer and model
    trainer = Trainer(args, save_dir)
    model = models.__dict__[args.dataset]()
    model = torch.nn.DataParallel(model)
    model.to(trainer.device)
    torch.backends.cudnn.benchmark = True
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ### Load checkpoint
    start_epoch = 0
    logger = trainer.logger
    if args.if_resume or args.if_onlyeval:
        try:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pkl'))
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            logger = Logger(os.path.join(save_dir, 'log.txt'), title=title, resume=True)
        except:
            pass

    if args.if_onlyeval:
        print('\nEvaluation only')
        test_loss, test_acc, test_acc5 = trainer.test(model)
        print(' Test Loss: %.8f, Test Acc(top1): %.2f, Test Acc(top5): %.2f' % (test_loss, test_acc, test_acc5))
        return

    ### Training
    for epoch in range(start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args.gamma, args.schedule_milestone)
        train_loss, train_acc, train_acc5 = trainer.train(model, optimizer)
        test_loss, test_acc, test_acc5 = trainer.test(model)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logger.append([lr, train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5])
        print('Epoch %d, Train acc: %f, Test acc: %f, lr: %f' % (epoch, train_acc, test_acc, lr))

        ### Save checkpoint
        save_dict = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'opt_state_dict': optimizer.state_dict()}
        torch.save(save_dict, os.path.join(save_dir, 'checkpoint.pkl'))
        torch.save(model, os.path.join(save_dir, 'model.pt'))

        ### Visualize
        trainer.logger_plot()


if __name__ == '__main__':
    main()
