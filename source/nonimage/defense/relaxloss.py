import os
import sys
import argparse
import random
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
SAVE_ROOT = os.path.join(FILE_DIR, '../../../results/%s/relaxloss')
import models as models
from base import NonImageTrainer
from utils import mkdir, str2bool, write_yaml, load_yaml, adjust_learning_rate, \
    AverageMeter, Bar, accuracy, one_hot_embedding, CrossEntropy_soft


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
    parser.add_argument('--num_epochs', '-ep', type=int, help='number of epochs')
    parser.add_argument('--alpha', type=float, help='the desired loss level')
    parser.add_argument('--upper', type=float, help='upper confidence level')
    parser.add_argument('--partition', type=str, choices=['target', 'shadow'], help='training partition')
    parser.add_argument('--if_resume', type=str2bool, help='If resume from checkpoint')
    parser.add_argument('--if_onlyeval', type=str2bool, help='If only evaluate the pre-trained model')
    return parser


def check_args(parser):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
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
        specific_configs = load_yaml(FILE_DIR + '/configs/%s.yml' % args.dataset)
        parser.set_defaults(**default_configs)
        parser.set_defaults(**specific_configs)
        args = parser.parse_args()
        write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


#############################################################################################################
# helper functions
############################################################################################################
class Trainer(NonImageTrainer):
    def set_criterion(self):
        """Set up the relaxloss training criterion"""
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy_soft = CrossEntropy_soft
        self.crossentropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.alpha = self.args.alpha
        self.upper = self.args.upper

    def train(self, model, optimizer, epoch):
        model.train()

        losses = AverageMeter()
        losses_ce = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            dataload_time.update(time.time() - time_stamp)
            outputs = model(inputs)
            loss_ce_full = self.crossentropy_noreduce(outputs, targets)
            loss_ce = torch.mean(loss_ce_full)

            if epoch % 2 == 0:  # gradient ascent/ normal gradient descent
                loss = (loss_ce - self.alpha).abs()
            else:
                if loss_ce > self.alpha:  # normal gradient descent
                    loss = loss_ce
                else:  # posterior flattening
                    confidence_target = self.softmax(outputs)[torch.arange(targets.size(0)), targets]
                    confidence_target = torch.clamp(confidence_target, min=0., max=self.upper)
                    confidence_else = (1.0 - confidence_target) / (self.num_classes - 1)
                    onehot = one_hot_embedding(targets, num_classes=self.num_classes)
                    soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.num_classes) \
                                   + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, self.num_classes)
                    loss = self.crossentropy_soft(outputs, soft_targets)

            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            losses_ce.update(loss_ce.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            ### Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Record the total time for processing the batch
            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

            ### Progress bar
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.trainloader),
                data=dataload_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()

        bar.finish()
        return (losses_ce.avg, top1.avg, top5.avg)


#############################################################################################################
# main function
#############################################################################################################
def main():
    ### config
    args, save_dir = check_args(parse_arguments())

    ### Set up trainer
    trainer = Trainer(args, save_dir)
    model = models.__dict__[args.dataset]()
    model = torch.nn.DataParallel(model)
    model = model.to(trainer.device)
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
        train_ce, train_acc, train_acc5 = trainer.train(model, optimizer, epoch)
        test_ce, test_acc, test_acc5 = trainer.test(model)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logger.append([lr, train_ce, test_ce, train_acc, test_acc, train_acc5, test_acc5])
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
