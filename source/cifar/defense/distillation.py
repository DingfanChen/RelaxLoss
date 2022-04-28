import os
import sys
import argparse
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
SAVE_ROOT = os.path.join(FILE_DIR, '../../../results/%s/%s/distillation')
import models as models
from base import CIFARTrainer
from utils import mkdir, str2bool, write_yaml, load_yaml, adjust_learning_rate, AverageMeter, Bar, plot_hist, accuracy


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, help='experiment name, used for set up save_dir')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], help='dataset name')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--model', type=str, help='model architecture')
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
    parser.add_argument('--teacher_path', '-teacher', type=str, help='path for the teacher model')
    parser.add_argument('--temperature', type=float, default=10, help='temperature for distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for KL term')
    parser.add_argument('--if_resume', type=str2bool, help='If resume from checkpoint')
    parser.add_argument('--if_data_augmentation', '-aug', type=str2bool, help='If use data augmentation')
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
    save_dir = os.path.join(SAVE_ROOT % (args.dataset, args.model), args.exp_name)
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
class Trainer(CIFARTrainer):
    def __init__(self, the_args, save_dir):
        super(Trainer, self).__init__(the_args, save_dir)
        self.load_teacher()

    def set_criterion(self):
        self.crossentropy = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.criterion = DistillationLoss(alpha=self.args.alpha, temperature=self.args.temperature,
                                          criterion=nn.CrossEntropyLoss())

    def load_teacher(self):
        teacher_path = os.path.join(self.args.teacher_path, 'model.pt')
        teacher_model = torch.load(teacher_path)
        self.teacher_model = teacher_model.to(self.device)
        print('Loading teacher model from ', teacher_path)

    def add_distill_loader(self):
        """Adding teacher prediction to the dataloader"""
        dataloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batchsize,
                                                 shuffle=False, num_workers=self.args.num_workers)

        self.teacher_model.eval()
        teacher_outputs = []
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                teacher_output = self.teacher_model(inputs).cpu().numpy()
            teacher_outputs.append(teacher_output)
        teacher_outputs = np.concatenate(teacher_outputs)
        distill_dataset = DistillDataset(self.trainset, teacher_outputs)
        self.distillset = distill_dataset
        self.distillloader = torch.utils.data.DataLoader(self.distillset, batch_size=self.args.train_batchsize,
                                                         shuffle=True, num_workers=self.args.num_workers)

    def train_dynamic(self, model, optimizer):
        model.train()
        self.teacher_model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ### Record the data loading time
            dataload_time.update(time.time() - time_stamp)

            ### Output
            outputs = model(inputs)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
            loss = self.criterion(outputs, targets, teacher_outputs)

            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
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
        return (losses.avg, top1.avg, top5.avg)

    def train(self, model, optimizer):
        model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets, teacher_outputs) in enumerate(self.distillloader):
            inputs, targets, teacher_outputs = inputs.to(self.device), targets.to(self.device), teacher_outputs.to(
                self.device)

            ### Record the data loading time
            dataload_time.update(time.time() - time_stamp)

            ### Output
            outputs = model(inputs)
            loss = self.criterion(outputs, targets, teacher_outputs)

            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
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
                size=len(self.distillloader),
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
        return (losses.avg, top1.avg, top5.avg)


class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, teacher_outputs):
        super(DistillDataset, self).__init__()
        self.dataset = dataset
        self.data = self.dataset.data
        self.targets = self.dataset.targets
        self.transform = self.dataset.transform
        self.target_transform = self.dataset.target_transform
        self.teacher_outputs = teacher_outputs

    def __getitem__(self, index):
        img, target, teacher_output = self.data[index], self.targets[index], self.teacher_outputs[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, teacher_output

    def __len__(self) -> int:
        return len(self.data)


class DistillationLoss(nn.Module):
    '''
    the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities
    '''

    def __init__(self, alpha, temperature, criterion, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.reduction = reduction
        self.criterion = criterion
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.KL = nn.KLDivLoss(reduction=reduction)

    def forward(self, preds, target, teacher_outputs):
        loss = self.criterion(preds, target)
        KL = self.KL(self.logsoftmax(preds / self.T), self.softmax(teacher_outputs / self.T))
        KD_loss = KL * (self.alpha * self.T * self.T) + loss * (1. - self.alpha)
        return KD_loss


#############################################################################################################
# main function
#############################################################################################################
def main():
    ### config
    args, save_dir = check_args(parse_arguments())

    ### Set up trainer and model
    trainer = Trainer(args, save_dir)
    if not args.if_data_augmentation:
        trainer.add_distill_loader()
    model = models.__dict__[args.model](num_classes=trainer.num_classes)
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
        if args.if_data_augmentation:
            train_loss, train_acc, train_acc5 = trainer.train_dynamic(model, optimizer)
        else:
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
    train_losses, test_losses = trainer.get_loss_distributions(model)
    plot_hist([train_losses, test_losses], ['train', 'val'], os.path.join(save_dir, 'hist_ep%d.png' % epoch))


if __name__ == '__main__':
    main()
