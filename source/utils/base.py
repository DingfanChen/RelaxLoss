import os
import sys
import time
import torch
import torch.nn as nn
import random
import numpy as np
import abc

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR, '../../data')
from .misc import Partition, get_all_losses, savefig
from .logger import AverageMeter, Logger
from .eval import accuracy, accuracy_binary
from progress.bar import Bar as Bar

__all__ = ['BaseTrainer']


class BaseTrainer(object):
    """The class that contains the code for base trainer class."""

    def __init__(self, the_args, save_dir):
        """The function to initialize this class."""
        self.args = the_args
        self.save_dir = save_dir
        self.data_root = DATA_ROOT
        self.set_cuda_device()
        self.set_seed()
        self.set_dataloader()
        self.set_logger()
        self.set_criterion()

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if hasattr(self.args, 'num_workers') and self.args.num_workers >= 1:
            torch.multiprocessing.set_start_method('spawn')
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def set_seed(self):
        """Set random seed"""
        random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.args.random_seed)

    @abc.abstractmethod
    def set_dataloader(self):
        """The function to set the dataset parameters"""
        self.dataset = None
        self.num_classes = None
        self.dataset_size = None
        self.transform_train = None
        self.transform_test = None

        self.partition = None
        self.trainset_idx = None
        self.testset_idx = None

        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None

    def set_logger(self):
        """Set up logger"""
        title = self.args.dataset
        self.start_epoch = 0
        logger = Logger(os.path.join(self.save_dir, 'log.txt'), title=title)
        logger.set_names(['LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Train Acc 5', 'Val Acc 5'])
        self.logger = logger

    def set_criterion(self):
        """Set up criterion"""
        self.criterion = nn.CrossEntropyLoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')

    def train(self, model, optimizer, *args):
        """Train"""
        model.train()
        criterion = self.criterion
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
            loss = criterion(outputs, targets)

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

    def test(self, model):
        """Test"""
        model.eval()
        criterion = self.crossentropy
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        bar = Bar('Processing', max=len(self.testloader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                ### Record the data loading time
                dataload_time.update(time.time() - time_stamp)

                ### Forward
                outputs = model(inputs)

                ### Evaluate
                loss = criterion(outputs, targets)
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                ### Record the total time for processing the batch
                batch_time.update(time.time() - time_stamp)
                time_stamp = time.time()

                ### Progress bar
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(self.testloader),
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

    def get_loss_distributions(self, model):
        """ Obtain the member and nonmember loss distributions"""
        train_losses = get_all_losses(self.trainloader, model, self.crossentropy_noreduce, self.device)
        test_losses = get_all_losses(self.testloader, model, self.crossentropy_noreduce, self.device)
        return train_losses, test_losses

    def logger_plot(self):
        """ Visualize the training progress"""
        self.logger.plot(['Train Loss', 'Val Loss'])
        savefig(os.path.join(self.save_dir, 'loss.png'))

        self.logger.plot(['Train Acc', 'Val Acc'])
        savefig(os.path.join(self.save_dir, 'acc.png'))
