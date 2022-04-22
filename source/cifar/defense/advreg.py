import os
import sys
import argparse
import random
import shutil
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
SAVE_ROOT = os.path.join(FILE_DIR, '../../../results/%s/%s/advreg')
import models as models
from base import CIFARTrainer
from utils import mkdir, str2bool, write_yaml, load_yaml, adjust_learning_rate, \
    Logger, AverageMeter, Bar, accuracy, accuracy_binary, savefig


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
    parser.add_argument('--if_resume', type=str2bool, help='If resume from checkpoint')
    parser.add_argument('--if_data_augmentation', '-aug', type=str2bool, help='If use data augmentation')
    parser.add_argument('--if_onlyeval', type=str2bool, help='If only evaluate the pre-trained model')
    parser.add_argument('--lr_attack', type=float, default=0.001, help='attack learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for the adversarial loss')
    parser.add_argument('--attack_steps', type=int, default=5, help='attacker update steps per one target step')
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


#############################################################################################################
# helper functions
#############################################################################################################
class Attack(nn.Module):
    def __init__(self, input_dim, num_classes=1, hiddens=[100]):
        super(Attack, self).__init__()
        self.layers = []
        for i in range(len(hiddens)):
            if i == 0:
                layer = nn.Linear(input_dim, hiddens[i])
            else:
                layer = nn.Linear(hiddens[i - 1], hiddens[i])
            self.layers.append(layer)
        self.last_layer = nn.Linear(hiddens[-1], num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        output = x
        for layer in self.layers:
            output = self.relu(layer(output))
        output = self.last_layer(output)
        output = self.sigmoid(output)
        return output


class Trainer(CIFARTrainer):
    def __init__(self, the_args, save_dir):
        super(Trainer, self).__init__(the_args, save_dir)
        self.set_specific()

    def set_logger(self):
        """Set up logger"""
        title = self.args.dataset + '_' + self.args.model
        self.start_epoch = 0
        logger = Logger(os.path.join(self.save_dir, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Train Acc 5', 'Val Acc 5',
                          'Attack Train Loss', 'Attack Train Acc', 'Attack Test Acc'])
        self.logger = logger

    def set_specific(self):
        """ Pseudo attack training setting"""
        ### Criterion
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy = nn.CrossEntropyLoss()
        self.criterion = self.crossentropy
        self.attack_criterion = nn.MSELoss()
        self.alpha = self.args.alpha

        ### Data
        partition = self.partition
        attack_member_idx, attack_nonmember_idx = partition.get_pseudoattack_train_indices()
        attack_nonmember_data = self.dataset(root=self.data_root, indices=attack_nonmember_idx,
                                             download=True, transform=self.transform_train)
        self.attack_member_loader = self.trainloader
        self.attack_nonmember_loader = torch.utils.data.DataLoader(attack_nonmember_data,
                                                                   batch_size=self.args.train_batchsize,
                                                                   shuffle=True, num_workers=self.args.num_workers)

    def attack_input_transform(self, x, y):
        """Transform the input to attack model"""
        out_x = x
        out_x, _ = torch.sort(out_x, dim=1)
        one_hot = torch.from_numpy((np.zeros((y.size(0), self.num_classes)) - 1)).cuda().type(
            torch.cuda.FloatTensor)
        out_y = one_hot.scatter_(1, y.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)
        return out_x, out_y

    def logger_plot(self):
        """ Visualize the training progress"""
        self.logger.plot(['Train Loss', 'Val Loss'])
        savefig(os.path.join(self.save_dir, 'loss.png'))

        self.logger.plot(['Train Acc', 'Val Acc'])
        savefig(os.path.join(self.save_dir, 'acc.png'))

        self.logger.plot(['Attack Train Acc', 'Attack Test Acc'])
        savefig(os.path.join(self.save_dir, 'attack_acc.png'))

    def train_privately(self, target_model, attack_model, target_optimizer, num_batches=10000):
        """ Target model should minimize the CE while making the attacker's output close to 0.5"""
        target_model.train()
        attack_model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        max_batches = min(num_batches, len(self.attack_member_loader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx >= num_batches:
                break
            data_time.update(time.time() - end)

            ### Forward and compute loss
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = target_model(inputs)
            inference_input_x, inference_input_y = self.attack_input_transform(outputs, targets)
            inference_output = attack_model(inference_input_x, inference_input_y)
            loss = self.criterion(outputs, targets) + ((self.alpha) * (torch.mean((inference_output)) - 0.5))

            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size()[0])
            top1.update(prec1, inputs.size()[0])
            top5.update(prec5, inputs.size()[0])

            ### Optimization
            target_optimizer.zero_grad()
            loss.backward()
            target_optimizer.step()

            ### Record the total time for processing the batch
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 100 == 0:
                print(
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                        batch=batch_idx,
                        size=max_batches,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                    ))
        return (losses.avg, top1.avg, top5.avg)

    def train_attack(self, target_model, attack_model, attack_optimizer, num_batches=100000):
        """ Train pseudo attacker"""
        target_model.eval()
        attack_model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        max_batches = min(num_batches, len(self.attack_member_loader))
        for batch_idx, (member, nonmember) in enumerate(zip(self.attack_member_loader, self.attack_nonmember_loader)):
            if batch_idx >= num_batches:
                break
            data_time.update(time.time() - end)

            inputs_member, targets_member = member
            inputs_nonmember, targets_nonmember = nonmember
            inputs_member, targets_member = inputs_member.to(self.device), targets_member.to(self.device)
            inputs_nonmember, targets_nonmember = inputs_nonmember.to(self.device), targets_nonmember.to(self.device)
            outputs_member_x, outputs_member_y = self.attack_input_transform(target_model(inputs_member),
                                                                             targets_member)
            outputs_nonmember_x, outputs_nonmember_y = self.attack_input_transform(target_model(inputs_nonmember),
                                                                                   targets_nonmember)
            attack_input_x = torch.cat((outputs_member_x, outputs_nonmember_x))
            attack_input_y = torch.cat((outputs_member_y, outputs_nonmember_y))
            attack_labels = np.zeros((inputs_member.size()[0] + inputs_nonmember.size()[0]))
            attack_labels[:inputs_member.size()[0]] = 1.  # member=1
            attack_labels[inputs_member.size()[0]:] = 0.  # nonmember=0

            indices = np.arange(len(attack_input_x))
            np.random.shuffle(indices)
            attack_input_x = attack_input_x[indices]
            attack_input_y = attack_input_y[indices]
            attack_labels = attack_labels[indices]
            is_member_labels = torch.from_numpy(attack_labels).type(torch.FloatTensor).to(self.device)
            attack_output = attack_model(attack_input_x, attack_input_y).view(-1)

            ### Record accuracy and loss
            loss_attack = self.attack_criterion(attack_output, is_member_labels)
            prec1 = accuracy_binary(attack_output.data, is_member_labels.data)
            losses.update(loss_attack.item(), len(attack_output))
            top1.update(prec1.item(), len(attack_output))

            ### Optimization
            attack_optimizer.zero_grad()
            loss_attack.backward()
            attack_optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 100 == 0:
                print(
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                        batch=batch_idx,
                        size=max_batches,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                    ))
        return (losses.avg, top1.avg)

    def test_attack(self, target_model, attack_model):
        """ Test pseudo attack model"""
        target_model.eval()
        attack_model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()
        bar = Bar('Processing', max=len(self.attack_member_loader))

        for batch_idx, (member, nonmember) in enumerate(zip(self.attack_member_loader, self.attack_nonmember_loader)):
            inputs_member, targets_member = member
            inputs_nonmember, targets_nonmember = nonmember
            data_time.update(time.time() - end)
            inputs_member, targets_member = inputs_member.to(self.device), targets_member.to(self.device)
            inputs_nonmember, targets_nonmember = inputs_nonmember.to(self.device), targets_nonmember.to(self.device)
            outputs_member_x, outputs_member_y = self.attack_input_transform(target_model(inputs_member),
                                                                             targets_member)
            outputs_nonmember_x, outputs_nonmember_y = self.attack_input_transform(target_model(inputs_nonmember),
                                                                                   targets_nonmember)
            attack_input_x = torch.cat((outputs_member_x, outputs_nonmember_x))
            attack_input_y = torch.cat((outputs_member_y, outputs_nonmember_y))
            attack_labels = np.zeros((inputs_member.size()[0] + inputs_nonmember.size()[0]))
            attack_labels[:inputs_member.size()[0]] = 1.  # member=1
            attack_labels[inputs_member.size()[0]:] = 0.  # nonmember=0
            is_member_labels = torch.from_numpy(attack_labels).type(torch.FloatTensor).to(self.device)
            attack_output = attack_model(attack_input_x, attack_input_y).view(-1)

            loss_attack = self.attack_criterion(attack_output, is_member_labels)
            prec1 = accuracy_binary(attack_output.data, is_member_labels.data)
            losses.update(loss_attack.item(), len(attack_output))
            top1.update(prec1.item(), len(attack_output))

            batch_time.update(time.time() - end)
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                batch=batch_idx + 1,
                size=len(self.attack_member_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)


########################################################################
### main function
########################################################################
def main():
    ### config
    args, save_dir = check_args(parse_arguments())

    ### Set up trainer & target model
    trainer = Trainer(args, save_dir)
    model = models.__dict__[args.model](num_classes=trainer.num_classes)
    model = torch.nn.DataParallel(model)
    model = model.to(trainer.device)
    torch.backends.cudnn.benchmark = True
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ### Set up attack model
    attack_model = Attack(input_dim=trainer.num_classes)
    attack_model = torch.nn.DataParallel(attack_model)
    attack_model = attack_model.to(trainer.device)
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=args.lr_attack)

    ### Set up logger
    start_epoch = 0
    logger = trainer.logger
    if args.if_resume or args.if_onlyeval:
        try:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pkl'))
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
            attack_optimizer.load_state_dict(checkpoint['attack_opt_state_dict'])
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

        if epoch < 3:
            train_loss, train_acc, train_acc5 = trainer.train(model, optimizer)
            attack_loss, attack_acc = 0, 0
            for i in range(args.attack_steps):
                attack_loss, attack_acc = trainer.train_attack(model, attack_model, attack_optimizer)
            test_loss, test_acc, test_acc5 = trainer.test(model)
            attack_test_loss, attack_test_acc = trainer.test_attack(model, attack_model)

        else:
            attack_loss, attack_acc = trainer.train_attack(model, attack_model, attack_optimizer)
            train_loss, train_acc, train_acc5 = trainer.train_privately(model, attack_model, optimizer)
            test_loss, test_acc, test_acc5 = trainer.test(model)
            attack_test_loss, attack_test_acc = trainer.test_attack(model, attack_model)

        logger.append([train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5,
                       attack_loss, attack_acc, attack_test_acc])

        ### Save checkpoint
        save_dict = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'attack_model_state_dict': attack_model.state_dict(),
                     'opt_state_dict': optimizer.state_dict(),
                     'attack_opt_state_dict': attack_optimizer.state_dict()}
        torch.save(save_dict, os.path.join(save_dir, 'checkpoint.pkl'))
        torch.save(model, os.path.join(save_dir, 'model.pt'))

        ### Visualize
        trainer.logger_plot()


if __name__ == '__main__':
    main()
