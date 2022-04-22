import os
import numpy as np
import math
import random
import time
import abc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy.stats import norm, kurtosis, skew
from sklearn import metrics
from tqdm import tqdm
from progress.bar import Bar as Bar
from .logger import Logger, AverageMeter
from .eval import accuracy_binary, accuracy, metrics_binary
from .misc import *
from .base import BaseTrainer

__all__ = ['Benchmark', 'Benchmark_Blackbox', 'BaseAttacker']


class Benchmark(object):
    def __init__(self, shadow_train_scores, shadow_test_scores, target_train_scores, target_test_scores):
        self.s_tr_scores = shadow_train_scores
        self.s_te_scores = shadow_test_scores
        self.t_tr_scores = target_train_scores
        self.t_te_scores = target_test_scores
        self.num_methods = len(self.s_tr_scores)

    def load_labels(self, s_tr_labels, s_te_labels, t_tr_labels, t_te_labels, num_classes):
        """Load sample labels"""
        self.num_classes = num_classes
        self.s_tr_labels = s_tr_labels
        self.s_te_labels = s_te_labels
        self.t_tr_labels = t_tr_labels
        self.t_te_labels = t_te_labels

    def _thre_setting(self, tr_values, te_values):
        """Select the best threshold"""
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_thre_perclass(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        """MIA by thresholding per-class feature values """
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels == num], s_te_values[self.s_te_labels == num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(self.t_te_labels) + 0.0))
        info = 'MIA via {n} (pre-class threshold): the attack acc is {acc:.3f}'.format(n=v_name, acc=mem_inf_acc)
        print(info)
        return info, mem_inf_acc

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        """MIA by thresholding overall feature values"""
        t_tr_mem, t_te_non_mem = 0, 0
        thre = self._thre_setting(s_tr_values, s_te_values)
        t_tr_mem += np.sum(t_tr_values >= thre)
        t_te_non_mem += np.sum(t_te_values < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(t_tr_values) + 0.0) + t_te_non_mem / (len(t_te_values) + 0.0))
        info = 'MIA via {n} (general threshold): the attack acc is {acc:.3f}'.format(n=v_name, acc=mem_inf_acc)
        print(info)
        return info, mem_inf_acc

    def _mem_inf_roc(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        """MIA AUC given the feature values (no need to threshold)"""
        labels = np.concatenate((np.zeros((len(t_te_values),)), np.ones((len(t_tr_values),))))
        results = np.concatenate((t_te_values, t_tr_values))
        auc = metrics.roc_auc_score(labels, results)
        ap = metrics.average_precision_score(labels, results)
        info = 'MIA via {n}: the attack auc is {auc:.3f}, ap is {ap:.3f}'.format(n=v_name, auc=auc, ap=ap)
        print(info)
        return info, auc

    def compute_attack_acc(self, method_names, score_signs, if_per_class_thres=False):
        """Compute Attack accuracy"""
        if if_per_class_thres:
            mem_inf_thre_func = self._mem_inf_thre_perclass
            loginfo = 'per class threshold\n'
        else:
            mem_inf_thre_func = self._mem_inf_thre
            loginfo = 'overall threshold\n'
        results = []
        for i in range(self.num_methods):
            if score_signs[i] == '+':
                info, result = mem_inf_thre_func(method_names[i], self.s_tr_scores[i], self.s_te_scores[i],
                                                 self.t_tr_scores[i], self.t_te_scores[i])
                loginfo += info + '\n'
                results.append(result)

            else:
                info, result = mem_inf_thre_func(method_names[i], -self.s_tr_scores[i], -self.s_te_scores[i],
                                                 -self.t_tr_scores[i], -self.t_te_scores[i])
                loginfo += info + '\n'
                results.append(result)
        return loginfo, method_names, results

    def compute_attack_auc(self, method_names, score_signs):
        """Compute attack AUC (and AP)"""
        loginfo = ''
        results = []
        for i in range(self.num_methods):
            if score_signs[i] == '+':
                info, result = self._mem_inf_roc(method_names[i], self.s_tr_scores[i], self.s_te_scores[i],
                                                 self.t_tr_scores[i], self.t_te_scores[i])
                loginfo += info + '\n'
                results.append(result)
            else:
                info, result = self._mem_inf_roc(method_names[i], -self.s_tr_scores[i], -self.s_te_scores[i],
                                                 -self.t_tr_scores[i], -self.t_te_scores[i])
                loginfo += info + '\n'
                results.append(result)
        return loginfo, method_names, results


class Benchmark_Blackbox(Benchmark):
    def compute_bb_scores(self):
        self.s_tr_outputs, self.s_tr_loss = self.s_tr_scores
        self.s_te_outputs, self.s_te_loss = self.s_te_scores
        self.t_tr_outputs, self.t_tr_loss = self.t_tr_scores
        self.t_te_outputs, self.t_te_loss = self.t_te_scores

        # whether the prediction is correct [num_samples,]
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels).astype(int)

        # confidence prediction of the ground-truth class [num_samples,]
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        # entropy of the prediction [num_samples,]
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        # proposed modified entropy [num_samples,]
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        """compute the entropy of the prediction"""
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        """-(1-f(x)_y) log(f(x)_y) - \sum_i f(x)_i log(1-f(x)_i)"""

        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _mem_inf_via_corr(self):
        """perform membership inference attack based on whether the input is correctly classified or not"""
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + 1 - t_te_acc)
        info = 'MIA via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(
            acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc)
        print(info)
        return info, mem_inf_acc

    def compute_attack_acc(self, method_names=[], all_methods=True, if_per_class_thres=True):
        """Compute Attack accuracy"""
        if if_per_class_thres:
            mem_inf_thre_func = self._mem_inf_thre_perclass
            loginfo = 'per class threshold\n'
        else:
            mem_inf_thre_func = self._mem_inf_thre
            loginfo = 'overall threshold\n'
        results = []
        methods = []
        if (all_methods) or ('correctness' in method_names):
            info, result = self._mem_inf_via_corr()
            loginfo += info + '\n'
        if (all_methods) or ('confidence' in method_names):
            info, result = mem_inf_thre_func('confidence', self.s_tr_conf, self.s_te_conf,
                                             self.t_tr_conf, self.t_te_conf)
            loginfo += info + '\n'
            results.append(result)
            methods.append('confidence ACC')
        if (all_methods) or ('entropy' in method_names):
            info, result = mem_inf_thre_func('entropy', -self.s_tr_entr, -self.s_te_entr,
                                             -self.t_tr_entr, -self.t_te_entr)
            loginfo += info + '\n'
            results.append(result)
            methods.append('entropy ACC')
        if (all_methods) or ('modified entropy' in method_names):
            info, result = mem_inf_thre_func('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr,
                                             -self.t_tr_m_entr, -self.t_te_m_entr)
            loginfo += info + '\n'
            results.append(result)
            methods.append('modified entropy ACC')
        if (all_methods) or ('loss' in method_names):
            info, result = mem_inf_thre_func('loss', -self.s_tr_loss, -self.s_te_loss,
                                             -self.t_tr_loss, -self.t_te_loss)
            loginfo += info + '\n'
            results.append(result)
            methods.append('loss ACC')
        return loginfo, methods, results

    def compute_attack_auc(self, method_names=[], all_methods=True):
        """Compute all attack AUC"""
        loginfo = ''
        methods = []
        results = []
        if (all_methods) or ('confidence' in method_names):
            info, result = self._mem_inf_roc('confidence', self.s_tr_conf, self.s_te_conf,
                                             self.t_tr_conf, self.t_te_conf)
            loginfo += info + '\n'
            results.append(result)
            methods.append('confidence AUC')
        if (all_methods) or ('entropy' in method_names):
            info, result = self._mem_inf_roc('entropy', -self.s_tr_entr, -self.s_te_entr,
                                             -self.t_tr_entr, -self.t_te_entr)
            loginfo += info + '\n'
            results.append(result)
            methods.append('entropy AUC')
        if (all_methods) or ('modified entropy' in method_names):
            info, result = self._mem_inf_roc('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr,
                                             -self.t_tr_m_entr, -self.t_te_m_entr)
            loginfo += info + '\n'
            results.append(result)
            methods.append('modified entropy AUC')
        if (all_methods) or ('loss' in method_names):
            info, result = self._mem_inf_roc('loss', -self.s_tr_loss, -self.s_te_loss,
                                             -self.t_tr_loss, -self.t_te_loss)
            loginfo += info + '\n'
            results.append(result)
            methods.append('modified entropy AUC')
        return loginfo, methods, results


def compute_norm_metrics(gradient):
    """Compute the metrics"""
    l1 = np.linalg.norm(gradient, ord=1)
    l2 = np.linalg.norm(gradient)
    Min = np.linalg.norm(gradient, ord=-np.inf)  ## min(abs(x))
    Max = np.linalg.norm(gradient, ord=np.inf)  ## max(abs(x))
    Mean = np.average(gradient)
    Skewness = skew(gradient)
    Kurtosis = kurtosis(gradient)
    return [l1, l2, Min, Max, Mean, Skewness, Kurtosis]


class BaseAttacker(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.set_cuda_device()
        self.set_seed()
        self.set_dataloader()
        self.set_criterion()
        self.load_models()

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def set_criterion(self):
        self.crossentropy = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)

    def set_seed(self):
        """Set random seed"""
        random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.args.random_seed)

    @abc.abstractmethod
    def set_dataloader(self):
        """The function to set the dataloader"""
        self.data_root = None
        self.dataset = None
        self.num_classes = None
        self.dataset_size = None
        self.transform_train = None
        self.transform_test = None
        self.target_trainloader = None
        self.target_testloader = None
        self.shadow_trainloader = None
        self.shadow_testloader = None
        self.loader_dict = None

    def load_models(self):
        target_path = os.path.join(self.args.target_path, 'model.pt')
        target_model = torch.load(target_path).to(self.device)
        self.target_model = target_model
        print('Loading target model from ', target_path)
        shadow_path = os.path.join(self.args.shadow_path, 'model.pt')
        shadow_model = torch.load(shadow_path).to(self.device)
        self.shadow_model = shadow_model
        print('Loading shadow model from ', shadow_path)
        self.model_dict = {'t': self.target_model, 's': self.shadow_model}

    def run_blackbox_attacks(self):
        """Run black-box attacks """
        t_logits_pos, t_posteriors_pos, t_losses_pos, t_labels_pos = self.get_blackbox_statistics(
            self.target_trainloader, self.target_model)
        t_logits_neg, t_posteriors_neg, t_losses_neg, t_labels_neg = self.get_blackbox_statistics(
            self.target_testloader, self.target_model)
        s_logits_pos, s_posteriors_pos, s_losses_pos, s_labels_pos = self.get_blackbox_statistics(
            self.shadow_trainloader, self.shadow_model)
        s_logits_neg, s_posteriors_neg, s_losses_neg, s_labels_neg = self.get_blackbox_statistics(
            self.shadow_testloader, self.shadow_model)

        ## metric_based attacks
        bb_benchmark = Benchmark_Blackbox(shadow_train_scores=[s_posteriors_pos, s_losses_pos],
                                          shadow_test_scores=[s_posteriors_neg, s_losses_neg],
                                          target_train_scores=[t_posteriors_pos, t_losses_pos],
                                          target_test_scores=[t_posteriors_neg, t_losses_neg])
        bb_benchmark.load_labels(s_labels_pos, s_labels_neg, t_labels_pos, t_labels_neg, self.num_classes)
        bb_benchmark.compute_bb_scores()

        ## nn attack
        info, names, results = self.run_nn_attack(s_logits_pos, s_logits_neg, t_logits_pos, t_logits_neg)

        ### Save results
        log_info = info
        all_names = [names]
        all_results = [results]
        info, names, results = bb_benchmark.compute_attack_acc()
        all_names.append(names)
        all_results.append(results)
        log_info += info
        info, names, results = bb_benchmark.compute_attack_auc()
        all_names.append(names)
        all_results.append(results)
        log_info += info
        self.bb_loginfo = log_info
        self.bb_results = np.concatenate(all_results)
        self.bb_names = np.concatenate(all_names)

    def run_whitebox_attacks(self):
        """Run white-box attacks"""

        def run_case(partition, subset, grad_type):
            if partition == 's':
                model_dir = self.args.shadow_path
            else:
                assert partition == 't'
                model_dir = self.args.target_path
            filename = f'{partition}_{subset}_{grad_type}'
            loadername = f'{partition}_{subset}'
            path = os.path.join(model_dir, 'attack', filename + '.pkl')

            if os.path.exists(path):
                stat = unpickle(path)
            else:
                if grad_type == 'x':
                    stat = self.gradient_based_attack_wrt_x(self.loader_dict[loadername], self.model_dict[partition])
                else:
                    assert grad_type == 'w'
                    stat = self.gradient_based_attack_wrt_w(self.loader_dict[loadername], self.model_dict[partition])
                savepickle(stat, path)
            return stat

        ### Grad w.r.t. x
        s_pos_x = run_case('s', 'pos', 'x')
        s_neg_x = run_case('s', 'neg', 'x')
        t_pos_x = run_case('t', 'pos', 'x')
        t_neg_x = run_case('t', 'neg', 'x')

        ### Grad w.r.t. w
        s_pos_w = run_case('s', 'pos', 'w')
        s_neg_w = run_case('s', 'neg', 'w')
        t_pos_w = run_case('t', 'pos', 'w')
        t_neg_w = run_case('t', 'neg', 'w')

        ### Save results
        all_names = []
        all_results = []
        log_info = ''
        wb_benchmark = Benchmark(shadow_train_scores=[s_pos_x['l1'], s_pos_x['l2'], s_pos_w['l1'], s_pos_w['l2']],
                                 shadow_test_scores=[s_neg_x['l1'], s_neg_x['l2'], s_neg_w['l1'], s_neg_w['l2']],
                                 target_train_scores=[t_pos_x['l1'], t_pos_x['l2'], t_pos_w['l1'], t_pos_w['l2']],
                                 target_test_scores=[t_neg_x['l1'], t_neg_x['l2'], t_neg_w['l1'], t_neg_w['l2']])
        info, names, results = wb_benchmark.compute_attack_acc(
            method_names=['grad_wrt_x_l1 ACC', 'grad_wrt_x_l2 ACC', 'grad_wrt_w_l1 ACC', 'grad_wrt_w_l2 ACC'],
            score_signs=['-', '-', '-', '-'])
        all_names.append(names)
        all_results.append(results)
        log_info += info

        info, names, results = wb_benchmark.compute_attack_auc(
            method_names=['grad_wrt_x_l1 AUC', 'grad_wrt_x_l2 AUC', 'grad_wrt_w_l1 AUC', 'grad_wrt_w_l2 AUC'],
            score_signs=['-', '-', '-', '-'])
        all_names.append(names)
        all_results.append(results)
        log_info += info
        print(log_info)
        self.wb_loginfo = log_info
        self.wb_results = np.concatenate(all_results)
        self.wb_names = np.concatenate(all_names)

    def save_results(self):
        """Save to attack_log.txt file and .csv file"""
        with open(os.path.join(self.save_dir, 'attack_log.txt'), 'a+') as f:
            log_info = '=' * 100 + '\n' + self.args.target_path + '\n' + self.wb_loginfo + '\n' + self.bb_loginfo
            f.writelines(log_info)
        write_csv(os.path.join(self.save_dir, 'attack_log.csv'),
                  self.args.target_path.split('/')[-1],
                  np.concatenate([self.bb_results, self.wb_results]),
                  np.concatenate([self.bb_names, self.wb_names]))

    def gradient_based_attack_wrt_x(self, dataloader, model):
        """Gradient w.r.t. input"""
        model.eval()

        ## store results
        names = ['l1', 'l2', 'Min', 'Max', 'Mean', 'Skewness', 'Kurtosis']
        all_stats = {}
        for name in names:
            all_stats[name] = []

        ## iterate over batches
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ## iterate over samples within a batch
            for input, target in zip(inputs, targets):
                input = torch.unsqueeze(input, 0)
                input.requires_grad = True
                output = model(input)
                loss = self.crossentropy(output, torch.unsqueeze(target, 0))
                model.zero_grad()
                loss.backward()

                ## get gradients
                gradient = input.grad.view(-1).cpu().numpy()

                ## get statistics
                stats = compute_norm_metrics(gradient)
                for i, stat in enumerate(stats):
                    all_stats[names[i]].append(stat)

        for name in names:
            all_stats[name] = np.array(all_stats[name])
        return all_stats

    def gradient_based_attack_wrt_w(self, dataloader, model):
        """Gradient w.r.t. weights"""
        model.eval()

        ## store results
        names = ['l1', 'l2', 'Min', 'Max', 'Mean', 'Skewness', 'Kurtosis']
        all_stats = {}
        for name in names:
            all_stats[name] = []

        ## iterate over batches
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ## iterate over samples within a batch
            for input, target in zip(inputs, targets):
                input = torch.unsqueeze(input, 0)
                output = model(input)
                loss = self.crossentropy(output, torch.unsqueeze(target, 0))
                model.zero_grad()
                loss.backward()

                ## get gradients
                grads_onesample = []
                for param in model.parameters():
                    grads_onesample.append(param.grad.view(-1))
                gradient = torch.cat(grads_onesample)
                gradient = gradient.cpu().numpy()

                ## get statistics
                stats = compute_norm_metrics(gradient)
                for i, stat in enumerate(stats):
                    all_stats[names[i]].append(stat)

        for name in names:
            all_stats[name] = np.array(all_stats[name])
        return all_stats

    def get_blackbox_statistics(self, dataloader, model):
        """Compute the blackbox statistics (for blackbox attacks)"""
        model.eval()

        logits = []
        labels = []
        losses = []
        posteriors = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.crossentropy_noreduce(outputs, targets)
                posterior = self.softmax(outputs)
                logits.extend(outputs.cpu().numpy())
                posteriors.extend(posterior.cpu().numpy())
                labels.append(targets.cpu().numpy())
                losses.append(loss.cpu().numpy())
        logits = np.vstack(logits)
        posteriors = np.vstack(posteriors)
        labels = np.concatenate(labels)
        losses = np.concatenate(losses)
        return logits, posteriors, losses, labels

    def run_nn_attack(self, s_logits_pos, s_logits_neg, t_logits_pos, t_logits_neg, if_load_checkpoint=True):
        checkpoint_dir = os.path.join(self.args.shadow_path, 'attack', 'nn')
        mkdir(checkpoint_dir)
        trainer = NNAttackTrainer(self.args, checkpoint_dir)
        trainer.set_loader(s_logits_pos, s_logits_neg, t_logits_pos, t_logits_neg)
        if os.path.exists(os.path.join(checkpoint_dir, 'attack_model.pt')) and if_load_checkpoint:
            attack_model = torch.load(os.path.join(checkpoint_dir, 'attack_model.pt')).to(self.device)
            print('Load NN attack from checkpoint_dir')
        else:
            max_epoch = 200
            lr = 0.001
            attack_model = NNAttack(self.num_classes)
            optimizer = optim.Adam(attack_model.parameters(), lr=lr)
            logger = trainer.logger
            print('Train NN attack')
            for _ in range(max_epoch):
                train_loss, train_acc = trainer.train(attack_model, optimizer)
                test_loss, test_acc, _ = trainer.test(attack_model)
                logger.append([train_loss, test_loss, train_acc, test_acc])
            torch.save(attack_model, os.path.join(checkpoint_dir, 'attack_model.pt'))
        _, attack_acc, attack_auc = trainer.test(attack_model)
        info = 'MIA via NN : the attack acc is {acc:.3f} \n'.format(acc=attack_acc / 100)
        info += 'MIA via NN : the attack auc is {auc:.3f} \n'.format(auc=attack_auc)
        return info, ['NN ACC', 'NN AUC'], [attack_acc / 100, attack_auc]


class NNAttack(nn.Module):
    """NN attack model"""

    def __init__(self, input_dim, output_dim=1, hiddens=[100]):
        super(NNAttack, self).__init__()
        self.layers = []
        for i in range(len(hiddens)):
            if i == 0:
                layer = nn.Linear(input_dim, hiddens[i])
            else:
                layer = nn.Linear(hiddens[i - 1], hiddens[i])
            self.layers.append(layer)
        self.last_layer = nn.Linear(hiddens[-1], output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = self.relu(layer(output))
        output = self.last_layer(output)
        return output


class NNAttackTrainer(BaseTrainer):
    """Trainer for the NN attack"""

    @staticmethod
    def construct_dataloader(stat_pos, stat_neg):
        """Construct dataloader from statistics"""
        attack_data = np.concatenate([stat_neg, stat_pos], axis=0)
        attack_data = np.sort(attack_data, axis=1)
        attack_targets = np.concatenate([np.zeros(len(stat_neg)), np.ones(len(stat_pos))])
        attack_targets = attack_targets.astype(np.int)
        attack_indices = np.arange(len(attack_data))
        np.random.shuffle(attack_indices)
        attack_data = attack_data[attack_indices]
        attack_targets = attack_targets[attack_indices]
        tensor_x = torch.from_numpy(attack_data)
        tensor_y = torch.from_numpy(attack_targets)
        tensor_y = tensor_y.unsqueeze(-1).type(torch.FloatTensor)
        attack_dataset = data.TensorDataset(tensor_x, tensor_y)
        attack_loader = data.DataLoader(attack_dataset, batch_size=256, shuffle=True)
        return attack_loader

    def set_loader(self, s_logits_pos, s_logits_neg, t_logits_pos, t_logits_neg):
        """Set the training and testing dataloader"""
        self.trainloader = self.construct_dataloader(s_logits_pos, s_logits_neg)
        self.testloader = self.construct_dataloader(t_logits_pos, t_logits_neg)

    def set_criterion(self):
        """Set the training criterion (BCE by default)"""
        self.criterion = nn.BCELoss()

    def train(self, model, optimizer):
        """Train"""
        model.train()
        criterion = self.criterion
        losses = AverageMeter()
        top1 = AverageMeter()
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
            if outputs.shape[-1] == 1:
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                outputs = nn.Sigmoid()(outputs)
                prec1 = accuracy_binary(outputs.data, targets.data)
            else:
                prec1 = accuracy(outputs.data, targets.data)[0]
            loss = criterion(outputs, targets)

            ### Record accuracy and loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            ### Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Record the total time for processing the batch
            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

            ### Progress bar
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.trainloader),
                data=dataload_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg
            )
            bar.next()

        bar.finish()
        return (losses.avg, top1.avg)

    def test(self, model):
        """Test"""
        model.eval()
        criterion = self.criterion
        losses = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()
        ytest = []
        ypred_score = []

        bar = Bar('Processing', max=len(self.testloader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                ### Record the data loading time
                dataload_time.update(time.time() - time_stamp)

                ### Forward
                outputs = model(inputs)
                if outputs.shape[-1] == 1:
                    outputs = outputs.view(-1)
                    targets = targets.view(-1)
                    outputs = nn.Sigmoid()(outputs)
                    prec1 = accuracy_binary(outputs.data, targets.data)
                    ytest.append(targets.cpu().numpy())
                    ypred_score.append(outputs.cpu().numpy())
                else:
                    prec1 = accuracy(outputs.data, targets.data)[0]
                    ytest.append(targets.cpu().numpy())
                    outputs = nn.Softmax(dim=1)(outputs)
                    ypred_score.append(outputs.cpu().numpy()[:, 1])

                ### Evaluate
                loss = criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))

                ### Record the total time for processing the batch
                batch_time.update(time.time() - time_stamp)
                time_stamp = time.time()

                ### Progress bar
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(self.testloader),
                    data=dataload_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                )
                bar.next()
            bar.finish()
        ytest = np.concatenate(ytest)
        ypred_score = np.concatenate(ypred_score)
        auc, ap, f1, pos_num, frac = metrics_binary(ytest, ypred_score)
        return (losses.avg, top1.avg, auc)

    def set_logger(self):
        """Set up logger"""
        title = self.args.dataset
        self.start_epoch = 0
        logger = Logger(os.path.join(self.save_dir, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])
        self.logger = logger
