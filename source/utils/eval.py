from __future__ import print_function, absolute_import
import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import metrics

__all__ = ['accuracy', 'accuracy_binary', 'metrics_binary', 'plot_roc']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.reshape(correct[:k], (-1,)).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_binary(output, target):
    """Computes the accuracy for binary classification"""
    batch_size = target.size(0)

    pred = output.view(-1) >= 0.5
    truth = target.view(-1) >= 0.5
    acc = pred.eq(truth).float().sum(0).mul_(100.0 / batch_size)
    return acc


def metrics_binary(ytest, predict_y_score):
    """Evaluation metrics for binary classification (including auc, ap, f1)"""
    predict_y_label = np.array(predict_y_score >= 0.5).astype(np.int)
    pos_num = np.sum(predict_y_label)
    frac = np.sum(predict_y_label) * 100 / len(ytest)
    try:
        auc = metrics.roc_auc_score(ytest, predict_y_score)
    except ValueError:
        auc = 0
    ap = metrics.average_precision_score(ytest, predict_y_score)
    f1 = metrics.f1_score(ytest, predict_y_label)
    return auc, ap, f1, pos_num, frac


def plot_roc(pos_results, neg_results):
    labels = np.concatenate((np.zeros((len(neg_results),)), np.ones((len(pos_results),))))
    results = np.concatenate((neg_results, pos_results))
    fpr, tpr, threshold = metrics.roc_curve(labels, results, pos_label=1)
    auc = metrics.roc_auc_score(labels, results)
    ap = metrics.average_precision_score(labels, results)
    return fpr, tpr, threshold, auc, ap
