import torch.nn.functional as F
import torch


def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)


def adjust_learning_rate(optimizer, epoch, gamma, schedule_milestone):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    if epoch in schedule_milestone:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
