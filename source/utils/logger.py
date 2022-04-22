import matplotlib.pyplot as plt
import numpy as np

__all__ = ['AverageMeter', 'Logger']


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        self.fig, self.ax = plt.subplots(dpi=150)

        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = [i.rstrip() for i in name.rstrip().split('\t')]
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        try:
                            self.numbers[self.names[i]].append(float(numbers[i]))
                        except:
                            self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write("{:9}".format(name))
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            if type(num) == int:
                self.file.write("{0:10d}".format(num))
            elif type(num) == str:
                self.file.write("{0:10s}".format(num))
            else:
                self.file.write("{0:10.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None, formats=None,
             x=None, xticks=None,
             ylim=None, grid=True, legend=True,
             xlabel=None, ylabel=None, title=None):
        fig, ax = self.fig, self.ax
        plt.clf()
        names = self.names if names is None else names
        numbers = self.numbers

        if formats is None:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_largeint))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func_float))
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(select_format(formats[0])))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(select_format(formats[1])))

        for _, name in enumerate(names):
            if x is None:
                x = np.arange(len(numbers[name]))
            elif type(x) == str and x in self.names:
                x = np.asarray(numbers[x])
            plt.plot(x, np.asarray(numbers[name]), label=self.title + '(' + name + ')')

        if xticks is not None:
            try:
                plt.xticks((x[0], x[-1]), (xticks[0], xticks[-1]))
            except:
                plt.xticks(x, xticks)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if ylim is not None:
            plt.ylim(ylim)

        plt.grid(grid)

        if legend:
            plt.legend()

    def close(self):
        if self.file is not None:
            self.file.close()


def format_func_largeint(value, tick_number):
    if value == 0.:
        return '0'
    elif value < 1000:
        return '{}'.format(int(value))
    elif value < 10 ** 6:
        value /= 1000
        value = '{}k'.format(int(value))
    else:
        value /= 1000000
        value = '{}m'.format(int(value))
    return value


def format_func_text(value, tick_number):
    return '{}'.format(int(value))


def format_func_float(value, tick_number):
    return '{:.1f}'.format(value)


def format_func_float2(value, tick_number):
    return np.round(value, 2)


def format_func_float3(value, tick_number):
    return np.round(value, 3)


def select_format(format_type):
    if format_type == 'int':
        return format_func_largeint
    elif format_type == 'text':
        return format_func_text
    elif format_type == 'float2':
        return format_func_float2
    elif format_type == 'float3':
        return format_func_float3
    else:
        return format_func_float
