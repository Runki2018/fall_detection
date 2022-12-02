import matplotlib.pyplot as plt
import numpy
import json

import numpy as np


def plot_bar(x, y, y_labels=('AP50', 'AP'), total_width=0.8):
    """
    x: numpy.ndarray([N])
    y: list([numpy.ndarray([N]), numpy.ndarray([N])])
    """
    assert len(y) == len(y_labels)
    width = total_width / len(y)
    xx = np.arange(len(x))
    x_labels = [str(round(v, 2)) for v in x]
    plt.xticks(xx, x_labels)
    xx = xx - (total_width - width) / 2
    for i, yy in enumerate(y):
        plt.bar(xx + i*width, yy, width=width, label=y_labels[i])
    plt.legend()
    plt.show()

def plot_line(x, y):
    xx = np.arange(len(x))
    plt.plot(xx, y)
    x_labels = [str(round(v, 2)) for v in x]
    plt.xticks(xx, x_labels)
    plt.show()


if __name__ == '__main__':
    with open('runs/v7tiny_e50_640_10k/search_dm_wm2/model_size.json', 'r') as fd:
        json_dict = json.load(fd)
    y = np.array(json_dict['results'])  # P, R, AP50, AP
    y = [y[:, 2] - 0.3, y[:, 3]]  # AP50, AP
    x = np.array(json_dict['model_size'])
    plot_bar(x, y, y_labels=('AP50', 'AP'), total_width=0.8)
    plot_line(x, y[1])
    plot_line(x, y[1]/x)

