import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd = pd.read_csv('data/train_v2.csv')


def output_single_dists(num):
    for i in range(num):
        plt.hist(pd['f' + str(i + 1)], bins=100)
        plt.title('Feature ' + str(i + 1) + ' Distribution')
        plt.savefig('data-visualisation/individual-features-distributions/feature-' + str(i + 1) + '-dist')
        plt.clf()


def mult_dist_plot(num_plot, start=0):
    x = num_plot

    fig, axs = plt.subplots(x)
    fig.suptitle('Features')
    for i in range(x):
        axs[i].hist(pd['f' + str(i + 1)], bins=100)


output_single_dists(10)
