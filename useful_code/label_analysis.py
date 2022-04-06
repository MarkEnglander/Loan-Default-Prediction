import numpy as np
import matplotlib.pyplot as plt
from useful_code.useful_functions import separate_into_xy, separate_into_xy_regression_case


def extract_y_class(df1):
    X, y = separate_into_xy(df1)
    y = np.array(y)
    return y


def extract_y_reg(df1):
    X, y = separate_into_xy_regression_case(df1)
    y = np.array(y)
    return y


# maybe it's worth testing to make sure that the number of y's with 1 is as expected?
def visualize_labels(y, label='class'):
    plt.hist(y, bins=100, log=False)
    plt.savefig('data-visualisation/label-analysis/label-distribution-' + label)
    plt.clf()

    plt.hist(y, bins=100, log=True)
    plt.savefig('data-visualisation/label-analysis/label-distribution-' + label + '-log')
    plt.clf()


def pie_chart_class(y):
    non_zero = np.count_nonzero(y)
    one_class = non_zero / len(y)
    zero_class = 1 - one_class
    labels = [str(np.round(100 * zero_class, 1)) + '%', str(np.round(100 * one_class, 1)) + '%']
    colors = ['green', 'red']
    sizes = [zero_class, one_class]
    plt.pie(sizes, labels=labels, colors=colors)
    plt.savefig('data-visualisation/label-analysis/class-pie-chart')
    plt.clf()


def do_dist_plots(df):
    y_class = extract_y_class(df)
    y_reg = extract_y_reg(df)
    visualize_labels(y_reg, 'reg')
    visualize_labels(y_class, 'class')


def this_main(df):
    y = extract_y_class(df)
    pie_chart_class(y)
