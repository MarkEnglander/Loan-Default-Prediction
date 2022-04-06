import pandas as pd
import numpy as np


def accuracy(y, test_y):
    y_arr, test_y_arr = np.array(y), np.array(test_y)
    total = len(y_arr)
    correct = 0
    for i in range(len(y_arr)):
        if y_arr[i] == test_y_arr[i]:
            correct += 1
    return correct / total


def recall(y, test_y):
    y_arr, test_y_arr = np.array(y), np.array(test_y)
    total = 0
    recalled = 0
    for i in range(len(y_arr)):
        if y_arr[i] == 1:
            total += 1
            if test_y_arr[i] == 1:
                recalled += 1
    if total == 0 or recalled == 0:
        if 1 not in y_arr or 1 not in test_y_arr:
            raise Exception('This model predicted 0 for everything')
    return recalled / total
