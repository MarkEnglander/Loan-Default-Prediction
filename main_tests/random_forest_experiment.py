import main_upsample_run_1
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


# since we expect it to converge faster at the start, split determines at which point we start doing less tests
def run_rf_experiment(start=10, stop=400, num=25, split=100, frac_before_split=0.6, log_plot=False):
    split = int((stop + start) / 2) if split > stop else split
    # Generate the n_estimators space we want to try out
    space1 = np.linspace(start=start, stop=split, num=int(num * frac_before_split), dtype=int)
    space2 = np.linspace(start=split + 5, stop=stop, num=int(num * (1-frac_before_split)), dtype=int)
    n_estimators_space = np.concatenate([space1, space2])
    print(n_estimators_space)

    a_11, a_12, a_21, a_22 = [], [], [], []
    for n_estimators in n_estimators_space:
        a = main_upsample_run_1.run_classifier(RandomForestClassifier(n_estimators=n_estimators), suppress_prints=True)
        a_11.append(a[0][0])
        a_12.append(a[0][1])
        a_21.append(a[1][0])
        a_22.append(a[1][1])
        print('Done with n_estimators set to ' + str(n_estimators))
    plt.title("Random Forest Confusion Matrix, varying n_estimators")

    if log_plot:
        a_11, a_12, a_21, a_22 = np.log(a_11), np.log(a_12), np.log(a_21), np.log(a_22)

    figure, axis = plt.subplots(3, 2)
    axis[0, 0].plot(n_estimators_space, a_11, label='a_11')
    axis[0, 0].set_title("a_11")
    axis[0, 0].grid()

    axis[0, 1].plot(n_estimators_space, a_12, label='a_12')
    axis[0, 1].set_title("a_12")
    axis[0, 1].grid()

    axis[1, 0].plot(n_estimators_space, a_21, label='a_21')
    axis[1, 0].set_title("a_21")
    axis[1, 0].grid()

    axis[1, 1].plot(n_estimators_space, a_22, label='a_22')
    axis[1, 1].set_title("a_22")
    axis[1, 1].grid()

    a, b, c, d = np.array(a_11), np.array(a_12), np.array(a_21), np.array(a_22)
    axis[2, 0].plot(n_estimators_space, (a + d)/(a + b + c + d))
    axis[2, 0].set_title('accuracy')
    axis[2, 0].grid()

    axis[2, 1].plot(n_estimators_space, (d / (c+d)))
    axis[2, 1].set_title('loan defaults recall')
    axis[2, 1].grid()

    plt.show()
