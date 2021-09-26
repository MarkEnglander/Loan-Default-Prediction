import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd = pd.read_csv('train_v2.csv')

x = 10

fig, axs = plt.subplots(x)
fig.suptitle('Features')
for i in range(x):
    axs[i].hist(pd['f' + str(i + 1)], bins=100)

plt.show()
