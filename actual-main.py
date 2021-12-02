import pandas as pd
import numpy as np

import automl
import label_analysis
import useful_functions


df = useful_functions.import_all_data()

label_analysis.this_main(df)

print("if you got here then something might have worked")