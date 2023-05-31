

# Best fit line --> equal of line y =mx+c
# pip install Quandl

import pandas as pd
import quandl


# df = quandl.get('WIKI/GOOGL')
# print(df.head())
#


import numpy as np # Computing library
from sklearn import preprocessing,svm #shuffle data, biased the data # negative 1 to 1 postive
from sklearn.linear_model import LinearRegression
import math
from sklearn.model_selection import cross_val_score