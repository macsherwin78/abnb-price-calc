#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

import pickle

from sklearn import ensemble
data_abb = pd.read_csv('ABB_ML_ready.csv')
data_abb.head()


X = data_abb.iloc[:,2:-1]
y = data_abb.iloc[:,1]


x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.10, 
                                                                          random_state=42,
                                                                          shuffle=True)



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model_gb = ensemble.GradientBoostingRegressor(**params)

model_gb.fit(x_training_set, y_training_set)

# Saving model to disk
pickle.dump(model_gb, open('model_gb.pkl','wb'))

