from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import numpy as np

# read in the data
X = np.array(pd.read_csv('lish-moa/train_features.csv'))
Y = np.array(pd.read_csv('lish-moa/train_targets_scored.csv'))

# multilabel stratified kfold
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=0)

for train_index, test_index in mskf.split(X, Y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   Y_train, Y_test = Y[train_index], Y[test_index]