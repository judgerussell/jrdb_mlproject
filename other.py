# %% 
# import statements

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np

def stratCV(model, nfolds, train_X, train_Y, **params):
    mskf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True)

    for train_index, valid_index in mskf.split(train_X, train_Y):
        print("TRAIN:", train_index, "VALID:", valid_index)
        X_train, X_valid = train_X[train_index], train_X[valid_index]
        Y_train, Y_valid = train_Y[train_index], train_Y[valid_index]

    m = MultiOutputRegressor(model(**params))
    m.fit(X_train, Y_train)
    y_preds = m.predict(X_valid)
    y_score = log_loss(Y_valid, y_preds)
    print(y_score)
# %%

# import preprocessed data

X = np.load("preprocessed_boost/x.npy", allow_pickle=True)
Y = np.load("preprocessed_boost/y.npy", allow_pickle=True)
train_drug = np.load("preprocessed_boost/train_drug.npy", allow_pickle=True)

# %%
# creating k stratified folds for CV
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=.25, shuffle=True)


#stratify = np.concatenate((train_drug, train_Y), 1)


# multilabel stratified kfold
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True)

# %% 
# adaboost

N = 150

for n in range(15, N, 5): 
        stratCV(SVR, 10, train_X, train_Y)
    
# %%
