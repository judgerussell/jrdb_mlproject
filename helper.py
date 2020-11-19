
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

def stratCV(model, nfolds, train_X, train_Y, **params):
    mskf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True)

    for train_index, valid_index in mskf.split(train_X, train_Y):
        print("TRAIN:", train_index, "VALID:", valid_index)
        X_train, X_valid = train_X[train_index], train_X[valid_index]
        Y_train, Y_valid = train_Y[train_index], train_Y[valid_index]

    m = MultiOutputRegressor(model(params))
    m.fit(X_train, Y_train)
    y_score = m.score(X_valid, Y_valid)
    print(y_score)