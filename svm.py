# %% 
# import statements

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR,SVC
from sklearn.model_selection import ParameterGrid
import numpy as np
import joblib
# %%

def log_loss_metric(y_true, y_pred):
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

def stratCV(model, nfolds, train_X, train_Y, output_name, **params):
    mskf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True)
    scores = []
    for train_index, valid_index in mskf.split(train_X, train_Y):
        print("TRAIN:", train_index, "VALID:", valid_index)
        X_train, X_valid = train_X[train_index], train_X[valid_index]
        Y_train, Y_valid = train_Y[train_index], train_Y[valid_index]
            
        m = MultiOutputRegressor(model(**params))
        m.fit(X_train, Y_train)
        y_preds =  m.predict(X_valid)
        y_score = log_loss_metric(Y_valid, y_preds)
        print(y_score)
        scores.append(y_score)
    
    # Save to file in the current working directory
    joblib_file = "joblib_model_{}.pkl".format(output_name)
    joblib.dump((model, scores), joblib_file)

    return scores

    # %%

# import preprocessed data

X = np.load("preprocessed_boost/x.npy", allow_pickle=True)
Y = np.load("preprocessed_boost/y.npy", allow_pickle=True)
Y = Y.astype('float')
train_drug = np.load("preprocessed_boost/train_drug.npy", allow_pickle=True)

# %%
# creating k stratified folds for CV
train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=.25, shuffle=True)

n_folds = 6
# %%
#svm
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
param_grid = ParameterGrid(param_grid)

scores = [(log_loss_metric(valid_Y, MultiOutputClassifier(SVC(**x)).fit(train_X, train_Y)).predict(valid_X),x) for x in param_grid]

print(scores)

# %%
