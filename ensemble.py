# %% 
# import statements

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np
import joblib


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
# random forest
params = {"n_estimators": 200, "max_depth": 30}
#stratCV(RandomForestClassifier, n_folds, train_X, train_Y, 'random_forest', **params)
# %% 
# hist gradient boost
params = {"learning_rate": .1, "max_depth": 25, "early_stopping": True, "l2_regularization": True}    

hist = MultiOutputRegressor(HistGradientBoostingRegressor(**params))
hist.fit(train_X, train_Y)
y_preds = hist.predict(valid_X)
ctrl_i = np.where(valid_X[:, 0] == 1)
y_preds[ctrl_i] = 0
print(log_loss_metric(valid_Y, y_preds))
joblib_file = "joblib_model_hist.pkl"
joblib.dump((hist, y_preds), joblib_file)

# %%

