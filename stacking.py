# %%
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd
import joblib

def log_loss_metric(y_true, y_pred):
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

# import small data
X = np.load("preprocessed_boost/x.npy", allow_pickle=True)
Y = np.load("preprocessed_boost/y.npy", allow_pickle=True)
Y = Y.astype('float')
train_drug = np.load("preprocessed_boost/train_drug.npy", allow_pickle=True)

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=.25, shuffle=True)

hist_params = {"learning_rate": .05, "max_depth": 5, "early_stopping": True, "l2_regularization": .2}   

# models to be stacked
svc = SVC()
ridge = (Ridge())
kridge = (KernelRidge()) 
hist = (HistGradientBoostingRegressor(**hist_params))

models = [('ridge', ridge), ('kernel ridge', kridge), ('hist', hist)]
stack = MultiOutputRegressor(StackingRegressor(estimators=models))
print(train_Y.ndim)
stack.fit(train_X, train_Y)
y_preds = stack.predict(valid_X)
log_loss_metric(valid_Y, y_preds)

# %%
