# %%
# import statements
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
import numpy as np

# %%

# import preprocessed data

X = np.load("preprocessed/x.npy", allow_pickle=True)
Y = np.load("preprocessed/y.npy", allow_pickle=True)
train_drug = np.load("preprocessed/train_drug.npy", allow_pickle=True)

def log_loss_metric(y_true, y_pred):
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
# %%
# creating k stratified folds for CV
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=.25, shuffle=True)


#stratify = np.concatenate((train_drug, train_Y), 1)


# multilabel stratified kfold
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True)



# %%
# Ridge Regression
alpha = 0
for train_index, valid_index in mskf.split(train_X, train_Y):
    alpha += 0.5
    print("TRAIN:", train_index, "VALID:", valid_index)
    X_train, X_valid = train_X[train_index], train_X[valid_index]
    Y_train, Y_valid = train_Y[train_index], train_Y[valid_index]

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)
    y_preds = ridge.predict(X_valid)
    y_score = mean_squared_error(Y_valid, y_preds)
    print('alpha: {}'.format(alpha))
    print(y_score)

# %%
# RBF Kernel Ridge Regression

alpha = 0
for train_index, valid_index in mskf.split(train_X, train_Y):
    print("TRAIN:", train_index, "VALID:", valid_index)
    X_train, X_valid = train_X[train_index], train_X[valid_index]
    Y_train, Y_valid = train_Y[train_index], train_Y[valid_index]

    alpha += 0.1
    kridge = KernelRidge(alpha=alpha)
    kridge.fit(X_train, Y_train)
    y_preds = kridge.predict(X_valid)
    y_score = mean_squared_error(Y_valid, y_preds)
    print(y_score)
# %%
