# %%
# import statements
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
# %%

# import preprocessed data

X = np.load("preprocessed_boost/x.npy", allow_pickle=True)
Y = np.load("preprocessed_boost/y.npy", allow_pickle=True)
test_X = np.load("preprocessed_boost/test_X.npy", allow_pickle=True)
train_drug = np.load("preprocessed/train_drug.npy", allow_pickle=True)

targets = np.load("targets.npy", allow_pickle=True)
test_samples = np.load("test_samples.npy", allow_pickle=True)

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
        
        ctrl_index = np.where(X_valid[:, 0] == 1)
        m = MultiOutputRegressor(model(**params))
        m.fit(X_train, Y_train)
        y_preds =  m.predict(X_valid)
        y_preds[ctrl_index] = 0
        y_score = log_loss_metric(Y_valid, y_preds)
        print(y_score)
        scores.append(y_score)
    
    # Save to file in the current working directory
    joblib_file = "joblib_model_{}.pkl".format(output_name)
    joblib.dump((model, scores), joblib_file)

    return scores


# %%
# creating k stratified folds for CV
train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=.25, shuffle=True)


#stratify = np.concatenate((train_drug, train_Y), 1)


# multilabel stratified kfold
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True)



# %%
# Ridge Regression
params = {"alpha": 1}
stratCV(Ridge, 5, train_X, train_Y, 'ridgetest', **params)

ctrl_i = np.where(test_X[:, 0] == 1)
r = MultiOutputRegressor(Ridge())
r.fit(X, Y)
y_preds = r.predict(valid_X)
print(log_loss_metric(valid_Y, y_preds))
y_preds = r.predict(test_X)
y_preds[ctrl_i] = 0
print(y_preds.shape)
#output = pd.DataFrame(y_preds, columns=targets)
#output.set_index(test_samples, inplace=True)
#output.to_csv('submission.csv')

# %%

# %%
# RBF Kernel Ridge Regression

params = {"alpha": 1}
#stratCV(KernelRidge, 5, train_X, train_Y, 'kridgetest', **params)
kr = MultiOutputRegressor(KernelRidge())
kr.fit(train_X, train_Y)
y_preds = kr.predict(valid_X)
print(log_loss_metric(valid_Y, y_preds))

# %%
