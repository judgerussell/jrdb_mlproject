# %%
# import statements
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import log_loss
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import joblib

def log_loss_metric(y_true, y_pred):
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

def train(model, device, train_loader, opt, epoch):
    model.train()

    for

def NNstratCV(model, nfolds, train_X, train_Y, output_name, **params):
    device = torch.device('cpu')
    mskf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True)
    scores = []
    for train_index, valid_index in mskf.split(train_X, train_Y):
        # create CV split
        print("TRAIN:", train_index, "VALID:", valid_index)
        X_train, X_valid = train_X[train_index], train_X[valid_index]
        Y_train, Y_valid = train_Y[train_index], train_Y[valid_index]
            
        # create tensor
        X_train_tensor = torch.tensor(X_train, device)
        Y_train_tensor = torch.tensor(Y_train, device)
        X_valid_tensor = torch.tensor(X_valid, device)
        Y_valid_tensor = torch.tensor(Y_valid, device)

        # create data loader
        #train_data = utils.TensorDataset()
    
    # Save to file in the current working directory
    joblib_file = "joblib_model_{}.pkl".format(output_name)
    joblib.dump((model, scores), joblib_file)

    return scores

# %%

class NN1(nn.Module):
    def __init__(self, num_features, num_targets, hidden_sizes):
        super(NN1, self).__init__()

        self.batchnorm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU()

        self.linear1 = nn.Linear(num_features, hidden_sizes[0])
        self.dropout2 = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.utils.weight_norm(nn.Linear(hidden_sizes[0], num_targets))
        

    def forward(self, x):
        batch = self.batchnorm1(x)
        drop = self.dropout1(batch)
        relu = self.relu1(drop)

        lin1 = self.linear1(relu)
        drop2 = self.dropout2(lin1)
        relu2 = self.relu2(drop2)
        output = self.linear2(relu2)

# %%

# import preprocessed data

X = np.load("preprocessed_boost/x.npy", allow_pickle=True)
Y = np.load("preprocessed_boost/y.npy", allow_pickle=True)
train_drug = np.load("preprocessed/train_drug.npy", allow_pickle=True)


ctrl_index = np.where(X[:, 0] == 1)
print(ctrl_index)

