# %%
# import statements
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
import joblib

def log_loss_metric(y_true, y_pred):
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

def train(model, device, train_loader, opt, epoch):
    model.train()

    for index, (data, target) in enumerate(train_loader):
        opt.zero_grad()
        output = model(data.to(device))
        loss = nn.BCEWithLogitsLoss()(output, target.to(device))
        loss.backward()
        opt.step()
        if index % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    return loss 

def NNstratCV(model, nfolds, train_X, train_Y, output_name, params):
    device = torch.device('cpu')
    mskf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True)
    scores = []
    for train_index, valid_index in mskf.split(train_X, train_Y):
        # create CV split
        print("TRAIN:", train_index, "VALID:", valid_index)
        X_train, X_valid = train_X[train_index], train_X[valid_index]
        Y_train, Y_valid = train_Y[train_index], train_Y[valid_index]
            
        # create tensor
        X_train_tensor = torch.tensor(X_train, device=device)
        Y_train_tensor = torch.tensor(Y_train, device=device)
        X_valid_tensor = torch.tensor(X_valid, device=device)
        Y_valid_tensor = torch.tensor(Y_valid, device=device)

        # create data loader
        trainload = utils.DataLoader(utils.TensorDataset(X_train_tensor, Y_train_tensor), batch_size=32)
        testload = utils.DataLoader(utils.TensorDataset(X_valid_tensor, Y_valid_tensor), batch_size=32)
        nnmodel = model(**params)
        nnmodel.to(device)
        nnmodel.double()
        opt = optim.Adam(nnmodel.parameters(), lr=0.1)
        
        for epoch in range(10):
            train_acc = train(nnmodel, device, trainload, opt, epoch)
            
        # torch.save(nnmodel.state_dict(), "nn.pt")
    # Save to file in the current working directory
    joblib_file = "joblib_model_{}.pkl".format(output_name)
    joblib.dump((nnmodel, scores), joblib_file)

    return scores

# %%

# %%

class NN1(nn.Module):
    def __init__(self, num_features, num_targets, hidden_sizes):
        super(NN1, self).__init__()

        self.batchnorm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_sizes[0]))
        self.batchnorm2 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout2 = nn.Dropout(p=0.3)
        self.prelu = nn.PReLU()
        self.linear2 = nn.utils.weight_norm(nn.Linear(hidden_sizes[0], num_targets))
        

    def forward(self, x):
        batch = self.batchnorm1(x)
        drop = self.dropout1(batch)
        relu = self.relu1(drop)
        lin1 = self.linear1(relu)
        batch2 = self.batchnorm2(lin1)
        drop2 = self.dropout2(lin1)
        prelu = self.prelu(drop2)
        output = self.linear2(prelu)

        return output

class NN2(nn.Module):
    def __init__(self, num_features, num_targets, hidden_sizes):
        super(NN2, self).__init__()

        self.batchnorm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(p=0.4)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(num_features, hidden_sizes[0])
        self.batchnorm2 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout2 = nn.Dropout(p=0.25)
        self.prelu = nn.PReLU()
        self.linear2 = nn.utils.weight_norm(nn.Linear(hidden_sizes[0], hidden_sizes[1]))

        self.batchnorm3 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout3 = nn.Dropout(p=0.2)
        self.prelu2 = nn.PReLU()
        self.linear3 = nn.utils.weight_norm(nn.Linear(hidden_sizes[1], num_targets))
        

    def forward(self, x):
        batch = self.batchnorm1(x)
        drop = self.dropout1(batch)
        relu = self.relu1(drop)
        lin1 = self.linear1(relu)
        batch2 = self.batchnorm2(lin1)
        drop2 = self.dropout2(lin1)
        prelu = self.prelu(drop2)
        lin2 = self.linear2(prelu)
        drop3 = self.dropout3(lin2)
        prelu2 = self.prelu2(drop3)
        output = self.linear3(prelu2)

        return output

# %%

# import preprocessed data

X = np.load("preprocessed/x.npy", allow_pickle=True)
Y = np.load("preprocessed/y.npy", allow_pickle=True)
train_drug = np.load("preprocessed/train_drug.npy", allow_pickle=True)

print(Y.shape)
train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=.25, shuffle=True)

ctrl_index = np.where(X[:, 0] == 1)
print(ctrl_index)

#NNstratCV(NN1, 5, train_X, train_Y, 'neuralnet1', params={'num_features': 598, 'num_targets': 206, 'hidden_sizes': [1024]})
#NNstratCV(NN2, 5, train_X, train_Y, 'neuralnet2', params={'num_features': 598, 'num_targets': 206, 'hidden_sizes': [1024, 512]})
# %%
# train on full data

nnmodel = NN2(598, 206, [1024, 512])
nnmodel.to(device)
nnmodel.double()
opt = optim.Adam(nnmodel.parameters(), lr=0.1)
device = torch.device('cpu')
X_train_tensor = torch.tensor(train_X, device=device)
Y_train_tensor = torch.tensor(train_Y, device=device)
trainload = utils.DataLoader(utils.TensorDataset(X_train_tensor, Y_train_tensor), batch_size=32)
for epoch in range(10):
    train_acc = train(nnmodel, device, trainload, opt, epoch)

# %%
X_valid_tensor = torch.tensor(X_valid, device=device)

pred = nnmodel()
nn.BCEWithLogitsLoss()(pred, valid_X)
# %%
