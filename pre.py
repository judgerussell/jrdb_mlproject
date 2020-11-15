# %%
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


# read in the data

X = pd.read_csv('lish-moa/train_features.csv')
Y = np.array(pd.read_csv('lish-moa/train_targets_scored.csv'))
train_drug = np.array(pd.read_csv('lish-moa/train_drug.csv'))
test_X = (pd.read_csv('lish-moa/test_features.csv'))

# %%
# DATA PREPROCESSING

# replacing categorical values with real values

replace_dict = {"cp_type": {"trt_cp": 0, "ctl_vehicle": 1},
                  "cp_time": {24: 0, 48: 1, 72: 2},
                  "cp_dose": {"D1": 0, "D2": 1}
               }

X.replace(replace_dict, inplace=True)
X = np.array(X)
X = X[:, 1:]

test_X.replace(replace_dict, inplace=True)
test_X = np.array(test_X)
test_X = test_X[:, 1:]

cat = X[:, :3]
real = X[:, 3:]
test_cat = X[:, :3]
test_real = X[:, 3:]




# scaling


scaler = StandardScaler()
quan = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")

quan.fit(real)
scaler.fit(real)
real = quan.transform(real)
real = scaler.transform(real)
test_real = quan.transform(test_real)
test_real = scaler.transform(test_real)

gene = real[:, :772]
test_gene = test_real[:, :772]
cell = real[:, 772:]
test_cell = test_real[:, 772:]


# %%
# PCA performed on gene and cell data

gene_PCA = PCA()
cell_PCA = PCA()

gene_PCA.fit(np.concatenate((gene, test_gene), 1))
cell_PCA.fit(np.concatenate((cell, test_cell), 1))

percent = 0
# finding number of principal components
for i in range(len(gene_PCA.explained_variance_ratio_)):
   percent += gene_PCA.explained_variance_ratio_[i]
   if percent > .95:
      break

n_gene_components = i

percent = 0
# finding number of principal components
for i in range(len(cell_PCA.explained_variance_ratio_)):
   percent += cell_PCA.explained_variance_ratio_[i]
   if percent > .95:
      break

n_cell_components = i

## redoing PCA with optimal number of components
gene_PCA = PCA(n_components=518)
cell_PCA = PCA(n_components=40)

pca_genes = gene_PCA.fit_transform(np.concatenate( (gene, test_gene), 1))
pca_cells = cell_PCA.fit_transform(np.concatenate((cell, test_cell), 1))

print(n_gene_components)
print(n_cell_components)


# %%
# STRATIFIED K FOLD
# add drug id as a feature to stratify in the cross validation
target = np.concatenate((train_drug[:, 1:], Y[:, 1:]), 1)

# multilabel stratified kfold
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=0)

for train_index, test_index in mskf.split(X, target):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   Y_train, Y_test = Y[train_index], Y[test_index]



# %%
