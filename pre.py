# %%
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read in the data

X = pd.read_csv('lish-moa/train_features.csv')
Y = (pd.read_csv('lish-moa/train_targets_scored.csv'))
train_drug = np.array(pd.read_csv('lish-moa/train_drug.csv'))
test_X = pd.read_csv('lish-moa/test_features.csv')
non_scored = np.array(pd.read_csv('lish-moa/train_targets_nonscored.csv'))

# DATA PREPROCESSING
# %%
# replacing categorical values with real values

replace_dict = {"cp_type": {"trt_cp": 0, "ctl_vehicle": 1},
                  "cp_time": {24: 0, 48: 1, 72: 2},
                  "cp_dose": {"D1": 0, "D2": 1}
               }

X.replace(replace_dict, inplace=True)
X = np.array(X)
X = X[:, 1:]

test_X.replace(replace_dict, inplace=True)


test_samples = test_X.index.values
targets = Y.columns.values[1:]

test_X = np.array(test_X)
test_samples = test_X[:, 0]
test_X = test_X[:, 1:]

non_scored = non_scored[:,1:]
Y = np.array(Y)
Y = Y[:, 1:]


train_drug = train_drug[:, 1:]

X = X.astype(np.double)
Y = Y.astype(np.double)
test_X = test_X.astype(np.double)
non_scored = non_scored.astype(np.double)

# %%
# kmeans clustering on non-scored targets, add cluster value in as a feature
"""
# this was for testing different values of k
non_scored = non_scored[:,1:]
silhouette = []
kmax = 250

for k in range(245, kmax):
   kmeans = KMeans(n_clusters=k)
   kmeans.fit(non_scored)
   labels = kmeans.labels_
   silhouette.append(silhouette_score(non_scored, labels, metric='euclidean'))

print(silhouette)
plt.axes().set_xlim([245, kmax])
plt.plot(range(245,kmax), silhouette)
plt.show()
"""
#kmeans = KMeans(n_clusters=160)
#kmeans.fit(non_scored)
#cluster_labels = np.reshape(np.array(kmeans.labels_), (1, len(kmeans.labels_)))

#cat = np.concatenate((X[:, :3], cluster_labels.T), 1)
cat = X[:, :3]
real = X[:, 3:]
test_cat = test_X[:, :3]
test_real = test_X[:, 3:]



# %%
# feature selection: variance threshold

v = VarianceThreshold(threshold=0.1)
real = v.fit_transform(real)

v = VarianceThreshold(threshold=0.1)
test_real = v.fit_transform(test_real)


# %%
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

gene_PCA.fit(np.concatenate((gene, test_gene), 0))
cell_PCA.fit(np.concatenate((cell, test_cell), 0))

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


print(n_gene_components)
print(n_cell_components)

# redoing PCA with optimal number of components
gene_PCA = PCA(n_components=555)
cell_PCA = PCA(n_components=40)

pca_genes = gene_PCA.fit_transform(np.concatenate( (gene, test_gene), 0))
pca_cells = cell_PCA.fit_transform(np.concatenate((cell, test_cell), 0))

train_X = np.concatenate((cat, pca_genes[:len(X)], pca_cells[:len(X)]), 1)
test_X = np.concatenate((test_cat, pca_genes[len(X):], pca_cells[len(X):]), 1)

# %%
# writing to file

print(test_X.shape)
np.save("preprocessed/x.npy", train_X)
np.save("preprocessed/y.npy", Y)
np.save("preprocessed/test_X.npy", test_X)
np.save("preprocessed/train_drug.npy", train_drug)
np.save("targets.npy", targets)
np.save("test_samples.npy", test_samples)

# %%
