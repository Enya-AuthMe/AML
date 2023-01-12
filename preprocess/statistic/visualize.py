import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA


trainset_path = './Class_Dataset/trainset/trainset_ex1.csv'
testset_path = './Class_Dataset/testset/testset_ex1.csv'
df = pd.read_csv(trainset_path)  # 22744
df_test_org = pd.read_csv(testset_path)
df_test = df_test_org.drop(['alert_key'], axis=1, inplace=False)

X = df.loc[:, df.columns != 'sar']
Y = df.loc[:, df.columns == 'sar']

breakpoint()
# # t-SNE
# tsne = manifold.TSNE(n_components=3, init='random', random_state=5, verbose=1)
# X_tsne = tsne.fit_transform(X_train)
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)
# fig = plt.figure(figsize=(6, 6))
# tsne_plt = fig.add_subplot(211, projection='3d')
# tsne_plt.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], linewidths=1,
#                  alpha=.7, edgecolors='k', s=200, c=Y_train.values.tolist())

# PCA
pca = PCA(n_components=2, random_state=9527)
X_pca = pca.fit_transform(X)
fig = plt.figure(figsize=(6, 6))
pca_plt = fig.add_subplot(111)
pca_plt.scatter(X_pca[:, 0], X_pca[:, 1],  linewidths=1,
                alpha=.7, edgecolors='k', s=200, c=Y.values.tolist())
breakpoint()