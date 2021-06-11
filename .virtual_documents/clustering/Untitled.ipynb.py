import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


data = pd.read_csv("Wholesale customers data.csv")


data.info()


data.isnull().sum()


data.head(10)


data.describe()


data_norm = normalize(data)
data_norm = pd.DataFrame(data_norm, columns = data.columns)


data_norm.info()


data_norm.describe()


data_norm.head(10)


plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method = 'ward'))


plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method = 'ward'))
plt.axhline(y=6, color='r', linestyle='--')


cluster = AgglomerativeClustering(n_clusters = 2,
                                 affinity = 'euclidean',
                                 linkage = 'ward')
cluster.fit_predict(data_scaled)


plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c = cluster.labels_)


cols = list( data_scaled.columns )
n = len(cols)
fig, plots =  plt.subplots(n,n, figsize=(20,20))
fig.tight_layout(h_pad=5, w_pad=5)
for x in range(n):
    for y in range(n):
        label = cols[x] + "vs" + cols[y]
        plots[x][y].set_title(label)
        plots[x][y].scatter(data_scaled[cols[x]], data_scaled[cols[y]], c = cluster.labels_)
plt.show() 



