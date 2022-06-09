import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

data = 'Live.csv'

df = pd.read_csv(data)
df.shape
df.head()
df.info()
df.isnull().sum()
df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)
df.info()
df.describe()
df['status_id'].unique()
len(df['status_id'].unique())
df['status_published'].unique()
len(df['status_published'].unique())
df['status_type'].unique()
len(df['status_type'].unique())
df.drop(['status_id', 'status_published'], axis=1, inplace=True)
df.info()
df.head()


X = df

y = df['status_type']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['status_type'] = le.fit_transform(X['status_type'])

y = le.transform(y)
X.info()
X.head()


cols = X.columns
from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])
X.head()


kmeans = KMeans(n_clusters=4, random_state=0)

kmeans.fit(X)

labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Resultado: %d de %d amostras foram rotuladas corretamente." % (correct_labels, y.size))
print('Precisão: {0:0.2f}'. format(correct_labels/float(y.size)))
from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('Elbow Method')
plt.xlabel('Numero de clusters')
plt.ylabel('CS')
plt.show()