import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier 

species = ['Setosa', 'Versicolor', 'Verginica']
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('IRIS.csv', names=names)

# shape
print(dataset.shape)

# head
print('First 20 Observations')
print(dataset.head(20))

# Statistical Summary
# Now we can take a look at a summary of each attribute.
# This includes the count, mean, the min and max values as well as some percentiles.

# descriptions
print('\nStatistical Summary')
print(dataset.describe())

# Class Distribution
# Let’s now take a look at the number of instances (rows) that belong to each class. 
# We can view this as an absolute count.

# class distribution
print('\nClass Distribution')
print(dataset.groupby('class').size())

scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]

validation_size = 0.20
seed = 7

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

kn = KNeighborsClassifier(n_neighbors=1) 
kn.fit(X_train, y_train) 

x_new = np.array([[5, 2.9, 1, 0.2],[4, 2.0, 5, 3.2]]) 
prediction = kn.predict(x_new) 
print(prediction)
print("Predicted target value: {}\n".format(prediction)) 
print("Test score: {:.2f}".format(kn.score(X_test, y_test))) 