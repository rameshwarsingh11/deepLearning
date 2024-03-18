# This solution will convert an unsupervised output from a Self Organized map to supervised output with ANN to detect customer frauds in a ranking order
# Install Minison before running the file
#!pip install Minisom
#!pip show minisom
# Run code on google colab notebook for minimal installation needs.

# unsupervised deep learning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# import the data set
dataset = pd.read_csv("./sample_data/Credit_Card_Applications.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))

X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)

som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualization of the self organizing map
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ["o", "s"]
colors = ["r", "g"]
for i, x in enumerate(X):
    w = som.winner(x)
    plot(
        w[0] + 0.5,
        w[1] + 0.5,
        markers[y[i]],
        markeredgecolor=colors[y[i]],
        markerfacecolor="None",
        markersize=10,
        markeredgewidth=2,
    )
show()

# Detect data sets for fraud/outliers
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1, 2)], mappings[(1, 1)]), axis=0)
# Inverse scaling
frauds = sc.inverse_transform(frauds)

# the frauds object contain multiple lists of customers who somehow misrepresented their credit card application.
# frauds object can be inspected further to drill down the actual customer ids.

# Create matrix features
customers = dataset.iloc[:, 1:].values
customers

# Create dependent variable
# 0 = fraud, 1 = not fraud

# Create a vector with 0 values
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1


# Train the ANN

# Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

customers = sc.fit_transform(customers)

from keras.models import Sequential

from keras.layers import Dense

# ANN initialization
classifier = Sequential()
# Adding the input layer and first hidden layer
classifier.add(
    Dense(units=2, kernel_initializer="uniform", activation="relu", input_dim=15)
)

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compilation of ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Model fitting
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Making predictions
# Fraud customers prediction
y_pred = classifier.predict(customers)

# Creating the Ranking

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)

# Sort the dataset in descending order of their probability
y_pred = y_pred[y_pred[:, 1].argsort()]

# y_pred is a 2D array containing customer id and their probability of fraud.
