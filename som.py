#Self Organising map
#Install Minison before running the file
#!pip install Minisom
#!pip show minisom
#Run code on google colab notebook for minimal installation needs.

#unsupervised deep learning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom
from pylab import bone,pcolor,colorbar, plot, show

#import the data set
dataset = pd.read_csv('./sample_data/Credit_Card_Applications.csv')

X = dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))

X = sc.fit_transform(X)

#Training the SOM
som = MiniSom(x=10,y=10,input_len=15,sigma = 1.0, learning_rate = 0.5) 

som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#Visualization of the self organising map
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
  w = som.winner(x)
  plot(w[0]+ 0.5, w[1]+ 0.5, markers[y[i]], markeredgecolor= colors[y[i]],markerfacecolor='None', markersize = 10, markeredgewidth = 2 )
show()

#Detect data sets for fraud/outliers
mappings = som.win_map(X) 
frauds = np.concatenate((mappings[(4,4)], mappings[(5,2)]), axis =0)
#Inverse scaling
frauds = sc.inverse_transform(frauds)

#the frauds object contain multiple lists of customers who somehow misrepresented their credit card application.
# frauds object can be inspected further to drill down the actula customer ids.





