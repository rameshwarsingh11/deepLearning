# Recommendation system with the Boltzmann Machines
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import variable

# Dataset import
movies = pd.read_csv('sample_data/movies.dat',sep='::', header=None, engine = 'python', encoding='latin-1')

users = pd.read_csv('sample_data/users.dat',sep='::', header=None, engine = 'python', encoding='latin-1')

ratings = pd.read_csv('sample_data/ratings.dat',sep='::', header=None, engine = 'python', encoding='latin-1')

#Adding training and test sets
training_set = pd.read_csv('sample_data/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype= 'int')

test_set= pd.read_csv('sample_data/u1.test', delimiter='\t')
test_set=np.array(test_set, dtype='int')

nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Data conversion into Torch compatible data format
def convert(data):
  new_data = []
  for id_users in range(1,nb_users+1):
    id_movies = data[:,1][data[:,0]==id_users]
    id_ratings = data[:,2][data[:,0]==id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies-1] = id_ratings
    new_data.append(list(ratings))
  
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)

training_set = torch.FloatTensor(training_set)
test_set  = torch.FloatTensor(test_set)

# Data conversion to Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Like vs Dislike ratings conversion
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#Adding neural network structure for Restricted Boltzmann Machine
class RBM():
  def __init__(self,nv,nh):
    self.W = torch.randn(nh,nv)
    self.a = torch.randn(1, nh) #1 is added to make it 2D structure.
    self.b = torch.randn(1,nv)


  #sampling the hidden nodes according to the probability of nh and nv using sigmoid activation function
  def sample_h(self, x):
    wx = torch.mm(x, self.W.t())
    activation = wx + self.a.expand_as(wx)
    p_h_given_v = torch.sigmoid(activation)
    return p_h_given_v, torch.bernoulli(p_h_given_v)


  def sample_v(self, y):
    wy = torch.mm(y, self.W)
    activation = wy + self.b.expand_as(wy)
    p_v_given_h = torch.sigmoid(activation)
    return p_v_given_h, torch.bernoulli(p_v_given_h) 


  # Contrastive Divergence for Log-Likelihood gradient calculation
  # k : round trips/iterations
  # ph0 : initial probability
  def train(self, v0, vk,ph0, phk):
    self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
    self.b += torch.sum((v0-vk),0)
    self.a += torch.sum((ph0-phk),0)

nv = len(training_set[0])
nh = 100 #nh: number of hidden nodes calculating number of features.
batch_size = 100

rbm = RBM(nh,nv)

# Training the RBM
nb_epoch = 10
# After completion of this loop we will get ratings of the movies which were not earlier rated:
for epoch in range(1, nb_epoch+1):
  #Comparing the predictive and real ratings.
  #Find the loss measures
  train_loss = 0
  s = 0.
  for id_user in range(0,nb_users - batch_size, batch_size):
    vk = training_set[id_user:id_user+batch_size]
    v0 = training_set[id_user:id_user+batch_size]
    ph0,_ = rbm.sample_h(v0.t())

    for k in range(10): # k-steps in the Contrastive Divergence
      _,hk = rbm.sample_h(vk)
      _,vk = rbm.sample_v(hk)
      vk[v0<0] = v0[v0<0] # No need to count where user hasn't rated the movie
    
    phk,_ = rbm.sample_h(vk)
    rbm.train(v0,vk,ph0,phk)# Training will adjust the weights to update the biases
    train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[vk>=0])) # Applying error
    s +=1.
  print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))# train_loss/s will normalize the train_loss

# Testing the RBM machine code
test_loss = 0
s = 0.

for id_user in range(nb_users):
  v = training_set[id_user:id_user+1]
  vt = test_set[id_user:id_user+1]

  if len(vt[vt>=0]) > 0:
    _,h = rbm.sample_h(v)
    _,v = rbm.sample_v(h)
    train_loss += torch.mean(torch.abs(vt[vt>=0]-v[v0>=0])) # Applying error
    s +=1.
print('test loss: '+str(test_loss/s))
  