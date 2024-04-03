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

# Data convervion to Torch tensors
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

