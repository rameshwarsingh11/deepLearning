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

