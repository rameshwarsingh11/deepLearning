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