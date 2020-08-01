import numpy as np
import itertools

radius_default = 25

def adj_helper(i, j, A, types, hits):
    gamma_bool = (types[i] == 'g' and types[j] == 'g')
    compton_bool = (types[i] == 'eg' or types[j] == 'eg')
    if gamma_bool or compton_bool or np.sqrt(np.sum((hits[i] - hits[j]) ** 2)) <= radius_default:
        A[i][j] = A[j][i] = 1



# Padding to maximum dimension
def train_pad_helper(i, train_X, train_Ri, train_Ro, train_y, max_train_hits, max_train_edges):
    train_X[i] = np.pad(train_X[i], [(0, max_train_hits - len(train_X[i])), (0, 0)], mode='constant')
    train_Ri[i] = np.pad(train_Ri[i],
                         [(0, max_train_hits - len(train_Ri[i])), (0, max_train_edges - len(train_Ri[i][0]))],
                         mode='constant')
    train_Ro[i] = np.pad(train_Ro[i],
                         [(0, max_train_hits - len(train_Ro[i])), (0, max_train_edges - len(train_Ro[i][0]))],
                         mode='constant')
    train_y[i] = np.pad(train_y[i], [(0, max_train_edges - len(train_y[i]))], mode='constant')