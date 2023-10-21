import sys
sys.path.append('/Volumes/T7/COSI Research/gamma-ai/energylossestimate')

from utils import distribute_events_to_energy_bins, get_data_matrix
from showerProfileUtils import parseTrainingData
import numpy as np
import matplotlib.pyplot as plt

# --- ASSESSING THE VIABILITY OF POINTNET ---

verbose = False

# 1: What is the highest resolution (100k training set) achievable such that each bin has at least K events?
event_list = parseTrainingData()
energy_resolution = 0.0895 # [] = GeV, lower number = higher resolution
K = 1000 # minimum events per range
events = distribute_events_to_energy_bins(event_list, energy_resolution)

if all([len(events[energy_range]) > K for energy_range in events.keys()]):
    print('Up the resolution.')
elif any([len(events[energy_range]) == K for energy_range in events.keys()]):
    print(f'Best resolution for {K} events per bin.')
else:
    print('Resolution too high.')
    
if verbose:
    print(' \n'.join([f'{energy_range}GeV to {energy_range + energy_resolution}GeV : {len(events[energy_range])}' for energy_range in events.keys()]))

# 2: Use PCA to determine how distinct each energy range is.
data_matrix = get_data_matrix(should_load=True, file_path='shower_profile.csv') # get_data_matrix(load=False, event_dict=events, curves_per_range=K // 4, curve_resolution=0.5)

# 1: Demean data matrix
mean_vector = np.mean(data_matrix, axis=0)
demeaned_matrix = data_matrix - mean_vector

# 2: SVD
U, S, Vt = np.linalg.svd(demeaned_matrix)
plt.stem(S) # --> 3 singular values should be good
plt.show()
