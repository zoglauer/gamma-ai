from PCA_KNN_utils import distribute_events_to_energy_bins, get_data_matrix, process_curve, kev_to_gev, create_curves, energy_box_plot
from showerProfileUtils import parseTrainingData
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

"""
Aditya, you may want to edit the distribute_events_to_energy_bins function or add a similar one that does logarithmic bins. 
You'll have to make additional changes to this file to make it all work (e.g. line 31)

Ethan, to remove the outlier curves you'll have to look around in the get_data_matrix function. Start there and explore 
EnergyLossDataProcessing.py. Good luck :). By the way, if you want to 3d plot the outlier events, you can use showPlot() from 
EnergyLossDataProcessing.py or your own code. 
"""

# --- PARAMETERS ---
resolution = 0.05 # in radiation lengths
features = int(14 / resolution)

# --- EVENT TO BINS ---
training_events = parseTrainingData()
energy_resolution = 0.0895
training_dict = distribute_events_to_energy_bins(training_events, energy_resolution)

# --- TRAINING DATA MATRIX GENERATION ---
training_data_matrix = get_data_matrix(should_load=False, 
                                       file_path= f'curve_matrix_100K_gamma_fit_{features}_features.csv', 
                                       event_dict=training_dict, 
                                       curves_per_range=250, 
                                       curve_resolution=resolution) 
# UNCOMMENT TO LOAD MATRIX AFTER GENERATION: training_data_matrix = get_data_matrix(should_load=True, file_path='100K_matrix_res_one_fifth_x0.csv')

# --- PCA ---
pca = PCA(n_components=0.98)
pca_matrix = pca.fit_transform(training_data_matrix)

# --- CENTROIDS --- 
n_rows_per_range = 250
energy_ranges = np.arange(0, 5.012, 0.0895)

centroids = []
for i in range(0, len(pca_matrix), n_rows_per_range):
    centroid = np.mean(pca_matrix[i:i+n_rows_per_range, :], axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)

# PLOT CENTROIDS
for i in range(centroids.shape[1]):
    plt.figure()
    plt.scatter(energy_ranges, centroids[:, i])
    plt.title(f"Centroid of PCA Column {i+1}")
    plt.xlabel("Energy Range (GeV)")
    plt.ylabel(f"Centroid Value")
    plt.grid(True)
    plt.show()
    
"""I have left out the KNN / curve comparison part for now, since we'll do it at our integration meeting."""
