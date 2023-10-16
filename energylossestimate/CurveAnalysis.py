from ShowerProfileUtils import parseTrainingData
from EnergyLossDataProcessing import toDataSpace, zBiasedInlierAnalysis, plot_3D_data, discretize_energy_deposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from Curve import Curve
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import matplotlib.cm as cm
import pandas as pd
import os.path
import csv


def gev_to_kev(gev):
    return gev * (10 ** 6)

def create_curves(event_list: list) -> list:
    
    resolution = 1.0 # essentially the bin size
    curves = []
    
    num_curves = 400 # number of curves to be used for analysis

    index = 0
    while len(curves) < num_curves:
        
        event = event_list[index]
        
        data = toDataSpace(event)
        inlierData, outlierData = zBiasedInlierAnalysis(data)

        if inlierData is not None and len(inlierData > 20):

            t_expected, dEdt_expected = discretize_energy_deposition(inlierData, resolution)
            gamma_energy = event.gamma_energy
            curve = Curve.fit(t_expected, dEdt_expected, gamma_energy, resolution)

            if curve is not None:
                curves.append(curve)
        
        index += 1
        
    print(len(curves))
    
    return curves

def plot_curves(ax, data_matrix_slice, threshold=1000):
    # Filter out low values based on the threshold
    t_values_filtered = np.tile(np.arange(data_matrix_slice.shape[1]), (data_matrix_slice.shape[0], 1))
    dEdt_values_filtered = np.where(data_matrix_slice >= threshold, data_matrix_slice, 0)
    
    # Prepare data for KDE
    t_values = t_values_filtered[dEdt_values_filtered >= threshold]
    dEdt_values = dEdt_values_filtered[dEdt_values_filtered >= threshold]
    data = np.vstack([t_values.flatten(), dEdt_values.flatten()])
    
    # Compute KDE
    kde = gaussian_kde(data, bw_method=0.2)
    
    # Create grid for KDE
    t_grid, dEdt_grid = np.meshgrid(np.linspace(0, 15, 200), np.linspace(0, 140000, 200))
    grid_coords = np.vstack([t_grid.ravel(), dEdt_grid.ravel()])

    # Evaluate KDE on the grid
    kde_values = kde(grid_coords).reshape(t_grid.shape)
    
    # Plot KDE
    ax.imshow(kde_values, origin='lower', cmap='inferno', aspect='auto',
              extent=[0, 15, 0, 140000], alpha=1)
    
    # Plot the original curves
    for row in data_matrix_slice:
        ax.plot(np.arange(len(row)), row, alpha=0.1, color='white')
    
    ax.set_xlabel('Penetration (t)')
    ax.set_ylabel('Energy Deposition (KeV)')
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 140000])
    ax.grid(True)
    
def create_data_matrix(all_curves):
    height = len(all_curves)
    max_bins = 14 # coverage of penetration depth for all energy ranges
    data_matrix = np.zeros((height, max_bins))
    for i in range(height):
        row = list(all_curves[i].dEdt[:max_bins])
        row = [max(0, value) for value in row]
        row.extend([0 for _ in range(14 - len(row))])
        data_matrix[i] = row
    return data_matrix

def save(data_matrix):
    headers = [f'{i}' for i in range(len(data_matrix[0]))]
    
    with open('shower_profile.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        # Write each row in the matrix to the CSV
        for row in data_matrix:
            csvwriter.writerow(row)

def load(filename):
    return np.loadtxt(filename, delimiter=',')[:-1,:]

event_list = parseTrainingData()
energy_list = [event.gamma_energy for event in event_list]

# Box Plot of Energies for 10k dataset
"""
plt.figure(figsize=(10, 6))
plt.boxplot(energy_list, vert=True, patch_artist=True)
plt.title('Incident Energy of 10k Events')
avg = np.mean(energy_list)
sd = np.std(energy_list)
print(f'average energy: {avg}, standard deviation: {sd}')
plt.show()
"""

# --- ANALYZE THE CURVES BETWEEN 0 AND 1 GEV, 1 AND 2 GEV, ... 5 AND 6 GEV ---
# PCA is a means to an end to compare the curves for the shower profile.

should_load = True

file_path = 'shower_profile.csv'
if should_load and os.path.exists(file_path):
    data_matrix = load(file_path)
    
    # Plot Curves
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    rows_per_range = 400
    energy_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    for i, (low_E, high_E) in enumerate(energy_ranges):
        row_idx = i // 3
        col_idx = i % 3
        ax = axs[row_idx, col_idx]
        start_idx = i * rows_per_range
        end_idx = (i + 1) * rows_per_range
        plot_curves(ax, data_matrix[start_idx:end_idx])
        ax.set_title(f'Energies: {low_E}-{high_E} GeV')
    
    plt.tight_layout()
    plt.show()
else:

    # Generate Event Lists
    # [event.gamma_energy] = KeV
    zero_to_one_mev_events = [event for event in event_list if gev_to_kev(0) <= event.gamma_energy < gev_to_kev(1)]
    one_to_two_mev_events = [event for event in event_list if gev_to_kev(1) <= event.gamma_energy < gev_to_kev(2)]
    two_to_three_mev_events = [event for event in event_list if gev_to_kev(2) <= event.gamma_energy < gev_to_kev(3)]
    three_to_four_mev_events = [event for event in event_list if gev_to_kev(3) <= event.gamma_energy < gev_to_kev(4)]
    four_to_five_mev_events = [event for event in event_list if gev_to_kev(4) <= event.gamma_energy < gev_to_kev(5)]
    
    # Generate Curves (shower analysis)
    zero_to_one_mev_curves = create_curves(zero_to_one_mev_events)
    one_to_two_mev_curves = create_curves(one_to_two_mev_events)
    two_to_three_mev_curves = create_curves(two_to_three_mev_events)
    three_to_four_mev_curves = create_curves(three_to_four_mev_events)
    four_to_five_mev_curves = create_curves(four_to_five_mev_events)

    all_curves = zero_to_one_mev_curves + one_to_two_mev_curves + two_to_three_mev_curves + three_to_four_mev_curves+ four_to_five_mev_curves

    # create data matrix
    data_matrix = create_data_matrix(all_curves)

    # Save Curve Data
    save(data_matrix)

# --- Manual PCA --- 

# 1: Demean data matrix
mean_vector = np.mean(data_matrix, axis=0) # mean of each feature (row)
demeaned_matrix = data_matrix - mean_vector

# 2: SVD
U, S, Vt = np.linalg.svd(demeaned_matrix)
# plt.stem(S) # --> 3 singular values should be good

# 3: New Basis
new_basis = np.transpose(Vt)[:, 0:3]
# plt.plot(new_basis)
# plt.show()

# 4: Project data onto New Basis
proj = (demeaned_matrix @ new_basis)

# 5: View clusters
cumulative_counts = [0, 400, 800, 1200, 1600, 2000] # np.cumsum([0, len(zero_to_one_mev_curves), len(one_to_two_mev_curves), len(two_to_three_mev_curves), len(three_to_four_mev_curves), len(four_to_five_mev_curves)])

labels = ["zero_to_one", "one_to_two", "two_to_three", "three_to_four", "four_to_five"]
colors = ['r', 'g', 'b', 'y', 'm'] 

# 3D View
fig = plt.figure(figsize=(10, 5))
ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
for i in range(len(labels)):
    start_idx, end_idx = cumulative_counts[i], cumulative_counts[i+1]
    Axes3D.scatter(ax, *proj[start_idx:end_idx].T, c=colors[i], marker='o', s=20)
plt.legend(labels, loc='center left', bbox_to_anchor=(1.07, 0.5))

# Side Views
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i in range(len(labels)):
    start_idx, end_idx = cumulative_counts[i], cumulative_counts[i+1]
    axs[0].scatter(proj[start_idx:end_idx, 0], proj[start_idx:end_idx, 1], c=colors[i], edgecolor='none')
    axs[1].scatter(proj[start_idx:end_idx, 0], proj[start_idx:end_idx, 2], c=colors[i], edgecolor='none')
    axs[2].scatter(proj[start_idx:end_idx, 1], proj[start_idx:end_idx, 2], c=colors[i], edgecolor='none')
axs[0].set_title("View 1")
axs[1].set_title("View 2")
axs[2].set_title("View 3")
plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# 6: Find and display centroids
fig = plt.figure()
ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
for i in range(len(labels)):
    start_idx, end_idx = cumulative_counts[i], cumulative_counts[i+1]
    x_avg = np.mean(proj[start_idx:end_idx, 0])
    y_avg = np.mean(proj[start_idx:end_idx, 1])
    z_avg = np.mean(proj[start_idx:end_idx, 2])
    ax.scatter(x_avg, y_avg, z_avg, c=colors[i], marker='o', s=20, label=labels[i])
plt.legend(labels, loc='center left', bbox_to_anchor=(1.07, 0.5))
plt.show()

# -- END Manual PCA --

# # -- SKLEARN PCA -- 

# """scalar = sklearn.preprocessing.StandardScaler()
# scalar.fit(demeaned_matrix)
# scaled_matrix = scalar.transform(demeaned_matrix) #<-- use or don't use? probably don't"""

# #preform PCA
# pca_3 = sklearn.decomposition.PCA(n_components = 3, random_state = 2023)
# pca_3.fit(demeaned_matrix) #<-- could use scaled matrix here
# demeaned_pca_3 = pca_3.transform(demeaned_matrix)

# #initialize & add color coding to PCA
# colors = [0]*cumulative_counts[5] 
# num_bins = 5 #number of bins to seperate colors into
# color_strings = ['red', 'orange', 'green', 'blue', 'purple']
# counter = 0
# for i in range(num_bins):
#     for j in range(int(len(colors)/num_bins)):
#         colors[counter] = color_strings[i]
#         counter += 1
# colors += [color_strings[4]] #add 2001th (last) index from demeaned_matrix, scuffed

# #plot PCA
# ax = Axes3D(fig)
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter3D(demeaned_pca_3[:,0], demeaned_pca_3[:,1], demeaned_pca_3[:,2], s=50, alpha=0.6, c=colors)
# ax.set_title('PCA - SKLEARN')
# plt.show()

# #create PCA averages
# avg_matrix = []
# for i in range(num_bins):
#     interval = cumulative_counts[1] #really, should be num_curves in create curves function
#     range1 = i*interval
#     range2 = range1+interval
#     x = np.matrix(demeaned_matrix[range1:range2,0])
#     y = np.matrix(demeaned_matrix[range1:range2,1])
#     z = np.matrix(demeaned_matrix[range1:range2,2])
#     avg_values = [x.mean(), y.mean(), z.mean()]
#     avg_matrix.append(avg_values)
# avg_matrix = np.array(avg_matrix) #scuffed, need for slicing in plot

# #plot PCA averages
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter3D(avg_matrix[:,0], avg_matrix[:,1], avg_matrix[:,2], s=50, alpha=0.6, c=color_strings)
# ax.set_title('Average Values from PCA - SKLEARN')
# plt.show()

# # -- END SKLEARN PCA --