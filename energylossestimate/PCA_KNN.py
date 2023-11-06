from PCA_KNN_utils import distribute_events_to_energy_bins, get_data_matrix, process_curve, kev_to_gev, create_curves, energy_box_plot
from showerProfileUtils import parseTrainingData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter

# --- CLASSIFICATION BASED ON PCA + KNN ---

verbose = False

# 1: What is the highest resolution (100k training set) achievable such that each bin has at least K events?
training_events = parseTrainingData()
energy_resolution = 0.0895 # [] = GeV, lower number = higher resolution
K = 1000 # minimum events per range
training_dict = distribute_events_to_energy_bins(training_events, energy_resolution)

if all([len(training_dict[energy_range]) > K for energy_range in training_dict.keys()]):
    print('Up the resolution.')
elif any([len(training_dict[energy_range]) == K for energy_range in training_dict.keys()]):
    print(f'Best resolution for {K} events per bin.')
else:
    print('Resolution too high.')
    
if verbose:
    print(' \n'.join([f'{energy_range}GeV to {energy_range + energy_resolution}GeV : {len(training_dict[energy_range])}' for energy_range in training_dict.keys()]))

# 2: Use PCA to determine how distinct each energy range is.
resolution = 0.00005 # in radiation lengths
components = int(14 / resolution)
# training_data_matrix = get_data_matrix(should_load=True, file_path='100K_matrix_res_one_fifth_x0.csv')
training_data_matrix = get_data_matrix(should_load=False, file_path= 'curve_matrix_100K_gamma_fit_28_features.csv', event_dict=training_dict, curves_per_range=K // 4, curve_resolution=resolution) 
pca = PCA(n_components=components) # set to 0.95 for enough components for 95% explained variance
pca_matrix = pca.fit_transform(training_data_matrix)

# Visualize explained variance
"""explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Explained Variance per Principal Component')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')

plt.tight_layout()
plt.show()"""

# 4: Calculate centroids for each energy range
n_rows_per_range = 250
energy_ranges = np.arange(0, 5.012, 0.0895)

centroids = []
for i in range(0, len(pca_matrix), n_rows_per_range):
    centroid = np.mean(pca_matrix[i:i+n_rows_per_range, :], axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)

# 5: Plot the centroids
# for i in range(centroids.shape[1]):
#     plt.figure()
#     plt.scatter(energy_ranges, centroids[:, i])
#     plt.title(f"Centroid of PCA Column {i+1}")
#     plt.xlabel("Energy Range (GeV)")
#     plt.ylabel(f"Centroid Value")
#     plt.grid(True)
#     plt.show()
    
# --- CLASSIFICATION ---

# Step 1 & 2: Feature Selection and Data Preparation
selected_columns = [0, 2, 4, 11] # Strongest Principal Components: 1, 3, 5, 12
selected_centroids = centroids[:, selected_columns]
selected_pca_matrix = pca_matrix[:, selected_columns]

# Step 3: Classifier Training
knn = KNeighborsClassifier(n_neighbors=3)
energy_labels = [f"[{round(energy_ranges[i], 4)}, {round(energy_ranges[i] + 0.0895, 4)}]" for i in range(len(energy_ranges))]
knn.fit(selected_centroids, energy_labels)

def classify_knn(point, pca, knn):
    point_transformed = pca.transform(point)
    point_selected = point_transformed[:, selected_columns]
    closest_energy_range = knn.predict(point_selected)[0]
    return closest_energy_range

def classify_knn_top_k(point, pca, knn, selected_columns, labels, k=2):
    # Transform the point using PCA
    point_transformed = pca.transform(point)  # Ensure point is a 2D array
    point_selected = point_transformed[:, selected_columns]
    
    # Find the k nearest neighbors
    distances, indices = knn.kneighbors(point_selected, n_neighbors=3)
    
    # Flatten the indices array to use it for indexing labels
    neighbor_indices = indices.flatten().astype(int)
    
    # Get the labels of the nearest neighbors
    neighbor_labels = [labels[idx] for idx in neighbor_indices]
    
    # Count the labels to find the top k
    label_counts = Counter(neighbor_labels)
    top_k_labels = label_counts.most_common(k)
    
    return top_k_labels

def correct(range_str, value):
    low, high = eval(range_str)
    return low <= value <= high

def close(range_str, value, cushion):
    low, high = eval(range_str)
    return low - cushion <= value <= high + cushion

# Step 4: Validation (using cross-validation)
"""cv_labels = [f"[{energy_ranges[int(i/n_rows_per_range)]}, {energy_ranges[int(i/n_rows_per_range)] + 0.0895}]" for i in range(len(pca_matrix))]
cv_scores = cross_val_score(knn, pca_matrix, cv_labels, cv=10)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))"""

# --- TEST DATASET ---
test_events = parseTrainingData(training_file='EnergyLoss.10k.v1.data')
test_curves = create_curves(test_events, resolution=resolution, num_curves=1000)
# energy_box_plot(test_curves) # show a box plot of the energies
processed_test_curves = list(map(lambda x: process_curve(components, x), test_curves))
correct_count = 0
close_count = 0
loose_count = 0
for i in range(len(test_curves)):
    closest_energy_range = classify_knn([processed_test_curves[i]], pca, knn) # closest_energy_ranges = classify_knn_top_k([processed_test_curves[i]], pca, knn, selected_columns, energy_labels, k=3) 
    if correct(closest_energy_range, kev_to_gev(test_curves[i].energy)):
        correct_count += 1
    if close(closest_energy_range, kev_to_gev(test_curves[i].energy), 0.0895):
        close_count += 1
    if close(closest_energy_range, kev_to_gev(test_curves[i].energy), 0.455):
        loose_count += 1
print(correct_count, close_count, loose_count)

"""
--- PART I

56 energy ranges of size 0.0895GeV. 
Experiment: (uniformly) randomly pick one of 56 ranges to classify an event as. Classify n events. X = random variable counting the number of
correctly classified events. 
Define X_i as an indicator of whether classification i is correct. 
E[X] = E[X_1 + X_2 + ... X_n] = nE[X_i]
E[X_i] = 1(1/56) + 0(55/56) = 1/56
E[X] = n/56. 
For n=1000, we expect to classify around 18 events correctly, by purely random guessing.

--- PART II 

With our model;
- 4D polynomial fit
- 250 curves per energy range 
- resolution: 0.5 radiation lengths

On 1000 events we find:
40 correctly classified within 0.0895GeV (Random Expected = 18)
89 correctly classified within 0.2685GeV
254 correctly classified within 1GeV 

--- PART III

Different model:
- 4D polynomial fit
- 250 curves per energy range
- resolution: 1 radiation length

On 1000 events we find:
38 correctly classified within 0.0895GeV (Random Expected = 18)
88 correctly classified within 0.2685GeV
255 correctly classified within 1GeV 

--- PART IV

Different model:
- 4D polynomial fit
- 250 curves per energy range
- resolution: 0.2 radiation lengths

On 1000 events we find:
37 correctly classified within 0.0895GeV (Random Expected = 18)
92 correctly classified within 0.2685GeV
264 correctly classified within 1GeV 
"""
