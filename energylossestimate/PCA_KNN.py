from energylossestimate.PCA_KNN_utils import distribute_events_to_energy_bins, get_data_matrix, get_random_curve, process_curve, kev_to_gev
from showerProfileUtils import parseTrainingData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# --- CLASSIFICATION BASED ON PCA + KNN ---

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
# mean_vector = np.mean(data_matrix, axis=0)
# demeaned_matrix = data_matrix - mean_vector

# 2: SVD
# U, S, Vt = np.linalg.svd(demeaned_matrix)
# plt.stem(S)
# plt.show()

# 3: SkLearn PCA
# scaler = MinMaxScaler()
# data_rescaled = scaler.fit_transform(data_matrix)
pca = PCA(n_components=0.95) # enough components for 95% explained variance
pca_matrix = pca.fit_transform(data_matrix)

# Visualize explained variance
explained_variance = pca.explained_variance_ratio_
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
plt.show()

# 4: Calculate centroids for each energy range
n_rows_per_range = 250
energy_ranges = np.arange(0, 5.012, 0.0895)

centroids = []
for i in range(0, len(pca_matrix), n_rows_per_range):
    centroid = np.mean(pca_matrix[i:i+n_rows_per_range, :], axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)

# # 5: Plot the centroids
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
# selected_columns = [0, 2, 4, 10] # Strongest Principal Components: 1, 3, 5, 11
selected_centroids = centroids # centroids[:, selected_columns]
selected_pca_matrix = pca_matrix # pca_matrix[:, selected_columns]

# Step 3: Classifier Training
knn = KNeighborsClassifier(n_neighbors=3)
# Convert energy ranges to string labels
energy_labels = [f"{energy_ranges[i]}-{energy_ranges[i] + 0.0895}" for i in range(len(energy_ranges))]
# Fit the k-NN model
knn.fit(selected_centroids, energy_labels)

# Step 4: Validation (using cross-validation)
cv_labels = [f"{energy_ranges[int(i/n_rows_per_range)]}-{energy_ranges[int(i/n_rows_per_range)] + 0.0895}" for i in range(len(pca_matrix))]
cv_scores = cross_val_score(knn, selected_pca_matrix, cv_labels, cv=10)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

# Function to classify a new point using k-NN
def classify_knn(point, pca, knn):
    # scaled_new_point = scaler.fit_transform(point)
    point_transformed = pca.transform(point)
    point_selected = point_transformed # new_point_transformed[0, selected_columns].reshape(1, -1)
    closest_energy_range = knn.predict(point_selected)[0]
    return closest_energy_range

curve = get_random_curve(event_list[100:200], resolution=0.5)
point = [process_curve(28, curve)]
closest_energy_range = classify_knn(point, pca, knn)
print(f"The closest energy range for the new point is {closest_energy_range} GeV. Actual energy = {kev_to_gev(curve.energy)}")
