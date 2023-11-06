from ShowerProfileUtils import parseTrainingData
from EnergyLossDataProcessing import toDataSpace, zBiasedInlierAnalysis, plot_3D_data, discretize_energy_deposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Curve import Curve
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import matplotlib.cm as cm
import pandas as pd
import os.path
import csv
import time

#NOTE: if loading a dataset, SET gev_interval, num_curves, AND training_file in ShowerProfileUtils.py
#to the values used to generate the dataset

gev_interval = 0.5 #must be 5 mod gev_interval = 0 for now (divisible by 1/5) #TODO: 0.2 doesn't work for some reason
num_curves = 100 #number of curves to be used for analysis
base_num_bins = 5 #number of energy ranges, in 1GeV interval (ie. 5 for 0-1, 1-2, 2-3, 3-4, 4-5)
num_bins = base_num_bins/gev_interval #total number of bins for analysis

load_avg_graphs_only = False # <-- toggle to true to display only average graph (to combat long runtimes) #TODO: add other graphs too

def gev_to_kev(gev):
    return gev * (10 ** 6)

#num_curves used inside here
def create_curves(event_list: list) -> list:
    
    resolution = gev_interval # essentially the bin size
    curves = []
    
    index = 0
    while len(curves) < num_curves and len(curves) < len(event_list):
        
        try:
            event = event_list[index]
        except:
            print(len(event_list))
            break #sketchy, here to fix error, not sure why it's here
                #(second part of while statement should've filtered what I think is causing it)
        
        data = toDataSpace(event)

        try:
            inlierData, outlierData = zBiasedInlierAnalysis(data)
        except Exception as e:
            #print(e)
            inlierData = zBiasedInlierAnalysis(data) #this one should be OK
            inlierData = inlierData[0] #comes out as a tuple, get the non-tuple element
            #this catches and fixes potential "too many values to unpack (expected 2)" errors

        if inlierData is not None and len(inlierData > 20):
            t_expected, dEdt_expected = discretize_energy_deposition(inlierData, resolution)
            gamma_energy = event.gamma_energy
            curve = Curve.fit(t_expected, dEdt_expected, gamma_energy, resolution)

            if curve is not None:
                curves.append(curve)
        
        index += 1
    
    return curves

def plot_curves(ax, curves, low_E, high_E):
                    
    for i, curve in enumerate(curves):
        ax.plot(curve.t, curve.dEdt, label=f'Curve {i}')
    
    ax.set_xlabel('Penetration (t)')
    ax.set_ylabel('Energy Deposition (KeV)')
    ax.set_title(f'Energies: {low_E}-{high_E} GeV')
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 150000])
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
        print(len(row))

    savefile_name = 'data_matrix.npy'
    #avg_savefile_name = 'defualt_averages.npy'
    save_to_file = True #default False
    if(save_to_file):
        print("SAVING DATA MATRIX TO FILE...")
        np.save(savefile_name, data_matrix)
        print("AVERAGES SAVED TO FILE:", savefile_name)
        #avg_matrix.tofile(avg_savefile_name, sep = ',')

    print("10 SECOND PAUSE...")
    time.sleep(10)
    return data_matrix

def save(data_matrix):
    headers = [f'{i}' for i in range(len(data_matrix[0]))]

    savefile_name = 'shower_profile_test.csv' # <-- change savefile name here
    
    with open(savefile_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        # Write each row in the matrix to the CSV
        for row in data_matrix:
            csvwriter.writerow(row)

def load(filename):
    return np.loadtxt(filename, delimiter=',')

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

# --- DISPLAY GRAPHS BEFORE COMPUTATION ---
# Workaround measure for long run times, use for display only
# Only for average for now
# TODO: clean this up

if load_avg_graphs_only:
    filename = 'sp_avgs_025int_2000curv_100k.npy'
    avg_matrix = np.load(filename)

    #initialize & add color coding to PCA
    avg_colors = []
    for i in range(len(avg_matrix[:,0])):
        if(np.isnan(avg_matrix[i,0])):
            continue
        else:
            avg_colors.append(i)
    #basically, just list 1,2,3...n to color the points in sequence

    #plot PCA averages
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(avg_matrix[:,0], avg_matrix[:,1], avg_matrix[:,2], s=50, alpha=0.6, c=avg_colors, cmap='rainbow')
    ax.set_title('Average Values from PCA - SKLEARN')
    plt.show()

    exit()

# --- ANALYZE THE CURVES BETWEEN 0 AND 1 GEV, 1 AND 2 GEV, ... 5 AND 6 GEV ---
# PCA is a means to an end to compare the curves for the shower profile.

should_load = False

#file_path = 'shower_profile.csv'
file_path = 'shower_profile_interval05_100_new.csv'
if should_load and os.path.exists(file_path):
    print("LOADING DATA FROM FILE: ", file_path)
    data_matrix = load(file_path)
    print("LOADED 'data_matrix' FROM: ", file_path)
else:

    # Generate Event Lists
    print("GENERATING EVENT LISTS...")
    # [event.gamma_energy] = KeV
    """    zero_to_one_mev_events = [event for event in event_list if gev_to_kev(0) <= event.gamma_energy < gev_to_kev(1)]
        one_to_two_mev_events = [event for event in event_list if gev_to_kev(1) <= event.gamma_energy < gev_to_kev(2)]
        two_to_three_mev_events = [event for event in event_list if gev_to_kev(2) <= event.gamma_energy < gev_to_kev(3)]
        three_to_four_mev_events = [event for event in event_list if gev_to_kev(3) <= event.gamma_energy < gev_to_kev(4)]
        four_to_five_mev_events = [event for event in event_list if gev_to_kev(4) <= event.gamma_energy < gev_to_kev(5)] """
    #generalization of the above
    events = []
    for i in np.arange(0, (1/gev_interval * 5), gev_interval):
        events.append([event for event in event_list if gev_to_kev(i) <= event.gamma_energy < gev_to_kev(i+gev_interval)])
    #potentially inefficient with large datasets

    # Generate Curves (shower analysis)
    print("GENERAING CURVES...")
    """   zero_to_one_mev_curves = create_curves(zero_to_one_mev_events)
        one_to_two_mev_curves = create_curves(one_to_two_mev_events)
        two_to_three_mev_curves = create_curves(two_to_three_mev_events)
        three_to_four_mev_curves = create_curves(three_to_four_mev_events)
        four_to_five_mev_curves = create_curves(four_to_five_mev_events) """
    all_curves = []
    for i in events:
        all_curves = all_curves + (create_curves(i))
        print("curve added") #print(i,"curve added")

    #all_curves = zero_to_one_mev_curves + one_to_two_mev_curves + two_to_three_mev_curves + three_to_four_mev_curves+ four_to_five_mev_curves

    # create data matrix
    print("CREATING DATA MATRIX...")
    data_matrix = create_data_matrix(all_curves)

    # Save Curve Data
    print("SAVING DATA MATRIX...")
    save(data_matrix)
    print("SAVED")

    # Plot Curves
    """ fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        plot_curves(axs[0, 0], zero_to_one_mev_curves, 0, 1)
        plot_curves(axs[0, 1], one_to_two_mev_curves, 1, 2)
        plot_curves(axs[0, 2], two_to_three_mev_curves, 2, 3)
        plot_curves(axs[1, 0], three_to_four_mev_curves, 3, 4)
        plot_curves(axs[1, 1], four_to_five_mev_curves, 4, 5) 

    plt.tight_layout()
    plt.show() """

print("BEGINNING MANUAL PCA")
# --- Manual PCA --- print("BEGINNING MANUAL PCA")
num_bins = 5 #bins for 0-1gev, 1-2gev...

# 1: Demean data matrix
print("DEMEANING MATRIX...")
mean_vector = np.mean(data_matrix, axis=0) # mean of each feature (row)
demeaned_matrix = data_matrix - mean_vector
print(demeaned_matrix.shape)

"""
# 2: SVD
print("PERFORMING SVD...")
U, S, Vt = np.linalg.svd(demeaned_matrix)
# plt.stem(S) # --> 3 singular values should be good

# 3: New Basis
print("TRANSPOSING...")
new_basis = np.transpose(Vt)[:, 0:3]
# plt.plot(new_basis)
# plt.show()

# 4: Project data onto New Basis
print("PROJECTING...")
proj = (demeaned_matrix @ new_basis)

# 5: View clusters
#cumulative_counts = [0, 400, 800, 1200, 1600, 2000] # np.cumsum([0, len(zero_to_one_mev_curves), len(one_to_two_mev_curves), len(two_to_three_mev_curves), len(three_to_four_mev_curves), len(four_to_five_mev_curves)])
cumulative_counts = []
for i in range(num_bins+1):
    cumulative_counts.append(i*num_curves)
#generate cumulative counts

labels = ["zero_to_one", "one_to_two", "two_to_three", "three_to_four", "four_to_five"]
colors = ['r', 'g', 'b', 'y', 'm'] 

fig = plt.figure(figsize=(10, 5))
ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
for i in range(len(labels)):
    start_idx, end_idx = cumulative_counts[i], cumulative_counts[i+1]
    Axes3D.scatter(ax, *proj[start_idx:end_idx].T, c=colors[i], marker='o', s=20)
plt.legend(labels, loc='center left', bbox_to_anchor=(1.07, 0.5))

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
"""
# -- END Manual PCA --
# -- SKLEARN PCA -- 

print("BEGINNING SKLEARN PCA ANALYSIS")

"""scalar = sklearn.preprocessing.StandardScaler()
scalar.fit(demeaned_matrix)
scaled_matrix = scalar.transform(demeaned_matrix) #<-- use or don't use? probably don't"""

#preform PCA
pca_3 = sklearn.decomposition.PCA(n_components = 3, random_state = 2023)
pca_3.fit(demeaned_matrix) #<-- could use scaled matrix here
demeaned_pca_3 = pca_3.transform(demeaned_matrix)

#initialize & add color coding to PCA
num_bins = 5/gev_interval #number of bins to seperate colors into
#colors = [0]*int(num_bins*num_curves)
colors = []
#color_strings = ['red', 'orange', 'green', 'blue', 'purple', 'red', 'orange', 'green', 'blue', 'purple'] #HARDCODED HELP HERE
counter = 0
for i in range(int(num_bins)): #TODO
    for j in range(int(len(colors)/num_bins)):
        #colors[counter] = color_strings[i]
        colors[counter] = i #set index value, per bin, for mapping
        counter += 1
colors += [int(num_bins)] #[color_strings[4]] #add 2001th (last) index from demeaned_matrix, scuffed """

#note for later - c can either take specific colors, OR a range of values (1,1,1,2,2,2,3,3,3..etc) for mapping onto a cmap
#plot PCA
"""fig = plt.figure(figsize=(10, 5))
ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(demeaned_pca_3[:,0], demeaned_pca_3[:,1], demeaned_pca_3[:,2], s=50, alpha=0.6, c='red')#c=colors, cmap="rainbow")
ax.set_title('PCA - SKLEARN')
plt.show() """

print("BEGINNING SKLEARN PCA AVERAGE CALCULATIONS")

#create PCA averages
avg_matrix = []
avg_interval = num_curves #default to num_curves
num_avg_bins = round((num_bins*num_curves)/avg_interval) #e.g. (5*400)/400 = 2000/400 = 5, for basic case. total num of curves/interval
#rounding in ^^ is really bad, find workaround
for i in range(num_avg_bins):
    interval = avg_interval #getting the interval per energy range
    range1 = i*interval
    range2 = range1+interval
    x = np.matrix(demeaned_matrix[range1:range2,0])
    y = np.matrix(demeaned_matrix[range1:range2,1])
    z = np.matrix(demeaned_matrix[range1:range2,2])
    avg_values = [x.mean(), y.mean(), z.mean()]
    avg_matrix.append(avg_values)
avg_matrix = np.array(avg_matrix) #scuffed, need for slicing in plot

"""testing code - use to verify that coloring applies in sequence
#avg_matrix.insert(0, [100,100,100])
#avg_matrix.append([110,110,110])
#print(avg_matrix[:,0])
#print(avg_matrix[:,1])
#print(avg_matrix[:,2])
#print(len(avg_colors)) <-- move this to after the loop """

print("ADDING AVERAGE COLORS...")

#initialize & add color coding to PCA
avg_colors = []
for i in range(len(avg_matrix[:,0])):
    if(np.isnan(avg_matrix[i,0])):
        continue
    else:
        avg_colors.append(i)
#basically, just list 1,2,3...n to color the points in sequence

#plot PCA averages
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(avg_matrix[:,0], avg_matrix[:,1], avg_matrix[:,2], s=50, alpha=0.6, c=avg_colors, cmap='rainbow')
ax.set_title('Average Values from PCA - SKLEARN')
plt.show()

#toggle saving to file - CHANGE NAME BEFORE DOING SO FROM DEFAULT
#naming convention - sp_avgs_(xxx)int_(xxxx)curv_(xxx)k.csv
#sp - shower profile, avgs - averages, (xxx)int - interval, without decimal (ie 0.25 = 025), 
#(xxx)curv for num_curves used, (xxx)k for dataset size used
avg_savefile_name = 'sp_avgs_025int_2000curv_100k.npy'
#avg_savefile_name = 'defualt_averages.npy'
save_to_file = False #default False
if(save_to_file):
    print("SAVING AVERAGES TO FILE...")
    np.save(avg_savefile_name, avg_matrix)
    print("AVERAGES SAVED TO FILE:", avg_savefile_name)
    #avg_matrix.tofile(avg_savefile_name, sep = ',')

# -- END SKLEARN PCA --