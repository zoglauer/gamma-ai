from EnergyLossDataProcessing import toDataSpace, zBiasedInlierAnalysis, discretize_energy_deposition
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Curve import Curve
import numpy as np
import os.path
import csv

def kev_to_gev(kev):
    return kev / (10**6)

def distribute_events_to_energy_bins(event_list, resolution, log_scale=False):
    events = {key:[] for key in np.arange(0.0, 5, resolution)} # e.g. resolution = 1.0, events = {0.0: [], 1.0: [], 2.0: [], 3.0: [], 4.0: []}
    for event in event_list:
        event_energy = kev_to_gev(event.gamma_energy)
        event_energy_range = event_energy - (event_energy % resolution)
        events[event_energy_range].append(event)
    return events

def create_curves(sliced_event_list: list, resolution: float = 1.0, num_curves: int = 400) -> list:
    
    curves = []

    index = 0
    while len(curves) < num_curves:
        
        if index >= len(sliced_event_list):
            print(f"Unable to generate enough curves. Number of events = {len(sliced_event_list)}. Number of curves desired = {num_curves}. Curves generated = {len(curves)}.")
            break
        
        event = sliced_event_list[index]

        data = toDataSpace(event)
        inlierData, outlierData = zBiasedInlierAnalysis(data)

        if inlierData is None or len(inlierData) < 0.85 * len(data): # inlier analysis failed (no trend), use regular data
            inlierData = data
            
        gamma_energy = event.gamma_energy
        t_expected, dEdt_expected = discretize_energy_deposition(inlierData, gamma_energy, resolution)
        curve = Curve.fit(t_expected, dEdt_expected, gamma_energy, resolution)

        if curve is not None:
            curves.append(curve)
        
        index += 1
    
    return curves

def load(filename):
    return np.loadtxt(filename, delimiter=',')[1:,:]

def save(data_matrix, file_path: str):
    headers = [f'{i}' for i in range(len(data_matrix[0]))]
    
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        # Write each row in the matrix to the CSV
        for row in data_matrix:
            csvwriter.writerow(row)

def create_data_matrix(curves, bins: int):
    height = len(curves)
    data_matrix = np.zeros((height, bins))
    for i in range(height):
        row = process_curve(bins, curves[i])
        data_matrix[i] = row
    return data_matrix

def get_data_matrix(should_load: bool, file_path: str, event_dict: dict = None, curves_per_range: int = None, curve_features: float = None, curve_resolution: float = None):
    if should_load and os.path.exists(file_path):
        return load(file_path)
    for energy_range in event_dict.keys():
        event_dict[energy_range] = create_curves(event_dict[energy_range], curve_resolution, curves_per_range)
        print(f'Curves generated for {energy_range}GeV.')
    curves_list = [curve for row in event_dict.values() for curve in row]
    data_matrix = create_data_matrix(curves_list, curve_features)
    plot_raw_values(curves_list, curves_per_range)
    save(data_matrix, file_path)
    return data_matrix

def get_random_curve(sliced_event_list: list, resolution: float = 1.0) -> Curve:
    return create_curves(sliced_event_list, resolution, num_curves=1)[0]

def process_curve(bins: int, curve: Curve):
    row = list(curve.y[:bins])
    row = [max(0, value) for value in row]
    row.extend([0 for _ in range(bins - len(row))])
    return row

def energy_box_plot(curve_list):
    # Box Plot of Energies for dataset
    energy_list = [curve.energy for curve in curve_list]
    plt.figure(figsize=(10, 6))
    plt.boxplot(energy_list, vert=True, patch_artist=True)
    plt.title(f'Incident Energy of {len(curve_list)} Events')
    avg = np.mean(energy_list)
    sd = np.std(energy_list)
    print(f'average energy: {avg}, standard deviation: {sd}')
    plt.show()
    
def plot_raw_values(curve_list, rows_per_range):
    plt.figure()
    
    # Calculate the total number of ranges
    num_ranges = len(curve_list) // rows_per_range
    
    # Generate colors from a colormap
    colors = cm.viridis(np.linspace(0, 1, num_ranges))
    
    for i, curve in enumerate(curve_list):
        # Determine which range the curve belongs to
        range_index = i // rows_per_range
        # Plot with the corresponding color
        plt.plot(curve.t, curve.dEdtoverE0, alpha=0.5, color=colors[range_index])

    plt.savefig('rawEData.png')
    