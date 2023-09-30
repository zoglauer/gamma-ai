from ShowerProfileUtils import parseTrainingData
from EnergyLossDataProcessing import toDataSpace, zBiasedInlierAnalysis, plot_3D_data, discretize_energy_deposition
import matplotlib.pyplot as plt
from Curve import Curve
import numpy as np
import os.path
import csv

def gev_to_kev(gev):
    return gev * (10 ** 6)

def create_curves(event_list: list) -> list:
    
    resolution = 1.0 # essentially the bin size
    curves = []
    
    for event in event_list:
        
        data = toDataSpace(event)
        inlierData, outlierData = zBiasedInlierAnalysis(data)

        if inlierData is not None and len(inlierData > 20):

            t_expected, dEdt_expected = discretize_energy_deposition(inlierData, resolution)
            gamma_energy = event.gamma_energy
            curve = Curve.fit(t_expected, dEdt_expected, gamma_energy, resolution)

            if curve is not None:
                curves.append(curve)
    
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

def save(data_matrix):
    headers = [f'{i}' for i in range(len(data_matrix[0]))]
    
    with open('shower_profile.csv', 'w', newline='') as csvfile:
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

# --- ANALYZE THE CURVES BETWEEN 0 AND 1 GEV, 1 AND 2 GEV, ... 5 AND 6 GEV ---

file_path = 'shower_profile.csv'
if os.path.exists(file_path):
    data_matrix = load(file_path)
    
    # TODO: PCA
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

    # TODO: gen data_matrix

    # Save Curve Data
    save(data_matrix)

    """
    # Plot Curves
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    plot_curves(axs[0, 0], zero_to_one_mev_curves, 0, 1)
    plot_curves(axs[0, 1], one_to_two_mev_curves, 1, 2)
    plot_curves(axs[0, 2], two_to_three_mev_curves, 2, 3)
    plot_curves(axs[1, 0], three_to_four_mev_curves, 3, 4)
    plot_curves(axs[1, 1], four_to_five_mev_curves, 4, 5)

    plt.tight_layout()
    plt.show()
    """
    