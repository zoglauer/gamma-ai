from ShowerProfileUtils import parseTrainingData
from EnergyLossDataProcessing import toDataSpace, zBiasedInlierAnalysis, plot_3D_data, discretize_energy_deposition
import matplotlib.pyplot as plt
from Curve import Curve
import numpy as np

def gev_to_kev(gev):
    return gev * (10 ** 6)

def plot_curves(ax, event_list, low_E, high_E):
    
    event_list = event_list[0:200] # max 200 for now
    
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
                
    for i, curve in enumerate(curves):
        ax.plot(curve.t, curve.dEdt, label=f'Curve {i}')
    
    ax.set_xlabel('Penetration (t)')
    ax.set_ylabel('Energy Deposition (KeV)')
    ax.set_title(f'Energies: {low_E}-{high_E} GeV')
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 150000])
    ax.grid(True)

event_list = parseTrainingData()
energy_list = [event.gamma_energy for event in event_list]

# Box Plot of Energies for 10k dataset
plt.figure(figsize=(10, 6))
plt.boxplot(energy_list, vert=True, patch_artist=True)
plt.title('Incident Energy of 10k Events')
avg = np.mean(energy_list)
sd = np.std(energy_list)
print(f'average energy: {avg}, standard deviation: {sd}')
plt.show()

# Plot the curves between 0 and 1 GeV, 1 and 2 GeV, ... 5 and 6 GeV
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# [event.gamma_energy] = KeV
zero_to_one_mev = [event for event in event_list if gev_to_kev(0) <= event.gamma_energy < gev_to_kev(1)]
one_to_two_mev = [event for event in event_list if gev_to_kev(1) <= event.gamma_energy < gev_to_kev(2)]
two_to_three_mev = [event for event in event_list if gev_to_kev(2) <= event.gamma_energy < gev_to_kev(3)]
three_to_four_mev = [event for event in event_list if gev_to_kev(3) <= event.gamma_energy < gev_to_kev(4)]
four_to_five_mev = [event for event in event_list if gev_to_kev(4) <= event.gamma_energy < gev_to_kev(5)]

plot_curves(axs[0, 0], zero_to_one_mev, 0, 1)
plot_curves(axs[0, 1], one_to_two_mev, 1, 2)
plot_curves(axs[0, 2], two_to_three_mev, 2, 3)
plot_curves(axs[1, 0], three_to_four_mev, 3, 4)
plot_curves(axs[1, 1], four_to_five_mev, 4, 5)

plt.tight_layout()
plt.show()


"""
# Find similar events and generate curves

similar_event_curves = []
similar_count = 0

resolution = 1

for event in event_list:
    
    if avg - 0.0015 * sd < event.gamma_energy < avg + 0.0015 * sd:
        
        similar_count += 1
        
        data = toDataSpace(event)
        inlierData, outlierData = zBiasedInlierAnalysis(data)

        if inlierData is not None and len(inlierData > 20):

            t_expected, dEdt_expected = discretize_energy_deposition(inlierData, resolution)
            # plt.plot(t_expected, dEdt_expected)
            
            gamma_energy = event.gamma_energy
            
            curve = Curve.fit(t_expected, dEdt_expected, gamma_energy, resolution)
            # plot_3D_data(data, inlierData)
            if curve is not None:
                similar_event_curves.append(curve)
                
for i, curve in enumerate(similar_event_curves):
    plt.plot(curve.t, curve.dEdt, label=f'Similar Energy Curve {i+1}')

print(f'Similar Count: {similar_count}. Curves Generated: {len(similar_event_curves)}')
print(f'avg energy: {event.gamma_energy}, sd= {event.gamma_energy * 0.0015}')

plt.xlabel('Penetration (t)')
plt.ylabel('Energy Deposition (KeV)')
plt.title('Curves with Energies within 0.0015 SD')
plt.legend()
plt.grid(True)
plt.show()
"""