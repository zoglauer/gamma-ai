from ShowerProfileUtils import parseTrainingData
from EnergyLossDataProcessing import toDataSpace, zBiasedInlierAnalysis, interpretAndDiscretize, plot_3D_data, discretize_energy_deposition
import matplotlib.pyplot as plt
from Curve import Curve
import numpy as np


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
