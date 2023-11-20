import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Parameters
csv_file_path = 'curve_matrix_100K_gamma_fit_600_features.csv'
energy_bin_size = 0.0895
curves_per_range = 500 # 500
num_energy_ranges = int(5.012 / energy_bin_size)
t_resolution = 0.05
x_tick_step = 20 # Used to decrease the number of ticks shown on the x axis
output_directory = f'plots_{csv_file_path}'  

# Load the data from the CSV file
data = pd.read_csv(csv_file_path)

# Create the output directory if it doesn't exist
Path(output_directory).mkdir(parents=True, exist_ok=True)

# Function to create and save a plot for each energy range
def create_and_save_plot(data_subset, set_number, output_dir):
    plt.figure(figsize=(12, 6))
    
    # Plot all rows in the data subset
    for index, row in data_subset.iterrows():
        plt.plot(row.index, row.values, color='lightblue', alpha=0.1)

    # Setting xticks to show every column index
    tick_positions = range(0, len(data_subset.columns), x_tick_step)
    tick_labels = [i // x_tick_step for i in tick_positions]
    plt.xticks(tick_positions, tick_labels)
    
    # Setting labels and title
    plt.xlabel('t (radiation lengths)')
    plt.ylabel('(1 / initial energy)*dE/dt')
    start_energy = (set_number - 1) * energy_bin_size
    end_energy = set_number * energy_bin_size
    plt.title(f'Energy Deposition for {start_energy:.4f} to {end_energy:.4f} GeV')

    # Save the plot
    file_path = os.path.join(output_dir, f'plot_range_{start_energy:.4f}_to_{end_energy:.4f}_GeV.png')
    plt.savefig(file_path)
    
    # Close the plot to free up resources
    plt.close()

# Generate and save plots
for set_number in range(0, num_energy_ranges + 1):
    start_row = set_number * curves_per_range
    end_row = start_row + curves_per_range
    subset = data.iloc[start_row:end_row]

    # Create and save the plot
    create_and_save_plot(subset, set_number + 1, output_directory)

print(f"All plots have been saved to {output_directory}")
