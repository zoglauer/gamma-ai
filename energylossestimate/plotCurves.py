import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Parameters
csv_file_path = 'curve_matrix_100K_gamma_fit_280_features.csv'
rows_per_set = 250  # Assuming each set has up to ~250 rows
num_sets = 56  # Total number of energy ranges
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
        plt.plot(row.index, row.values, color='lightblue', alpha=0.5)

    # Setting xticks to show every column index
    tick_positions = range(0, len(data_subset.columns), 20)
    tick_labels = [i // 20 for i in tick_positions]
    plt.xticks(tick_positions, tick_labels)
    
    # Setting labels and title
    plt.xlabel('t (radiation lengths)')
    plt.ylabel('(1 / initial energy)*dE/dt')
    start_energy = (set_number - 1) * 0.0895
    end_energy = set_number * 0.0895
    plt.title(f'Energy Deposition for {start_energy:.4f} to {end_energy:.4f} GeV')

    # Save the plot
    file_path = os.path.join(output_dir, f'plot_range_{start_energy:.4f}_to_{end_energy:.4f}_GeV.png')
    plt.savefig(file_path)
    
    # Close the plot to free up resources
    plt.close()

# Generate and save plots
for set_number in range(1, num_sets + 1):
    start_row = (set_number - 1) * rows_per_set
    end_row = start_row + rows_per_set
    subset = data.iloc[start_row:end_row]

    # Create and save the plot
    create_and_save_plot(subset, set_number, output_directory)

print(f"All plots have been saved to {output_directory}")
