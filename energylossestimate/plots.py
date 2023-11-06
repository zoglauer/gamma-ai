import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Parameters
csv_file_path = 'curve_matrix_100K_gamma_fit_70_features.csv'
rows_per_set = 250
rows_to_select = 50
output_directory = 'plots_gamma_fit'  

# Load the data from the CSV file
data = pd.read_csv(csv_file_path)

# Create the output directory if it doesn't exist
Path(output_directory).mkdir(parents=True, exist_ok=True)

# Function to create and save a plot for a subset of rows
def create_and_save_plot(data_subset, set_number, output_dir):
    plt.figure(figsize=(12, 6))
    for index, row in data_subset.iterrows():
        plt.plot(row.index, row.values)
    plt.title(f'Column Values for Set {set_number}')
    plt.xlabel('Column Index')
    plt.ylabel('Value')
    plt.tight_layout()
    file_path = os.path.join(output_dir, f'plot_set_{set_number}.png')
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up resources

# Generate and save plots
num_sets = len(data) // rows_per_set
for set_number in range(num_sets + 1):  # Include the last set which may not have 250 rows
    start_row = set_number * rows_per_set
    end_row = start_row + rows_per_set
    subset = data.iloc[start_row:end_row]

    # Select 50 random rows from the subset if available, otherwise take all rows
    if len(subset) >= rows_to_select:
        subset = subset.sample(n=rows_to_select, random_state=set_number)
    else:
        subset = subset.sample(n=len(subset), random_state=set_number)

    # Create and save the plot
    create_and_save_plot(subset, set_number + 1, output_directory)

print(f"All plots have been saved to {output_directory}")
