"""
COPYRIGHT & USAGE NOTICE:
This code is provided strictly for EDUCATIONAL PURPOSES only. 
© 2026. All Rights Reserved.

PROGRAM DESCRIPTION:
--------------------
This script implements the Cumulative Sum (CUSUM) algorithm for anomaly detection 
in sensor data. It is designed to detect small, persistent shifts in a process 
by tracking the deviation of readings from an expected target value.

INPUTS:
-------
1. 'Dataset/dataset.csv': A data file containing sensor readings. The script 
   specifically processes the 'LIT101' column.
2. Parameters:
   - b (Reference value): A slack value used to ignore minor noise.
   - UCL/LCL (Control Limits): Thresholds that define the point at which a reading 
     is flagged as anomalous.

PROCESS:
--------
1. Baseline Calculation: Uses the first 1,000 readings to calculate the 'normal' 
   average (Target) and standard deviation of the system.
2. Data Windowing: Selects a specific test segment (readings 1,000 to 1,100) for analysis.
3. Cumulative Sum Analysis:
   - Iterates through the test data time-step by time-step.
   - Calculates 'upper' and 'lower' cumulative sums to track positive and negative drifts.
   - Resets the sums to zero if the drift indicates a return to normal behavior.
4. Anomaly Detection: Flags a sample as 'Anomalous' if the cumulative sum crosses 
   the defined Upper Control Limit (UCL) or Lower Control Limit (LCL).
5. Visualization: Generates a plot of sensor readings and marks anomalies with red dots.

OUTPUTS:
--------
1. 'CUSUM_results/cusum_results.csv': A detailed file containing original readings, 
   calculated increments, cumulative sums, and the final state (Normal/Anomalous).
2. Console Summary: Prints the location of the result file and the total number of 
   anomalies detected.
3. Graphical Plot: A file named 'CUSUM_results/cusum_plot.png' showing the sensor 
   readings over time with highlighted anomalies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

# --- DIRECTORY MANAGEMENT ---
# Define the name of the folder where we will save our results
results_folder = "CUSUM_results"

# If the folder already exists, we delete it and its contents to start fresh
if os.path.exists(results_folder):
    shutil.rmtree(results_folder)

# Create the new empty folder
os.makedirs(results_folder)

# Define the name of the file that contains our sensor data
# Pointing to the new folder 'Dataset'
file_path = os.path.join("Dataset", "dataset.csv")

# Load the data from the CSV file into a table (DataFrame) using the pandas library
df = pd.read_csv(file_path)

# Look at the column named 'LIT101' and extract all its values into a list for processing
sensor_readings = df["LIT101"].values

# We use the first 1000 data points to understand what "normal" looks like
# This is our training phase where we establish the baseline
training_data = sensor_readings[:1000]

# Calculate the average (mean) value of the training data to use as our target goal
Target = np.mean(training_data)

# Calculate how much the data typically fluctuates (standard deviation)
target_std = np.std(training_data)

# Select the next 100 data points (from index 1000 to 1100) to test if there are any anomalies
testing_data = sensor_readings[1000:1100]

def cusum_to_csv(data, Target, b=0.5, UCL=5, LCL=-5, output_file="cusum_results.csv", plot_file="cusum_plot.png"):
    """
    This function implements the CUSUM (Cumulative Sum) algorithm.
    It tracks small shifts away from the target value over time to detect anomalies.
    
    Parameters explained for students:
    - data: The sensor values we want to check.
    - Target: The 'normal' average we expect.
    - b: A small buffer to ignore tiny, natural fluctuations.
    - UCL: Upper Limit - if our sum goes above this, something is wrong (Positive shift).
    - LCL: Lower Limit - if our sum goes below this, something is wrong (Negative shift).
    """
    
    # Create empty placeholders to store our running totals (Cumulative Sums)
    # upper_cusum tracks values going too high, lower_cusum tracks values going too low
    upper_cusum = np.zeros(len(data))
    lower_cusum = np.zeros(len(data))
    
    # This list will store all the details for every second of the test
    results = []

    # Start checking the data point by point, starting from the second one
    for t in range(1, len(data)):
        x_t = data[t]  # This is the current sensor reading at this exact moment

        # Step 1: Calculate how much we are drifting upward
        # We take the previous total, add the current difference from target, and subtract the buffer 'b'
        upper_increment = upper_cusum[t-1] + (x_t - Target - b)
        
        # Step 2: Calculate how much we are drifting downward
        # We take the previous total, add the current difference from target, and add the buffer 'b'
        lower_increment = lower_cusum[t-1] + (x_t - Target + b)

        # Step 3: Update our totals. 
        # If the drift is negative for the upper side, we reset to 0 (no upward anomaly)
        upper_cusum[t] = max(0, upper_increment)
        
        # If the drift is positive for the lower side, we reset to 0 (no downward anomaly)
        lower_cusum[t] = min(0, lower_increment)

        # Step 4: Check if our totals have crossed the danger zones (Control Limits)
        if upper_cusum[t] > UCL or lower_cusum[t] < LCL:
            state = 'Anomalous'  # Danger: An anomaly has been detected!
            upper_cusum[t] = 0   # Reset the totals to start fresh after detection
            lower_cusum[t] = 0
        else:
            state = 'Normal'     # Everything looks fine

        # Save all the math we just did into a dictionary for this specific time step
        results.append({
            'x_t': x_t,
            'Target': Target,
            'b': b,
            'upper_increment': upper_increment,
            'lower_increment': lower_increment,
            'upper_cusum[t]': upper_cusum[t],
            'UCL': UCL,
            'lower_cusum[t]': lower_cusum[t],
            'LCL': LCL,
            'State': state
        })

    # Create the full file paths for saving inside our results folder
    csv_path = os.path.join(results_folder, output_file)
    plot_path = os.path.join(results_folder, plot_file)

    # Convert our list of results into a neat table and save it as a file
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Count how many times the system flagged an 'Anomalous' state
    anomalous_count = results_df['State'].value_counts().get('Anomalous', 0)
    print(f"Number of samples detected as anomalous: {anomalous_count}")
    
    # --- PLOTTING SECTION ---
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the sensor readings as a blue line
    plt.plot(results_df.index, results_df['x_t'], label='Sensor Reading (LIT101)', color='blue', alpha=0.7)
    
    # Find the indices where an anomaly was detected
    anomalies = results_df[results_df['State'] == 'Anomalous']
    
    # Mark these anomalies with red dots
    plt.scatter(anomalies.index, anomalies['x_t'], color='red', label='Detected Anomaly', zorder=5)
    
    # Add labels and formatting to make it professional and readable
    plt.title('CUSUM Anomaly Detection on Sensor LIT101', fontsize=14)
    plt.xlabel('Sample Number', fontsize=12)
    plt.ylabel('Sensor Reading Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot as an image file
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Show the final plot
    print("Displaying plot...")
    plt.show()

# Setting up the rules for our detection
b = 100  # Sensitivity buffer
UCL = 5  # Positive limit
LCL = -5 # Negative limit

# Run the detection function with our testing data
cusum_to_csv(testing_data, Target, b, UCL, LCL)
