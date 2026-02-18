"""
COPYRIGHT & USAGE NOTICE:
This code is provided strictly for EDUCATIONAL PURPOSES only. 
© 2026. All Rights Reserved.

PROGRAM DESCRIPTION:
--------------------
This program uses an Artificial Intelligence (AI) model called a "Neural Network" 
to learn the behavior of a water level sensor (LIT101). 

Once the AI learns what is "normal," it can automatically detect when something 
is wrong—such as a cyber-attack or a hardware failure.

INPUTS:
-------
1. 'Dataset/dataset.csv': A file containing history of sensor readings (the "textbook" for the AI).
2. Settings: How many past readings the AI should look at to guess the next one.

PROCESS:
--------
1. Teaching the AI: We show the AI thousands of examples of normal water levels.
2. Setting Safety Limits: We look at the small mistakes the AI makes and build 
   a "safety zone" around its predictions.
3. Looking for Trouble: We intentionally change the data (simulating an attack) 
   to see if the AI notices that the error has gone outside the safety zone.

OUTPUTS:
--------
1. 'MLP_results/': A folder where the program saves pictures of the graphs and 
   the "brain" (model) of the trained AI.
"""

import numpy as np
import pandas as pd
import math
import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random

# --- STEP 1: PREPARING THE FOLDER ---
# We want to save our results in a folder called 'MLP_results'
results_folder = "MLP_results"

# If the folder already exists from a previous run, delete it so we start fresh
if os.path.exists(results_folder):
    shutil.rmtree(results_folder)

# Create a brand new, empty folder for today's results
os.makedirs(results_folder)

# --- STEP 2: LOADING THE DATA ---
# Tell the program where the data file is located
file_path = os.path.join("Dataset", "dataset.csv")

# Read the file and turn it into a table we can work with
dataframe = pd.read_csv(file_path)
dataset = dataframe.values

# Extract the data for the three main parts of the water system:
# 'lit101' is the Water Level, 'mv101' is a Valve, and 'p101' is a Pump
lit101 = dataset[3600:3600*2, 1]
mv101 = dataset[3600:3600*2, 2]
p101 = dataset[3600:3600*2, 3]

# --- STEP 3: THE "SLIDING WINDOW" TOOL ---
# Since AI needs examples to learn, this function takes a long list of numbers 
# and cuts them into small chunks. 
# It creates "Question" (past readings) and "Answer" (the very next reading).
def Convert_window(dataset, look_back, look_front, interval):
    dataX = []
    dataY = []
    i = 0
    while (i < len(dataset) - look_back - look_front - interval - 1):
        a = dataset[i:i + look_back]
        dataX.append(a)
        b = dataset[i + look_back + interval:i + look_back + look_front + interval]
        dataY.append(b)
        i = i + look_front
    return np.array(dataX), np.array(dataY)

# --- STEP 4: VISUALIZING THE SYSTEM ---
# Create a picture showing how the Valve, Water Level, and Pump behave over time
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(mv101, label="Motor Valve (On/Off Status)")
plt.ylabel('Status')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(lit101, label="Water Level (LIT101)")
plt.ylabel('Level')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(p101, label="Water Pump (On/Off Status)")
plt.ylabel('Status')
plt.xlabel('Time (Seconds)')
plt.legend()

# Save this picture to our results folder
plt.savefig(os.path.join(results_folder, "system_behavior.png"))
plt.show()

# --- STEP 5: FINDING PATTERNS ---
# This graph helps us see if the current water level is related to past levels 
# (useful for deciding how much the AI should "look back")
plt.figure()
plt.acorr(lit101.astype(float), maxlags=10)
plt.title('Checking for Patterns in Water Level')
plt.xlabel('Time Lag')
plt.ylabel('Correlation Strength')
plt.savefig(os.path.join(results_folder, "autocorrelation.png"))
plt.show()

# --- STEP 6: SETTING THE AI "SEED" ---
# We use a lucky number (14) to make sure the AI starts moving in the same 
# way every time we run the program. This makes our results consistent.
seed = 14
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)

# --- STEP 7: TRAINING THE AI ---
# We tell the AI to look at the last 5 readings to guess the next 1 reading
look_back, look_front, interval = 5, 1, 0
X, Y = Convert_window(lit101, look_back, look_front, interval)

# We hide some data (33%) from the AI during training so we can test it later 
# like a final exam.
X_train, X_test_initial, Y_train, Y_test_initial = train_test_split(X, Y, test_size=0.33, random_state=seed)

# Convert data into a technical format the AI understands
X_train = np.asarray(X_train).astype(np.float32)
Y_train = np.asarray(Y_train).astype(np.float32)

# Create the AI's "Brain" structure:
# It has a layer to receive info, a hidden layer to think, and an output layer for the guess.
model = Sequential()
model.add(Dense(4, input_dim=look_back, activation='relu')) 
model.add(Dense(look_front, activation='relu'))             

# Prepare the math for training
model.compile(loss='mean_squared_error', optimizer='adam')

# Start the actual learning process (it goes through the data 10 times)
print("The AI is now learning from the data...")
model.fit(X_train, Y_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2, verbose=1)

# Save the finished "Brain" to a file
model.save(os.path.join(results_folder, 'LIT101.h5'))

# --- STEP 8: CALCULATING SAFETY LIMITS ---
# We show the AI some "new" normal data and see how close its guesses are.
lit101_v2 = dataset[3600*2+1:3600*3, 1]
X_valid, Y_valid = Convert_window(lit101_v2, look_back, look_front, interval)
X_valid = np.asarray(X_valid).astype(np.float32)

# Get the AI's guesses
Pred_valid = model.predict(X_valid)
# Calculate the error (Difference between Truth and AI's Guess)
error_valid = (Y_valid - Pred_valid)

# We set a "Safety Threshold." If the error gets bigger than this, we sound the alarm.
delta = 0.5
upper_thresh = max(error_valid) + delta
lower_thresh = min(error_valid) - delta

# Create a graph to show the AI's guesses compared to reality
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.title("AI Guessing vs Real Data (Normal)")
plt.plot(Y_valid, label="Real Water Level")
plt.plot(Pred_valid, label="AI Prediction")
plt.legend()

# Show the "Safety Zone" (the red dotted lines)
plt.subplot(2, 1, 2)
plt.plot(error_valid, label="AI Error")
plt.axhline(y=upper_thresh, color='r', linestyle='--', label="Safety Limit (Top)")
plt.axhline(y=lower_thresh, color='r', linestyle='--', label="Safety Limit (Bottom)")
plt.xlabel('Time (Seconds)')
plt.ylabel('Error Amount')
plt.legend()
plt.savefig(os.path.join(results_folder, "validation_results.png"))
plt.show()

# --- STEP 9: TESTING THE AI ON NORMAL DATA ---
# This proves the AI works well when things are fine.
lit101_v3 = dataset[3600*3+1:3600*4, 1]
X_normal, Y_normal = Convert_window(lit101_v3, look_back, look_front, interval)
X_normal = np.asarray(X_normal).astype(np.float32)

Pred_normal = model.predict(X_normal)
error_normal = (Y_normal - Pred_normal)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.title("Final Test: Normal Operation")
plt.plot(Y_normal, label="Real Level")
plt.plot(Pred_normal, label="AI Guess")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(error_normal, label="Error Level")
plt.axhline(y=upper_thresh, color='r', linestyle='--')
plt.axhline(y=lower_thresh, color='r', linestyle='--')
plt.legend()
plt.savefig(os.path.join(results_folder, "testing_normal.png"))
plt.show()

# --- STEP 10: SIMULATING A CYBER-ATTACK ---
# Now, we "lie" to the system by changing the water level readings maliciously.
Y_test_attack = Y_normal.copy()
for i in range(len(Y_normal)):
    # If the water is deep (>750), the attacker tells the system it's 10 units lower
    if(Y_test_attack[i] > 750):
        Y_test_attack[i] = Y_test_attack[i] - 10

# Can the AI detect this lie? 
# We compare the "Attacked Data" to what the AI *thinks* the level should be.
error_attack = (Y_test_attack - Pred_normal)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.title("Cyber-Attack Scenario: Can AI see the lie?")
plt.plot(Y_test_attack, label="Attacked (False) Data")
plt.plot(Pred_normal, label="AI Expected Data")
plt.legend()

# Notice how the error line now crashes through the Red Dotted Safety Limits!
plt.subplot(2, 1, 2)
plt.plot(error_attack, label="Prediction Error")
plt.axhline(y=upper_thresh, color='r', linestyle='--', label="Safety Boundaries")
plt.axhline(y=lower_thresh, color='r', linestyle='--')
plt.xlabel('Time (Seconds)')
plt.ylabel('Error Amount')
plt.legend()

plt.savefig(os.path.join(results_folder, "testing_attack_detection.png"))
plt.show()

print("\nSUCCESS: The AI has been trained and tested.")
print(f"You can find all pictures and the AI brain in the '{results_folder}' folder.")
