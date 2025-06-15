import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the path to the models folder
models_path = "models"

# Initialize a list to store the DataFrames
all_dfs = []
total_folders = 0

# Loop through each model folder in the models directory
for model_folder in os.listdir(models_path):
    model_path = os.path.join(models_path, model_folder)

    # Check if it is a directory and contains a validation folder
    if os.path.isdir(model_path) and "validation" in os.listdir(model_path):
        total_folders += 1
        validation_path = os.path.join(model_path, "validation")

        # Load the validation_set.csv
        csv_path = os.path.join(validation_path, "validation_set.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Ensure 'actual' column exists
            if 'actual' in df.columns:
                # Initialize 'naive' and 'random_guess' columns with NaN
                df['naive'] = np.nan
                df['random_guess'] = np.nan

                # Step 1: Create 'naive' column based on previous 'actual' values
                for i in range(2, len(df)):
                    if df.loc[i - 1, 'actual'] < df.loc[i - 2, 'actual']:
                        df.loc[i, 'naive'] = -1
                    else:
                        df.loc[i, 'naive'] = 1

                # Step 2: Create 'random_guess' column with random assignment
                df['random_guess'] = [random.choice([-1, 1]) for _ in range(len(df))]

                # Initialize 'naive_estimation' and 'random_estimation' columns with NaN
                df['naive_estimation'] = np.nan
                df['random_estimation'] = np.nan

                # Step 3: Create 'naive_estimation' and 'random_estimation' columns
                for i in range(1, len(df)):
                    if pd.notna(df.loc[i, 'naive']) and pd.notna(df.loc[i - 1, 'actual']):
                        # Naive estimation
                        if df.loc[i, 'actual'] > df.loc[i - 1, 'actual'] and df.loc[i, 'naive'] == 1:
                            df.loc[i, 'naive_estimation'] = True
                        elif df.loc[i, 'actual'] < df.loc[i - 1, 'actual'] and df.loc[i, 'naive'] == -1:
                            df.loc[i, 'naive_estimation'] = True
                        else:
                            df.loc[i, 'naive_estimation'] = False

                    if pd.notna(df.loc[i, 'random_guess']) and pd.notna(df.loc[i - 1, 'actual']):
                        # Random estimation
                        if df.loc[i, 'actual'] > df.loc[i - 1, 'actual'] and df.loc[i, 'random_guess'] == 1:
                            df.loc[i, 'random_estimation'] = True
                        elif df.loc[i, 'actual'] < df.loc[i - 1, 'actual'] and df.loc[i, 'random_guess'] == -1:
                            df.loc[i, 'random_estimation'] = True
                        else:
                            df.loc[i, 'random_estimation'] = False

                # Append the processed DataFrame to the list
                all_dfs.append(df)

# Concatenate all DataFrames
if all_dfs:
    aggregated_df = pd.concat(all_dfs)

    # Calculate the percentage of True values in 'naive_estimation' and 'random_estimation'
    true_naive_count = aggregated_df['naive_estimation'].sum()
    false_naive_count = len(aggregated_df['naive_estimation']) - true_naive_count
    total_naive_count = true_naive_count + false_naive_count

    true_random_count = aggregated_df['random_estimation'].sum()
    false_random_count = len(aggregated_df['random_estimation']) - true_random_count
    total_random_count = true_random_count + false_random_count

    # Calculate percentages
    naive_percentage = (true_naive_count / total_naive_count) * 100 if total_naive_count > 0 else 0
    random_percentage = (true_random_count / total_random_count) * 100 if total_random_count > 0 else 0

    # Print the percentages
    print(f"Percentage of True naive estimations: {naive_percentage:.2f}%")
    print(f"Percentage of True random estimations: {random_percentage:.2f}%")

    # Create pie charts for naive and random estimations
    labels = ['True', 'False']
    naive_sizes = [true_naive_count, false_naive_count]
    random_sizes = [true_random_count, false_random_count]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Naive estimation pie chart
    axs[0].pie(naive_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axs[0].set_title('Naive Estimation Results')
    axs[0].text(0, -1.2, f"Analysed {total_folders} stocks\nAnalysed {len(aggregated_df)} datapoints", ha='center', fontsize=10)

    # Random estimation pie chart
    axs[1].pie(random_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axs[1].set_title('Random Estimation Results')
    axs[1].text(0, -1.2, f"Analysed {total_folders} stocks\nAnalysed {len(aggregated_df)} datapoints", ha='center', fontsize=10)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig('naive_val.png')
    plt.show()
else:
    print("No data processed. Check if 'validation_set.csv' files are present and valid.")
