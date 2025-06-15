import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the models folder
models_path = "models"

# Initialize an empty list to store the dates and dictionaries for predictions and actual values
dates = None
simulation_data = {}
simulation_actual_values_data = {}

# Loop through each model folder in the models directory
for model_folder in os.listdir(models_path):
    model_path = os.path.join(models_path, model_folder)

    # Check if it is a directory and contains a validation folder
    if os.path.isdir(model_path) and "validation" in os.listdir(model_path):
        validation_path = os.path.join(model_path, "validation")

        # Load the validation_set.csv
        csv_path = os.path.join(validation_path, "validation_set.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Ensure 'actual' and 'prediction' columns exist
            if 'actual' in df.columns and 'prediction' in df.columns:
                # Initialize the dates if not already set
                if dates is None:
                    dates = df['datetime']

                # Initialize lists for this model's predictions and actual value changes
                predictions = []
                actual_values_changes = []

                # Evaluate predictions and calculate percentage changes
                for i in range(1, len(df)):
                    # Prediction comparison for simulation data
                    if df.loc[i, 'prediction'] > df.loc[i - 1, 'actual']:
                        predictions.append(1)
                    else:
                        predictions.append(-1)

                    # Calculate the percentage change for actual values
                    actual_change = 1 - (df.loc[i-1, 'actual'] / df.loc[i, 'actual'])
                    actual_values_changes.append(actual_change)

                # Align the predictions with the dates (fill the first row with NaN)
                simulation_data[model_folder] = [None] + predictions
                simulation_actual_values_data[model_folder] = [None] + actual_values_changes

# Create the DataFrames
simulation_df = pd.DataFrame(simulation_data)
simulation_df.insert(0, 'date', dates)

simulation_actual_values_df = pd.DataFrame(simulation_actual_values_data)
simulation_actual_values_df.insert(0, 'date', dates)

print(simulation_df)
print(simulation_actual_values_df)

# Multiply the corresponding elements of the two DataFrames to create 'exp_value_df'
if not simulation_df.empty and not simulation_actual_values_df.empty:
    if simulation_df.shape == simulation_actual_values_df.shape:
        exp_value_df = simulation_df.iloc[:, 1:] * simulation_actual_values_df.iloc[:, 1:]
        exp_value_df.insert(0, 'date', simulation_df['date'])

        # Display the resulting exp_value DataFrame
        print(exp_value_df)
    else:
        print("Error: The shapes of 'simulation_df' and 'simulation_actual_values_df' do not match.")
else:
    print("Error: One or both of the DataFrames are empty. Please check the data and try again.")

# Create average up the values by row for the exp_value_df, excluding the 'date' column
aggr_exp_value_df = exp_value_df.copy()
#We create the mean and multiply by 100 for percentages
aggr_exp_value_df['mean'] = aggr_exp_value_df.iloc[:, 1:].mean(axis=1)*100

# Keep only the 'date' and the summed values
aggr_exp_value_df = aggr_exp_value_df[['date', 'mean']]

print(aggr_exp_value_df)

# Create a bar chart for the aggregated expected values
plt.figure(figsize=(12, 6))
plt.bar(aggr_exp_value_df['date'], aggr_exp_value_df['mean'])
plt.xlabel('Date')
plt.ylabel('Mean of Expected Values in Percent')
plt.title('Aggregated Expected Values Over Time')
plt.xticks(rotation=45, ha='right')

# Save the plot as 'simulation.png'
plt.tight_layout()
plt.savefig('simulation.png')
plt.show()