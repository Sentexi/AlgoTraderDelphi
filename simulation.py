import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def run_simulation(models_path="models", tolerance=0.0):
    """Run simulation across all models using a tolerance around previous actual value."""
    dates = None
    simulation_data = {}
    simulation_actual_values_data = {}

    for model_folder in os.listdir(models_path):
        model_path = os.path.join(models_path, model_folder)

        if os.path.isdir(model_path) and "validation" in os.listdir(model_path):
            validation_path = os.path.join(model_path, "validation")
            csv_path = os.path.join(validation_path, "validation_set.csv")

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                if 'actual' in df.columns and 'prediction' in df.columns:
                    if dates is None:
                        dates = df['datetime']

                    predictions = []
                    actual_values_changes = []

                    for i in range(1, len(df)):
                        lower_bound = df.loc[i - 1, 'actual'] * (1 - tolerance)
                        upper_bound = df.loc[i - 1, 'actual'] * (1 + tolerance)

                        if tolerance > 0 and lower_bound <= df.loc[i, 'prediction'] <= upper_bound:
                            predictions.append(0)
                        elif df.loc[i, 'prediction'] > df.loc[i - 1, 'actual']:
                            predictions.append(1)
                        else:
                            predictions.append(-1)

                        actual_change = 1 - (df.loc[i - 1, 'actual'] / df.loc[i, 'actual'])
                        actual_values_changes.append(actual_change)

                    simulation_data[model_folder] = [None] + predictions
                    simulation_actual_values_data[model_folder] = [None] + actual_values_changes

    simulation_df = pd.DataFrame(simulation_data)
    simulation_df.insert(0, 'date', dates)

    simulation_actual_values_df = pd.DataFrame(simulation_actual_values_data)
    simulation_actual_values_df.insert(0, 'date', dates)

    print(simulation_df)
    print(simulation_actual_values_df)

    if not simulation_df.empty and not simulation_actual_values_df.empty:
        if simulation_df.shape == simulation_actual_values_df.shape:
            exp_value_df = simulation_df.iloc[:, 1:] * simulation_actual_values_df.iloc[:, 1:]
            exp_value_df.insert(0, 'date', simulation_df['date'])

            print(exp_value_df)
        else:
            print("Error: The shapes of 'simulation_df' and 'simulation_actual_values_df' do not match.")
    else:
        print("Error: One or both of the DataFrames are empty. Please check the data and try again.")

    aggr_exp_value_df = exp_value_df.copy()
    aggr_exp_value_df['mean'] = aggr_exp_value_df.iloc[:, 1:].mean(axis=1) * 100

    aggr_exp_value_df = aggr_exp_value_df[['date', 'mean']]

    print(aggr_exp_value_df)

    plt.figure(figsize=(12, 6))
    plt.bar(aggr_exp_value_df['date'], aggr_exp_value_df['mean'])
    plt.xlabel('Date')
    plt.ylabel('Mean of Expected Values in Percent')
    plt.title('Aggregated Expected Values Over Time')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('simulation.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with optional prediction tolerance.")
    parser.add_argument("--models-path", default="models", help="Path to the models directory")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="Prediction tolerance as decimal (e.g. 0.005 for ±0.5%).")
    args = parser.parse_args()

    run_simulation(models_path=args.models_path, tolerance=args.tolerance)
