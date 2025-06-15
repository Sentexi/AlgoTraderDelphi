import os
import pandas as pd
import matplotlib.pyplot as plt

def merge_validation_sets(models_dir='models'):
    merged_data = []

    # Count the number of folders (stocks)
    folder_count = 0

    for folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, folder)
        
        if os.path.isdir(model_path):
            folder_count += 1
            validation_path = os.path.join(model_path, 'validation', 'validation_set.csv')

            if os.path.exists(validation_path):
                try:
                    df = pd.read_csv(validation_path)
                    merged_data.append(df)
                except Exception as e:
                    print(f"Error reading {validation_path}: {e}")
    
    if merged_data:
        merged_df = pd.concat(merged_data, ignore_index=True)
        return merged_df, folder_count
    else:
        print("No validation sets found.")
        return pd.DataFrame(), 0
        
def analyze_model_folder(folder_path):
    # Extract R-squared value from predictions.txt
    cut_path = os.path.join(folder_path, 'cut', 'predictions.txt')
    r_squared = None
    if os.path.exists(cut_path):
        with open(cut_path, 'r') as file:
            for line in file:
                if "Test R-squared:" in line:
                    r_squared = float(line.split("Test R-squared:")[-1].strip())
                    break

    # Extract validation set and calculate true percentages
    validation_path = os.path.join(folder_path, 'validation', 'validation_set.csv')
    direction_true_percent = None
    in_bounds_true_percent = None

    if os.path.exists(validation_path):
        df = pd.read_csv(validation_path)

        if 'direction' in df.columns:
            direction_true_count = df['direction'].sum()
            direction_true_percent = (direction_true_count / len(df)) * 100

        if 'in_bounds' in df.columns:
            in_bounds_true_count = df['in_bounds'].sum()
            in_bounds_true_percent = (in_bounds_true_count / len(df)) * 100

    return r_squared, direction_true_percent, in_bounds_true_percent

def analyze_validation_set(result_df, folder_count):
    # Check if 'direction' column exists
    if 'direction' not in result_df.columns:
        print("'direction' column not found in the dataset.")
        return

    # Analysis for the 'direction' column
    true_count_direction = result_df['direction'].sum()
    false_count_direction = len(result_df) - true_count_direction
    total_count_direction = true_count_direction + false_count_direction
    prediction_percentage = (true_count_direction / total_count_direction) * 100 if total_count_direction > 0 else 0

    print(f"Predicted correctly {prediction_percentage:.1f}% of the time")

    # Check if 'in_bounds' column exists
    if 'in_bounds' not in result_df.columns:
        print("'in_bounds' column not found in the dataset.")
        return

    # Analysis for the 'in_bounds' column
    true_count_in_bounds = result_df['in_bounds'].sum()
    false_count_in_bounds = len(result_df) - true_count_in_bounds
    total_count_in_bounds = true_count_in_bounds + false_count_in_bounds
    in_bounds_percentage = (true_count_in_bounds / total_count_in_bounds) * 100 if total_count_in_bounds > 0 else 0

    print(f"Actual value was within bounds {in_bounds_percentage:.1f}% of the time")

    labels = ['True', 'False']

    # Sizes for 'direction' pie chart
    sizes_direction = [true_count_direction, false_count_direction]
    # Sizes for 'in_bounds' pie chart
    sizes_in_bounds = [true_count_in_bounds, false_count_in_bounds]

    fig, ax = plt.subplots(1, 4, figsize=(24, 8))

    # Pie chart for 'direction'
    ax[0].pie(sizes_direction, labels=labels, autopct='%1.1f%%', startangle=90)
    ax[0].axis('equal')
    ax[0].set_title("True vs False Predictions (Direction)")

    # Pie chart for 'in_bounds'
    ax[1].pie(sizes_in_bounds, labels=labels, autopct='%1.1f%%', startangle=90)
    ax[1].axis('equal')
    ax[1].set_title("True vs False Predictions (In Bounds)")
    
        # Modified histogram for 'rel_deviation' based on 'direction'
    if 'rel_deviation' in result_df.columns and 'direction' in result_df.columns:
        # Adjust rel_deviation based on direction
        result_df['adjusted_rel_deviation'] = result_df.apply(
            lambda row: row['rel_deviation'] if row['direction'] else -row['rel_deviation'], axis=1
        )
        
        
        # Plot histogram of adjusted relative deviation
        ax[2].hist(result_df['adjusted_rel_deviation'].dropna(), bins=30, edgecolor='black')
        ax[2].set_title("Histogram of Adjusted Relative Deviation")
        ax[2].set_xlabel("Adjusted Relative Deviation")
        ax[2].set_ylabel("Frequency")
        
    else:
        print("'rel_deviation' or 'direction' column not found in the dataset.")
        
        # Check and create histogram for 'pred_vs_act'
    if 'pred_vs_act' in result_df.columns and 'direction' in result_df.columns:
        # Adjust 'pred_vs_act' based on 'direction'
        result_df['adjusted_pred_vs_act'] = result_df.apply(
            lambda row: abs(row['pred_vs_act']) if row['direction'] else -abs(row['pred_vs_act']), axis=1
        )
        
        
        # Add a new subplot for the adjusted 'pred_vs_actual' histogram
        ax[3].hist(result_df['adjusted_pred_vs_act'].dropna(), bins=30, edgecolor='black')
        ax[3].set_title("Histogram of Adjusted Pred vs Actual (Custom Bins)")
        ax[3].set_xlabel("Adjusted Pred vs Actual")
        ax[3].set_ylabel("Frequency")
    else:
        print("'pred_vs_actual' or 'direction' column not found in the dataset.")

    '''
    # Old Histogram for 'rel_deviation'
    if 'rel_deviation' in result_df.columns:
        ax[2].hist(result_df['rel_deviation'].dropna(), bins=20, edgecolor='black')
        ax[2].set_title("Histogram of Relative Deviation")
        ax[2].set_xlabel("Relative Deviation")
        ax[2].set_ylabel("Frequency")
    else:
        print("'rel_deviation' column not found in the dataset.")
    '''

    # Add analysis text at the bottom
    datapoint_count = total_count_direction
    analysis_text = f"Analysed {folder_count} stocks\nAnalysed {datapoint_count} datapoints"
    plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, wrap=True)

    plt.tight_layout()
    plt.savefig("metaval.png")
    print("Plot saved as metaval.png")
    
def plot_r_square(models_dir='models'):
    results = []

    for folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, folder)

        if os.path.isdir(model_path):
            r_squared, direction_percent, in_bounds_percent = analyze_model_folder(model_path)

            # Collect the results
            results.append({
                'Model': folder,
                'R-squared': r_squared,
                'Direction True %': direction_percent,
                'In Bounds True %': in_bounds_percent
            })

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Filter out rows with missing data
    results_df = results_df.dropna()

    # Scatter plot of R-squared vs Direction True % and In Bounds True %
    plt.figure(figsize=(12, 6))

    # Plot R-squared vs Direction True %
    plt.scatter(results_df['R-squared'], results_df['Direction True %'], color='blue', label='Direction True %')

    # Plot R-squared vs In Bounds True %
    plt.scatter(results_df['R-squared'], results_df['In Bounds True %'], color='red', label='In Bounds True %')

    # Labeling the plot
    plt.xlabel('R-squared')
    plt.ylabel('True Percentage (%)')
    plt.title('R-squared vs True Percentages (Direction & In Bounds)')
    plt.legend()
    plt.grid()

    # Save the scatter plot
    plt.savefig('metaval2.png')
    print("Scatter plot saved as metaval2.png")
    
def plot_bar_charts(models_dir='models'):
    results = []

    # Collect data from model folders
    for folder in os.listdir(models_dir):
        model_path = os.path.join(models_dir, folder)

        if os.path.isdir(model_path):
            r_squared, direction_percent, in_bounds_percent = analyze_model_folder(model_path)

            # Collect the results
            if direction_percent is not None and in_bounds_percent is not None:
                results.append({
                    'Model': folder,
                    'Direction True %': direction_percent,
                    'In Bounds True %': in_bounds_percent
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort the DataFrame by 'Direction True %' in descending order
    results_df = results_df.sort_values(by='Direction True %', ascending=False)

    # Create the bar charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Bar chart for Direction True %
    axes[0].barh(results_df['Model'], results_df['Direction True %'], color='blue')
    axes[0].set_xlabel('Percentage (%)')
    axes[0].set_title('Percentage of Correct Predictions (Direction)')
    axes[0].invert_yaxis()  # Invert y-axis to have the highest value at the top

    # Bar chart for In Bounds True %
    axes[1].barh(results_df['Model'], results_df['In Bounds True %'], color='green')
    axes[1].set_xlabel('Percentage (%)')
    axes[1].set_title('Percentage of Predictions In Bounds')
    axes[1].invert_yaxis()  # Invert y-axis to match the order of the first chart

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('metaval3.png')
    print("Bar charts saved as metaval3.png")
    
def analyze_consistent_bounds(result_df):
    """
    Analyzes the 'consistent_bounds' cases and plots a pie chart for 'direction' values.

    Args:
        result_df (pd.DataFrame): The merged validation DataFrame.
    """
    # Check if 'consistent_bounds' and 'direction' columns exist
    if 'consistent_bounds' not in result_df.columns or 'direction' not in result_df.columns:
        print("'consistent_bounds' or 'direction' column not found in the dataset.")
        return

    # Filter for cases where 'consistent_bounds' is True
    filtered_df = result_df[result_df['consistent_bounds'] == True]

    # Count the number of True and False cases for 'direction'
    true_count = filtered_df['direction'].sum()
    false_count = len(filtered_df) - true_count
    total_count = true_count + false_count

    # Calculate percentages for the pie chart
    if total_count > 0:
        true_percentage = (true_count / total_count) * 100
        false_percentage = (false_count / total_count) * 100
    else:
        true_percentage = false_percentage = 0
        
    # Print the number of cases and True percentage to the console
    print(f"Consistent bounds direction true in {true_percentage:.1f}% of the time with {total_count} cases analyzed")

    # Create the pie chart
    labels = ['True', 'False']
    sizes = [true_count, false_count]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title("True vs False Predictions (Consistent Bounds)")

    # Add the total count below the pie chart
    plt.figtext(0.5, 0.01, f"Total datapoints evaluated: {total_count}", ha="center", fontsize=12, wrap=True)

    # Save the pie chart as 'metaval4.png'
    plt.savefig("metaval4.png")
    plt.close()
    print("Pie chart saved as metaval4.png")
    
def plot_true_percentage_per_day(models_dir='models'):
    """
    Iterates through all folders in the models directory, extracts the 'direction' column from 
    validation_set.csv, aggregates data by date, calculates true percentages, and plots a bar chart.
    """
    aggregated_df = None  # Initialize an empty DataFrame to hold combined data

    # Iterate through all folders in the models directory
    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        validation_path = os.path.join(folder_path, 'validation', 'validation_set.csv')

        # Check if the folder and validation file exist
        if os.path.isdir(folder_path) and os.path.exists(validation_path):
            try:
                # Load the validation_set.csv
                df = pd.read_csv(validation_path)

                # Ensure datetime column is in datetime format
                df['datetime'] = pd.to_datetime(df['datetime'])

                # Keep only the 'datetime' and 'direction' columns
                df = df[['datetime', 'direction']]

                # Rename the 'direction' column to the folder name
                df.rename(columns={'direction': folder}, inplace=True)

                # If aggregated_df is None, initialize it with the first DataFrame
                if aggregated_df is None:
                    aggregated_df = df
                else:
                    # Merge on 'datetime' column to align dates
                    aggregated_df = pd.merge(aggregated_df, df, on='datetime', how='outer')

            except Exception as e:
                print(f"Error processing folder {folder}: {e}")

    
    if aggregated_df is not None:
        # Drop rows with missing values
        aggregated_df.dropna(inplace=True)

        # Calculate the true count and total count per date
        aggregated_df['True_Count'] = (aggregated_df.iloc[:, 1:] == True).sum(axis=1)
        aggregated_df['Total_Count'] = len(aggregated_df.columns) - 1  # Exclude datetime column
        aggregated_df['True_Percentage'] = (aggregated_df['True_Count'] / aggregated_df['Total_Count']) * 100
        aggregated_df.to_csv('eval_direction.csv', index=False)
        print(aggregated_df)

        # Plot the True Percentage against Date
        plt.figure(figsize=(12, 8))
        plt.bar(aggregated_df['datetime'], aggregated_df['True_Percentage'], color='blue', alpha=0.7)

        # Customize the plot
        plt.title('True Percentage of Predictions Per Day', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('True Percentage (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot
        plt.savefig('metaval5.png')
        plt.close()
        print("Bar chart saved as metaval_5.png")
    else:
        print("No data found in the specified models directory.")

if __name__ == "__main__":
    result_df, folder_count = merge_validation_sets()
    plot_r_square()
    plot_true_percentage_per_day()
    
    if not result_df.empty:
        analyze_validation_set(result_df, folder_count)
        plot_bar_charts()
        analyze_consistent_bounds(result_df)
    else:
        print("No data to analyze.")
