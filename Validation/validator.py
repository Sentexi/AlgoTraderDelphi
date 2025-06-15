import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from DayTrader_XGBoost_module import run_analysis
from DayTrader_tools import create_lagged_features, add_technical_indicators, load_and_rename_data
import pytz

'''
# Provide the path to the CSV file
csv_file_path = "stock_data_daily/" + "Coca_Cola_KO-daily.csv"
run_analysis(csv_file_path)
'''

# Define the input and output folders
input_folder = "validation_stock_data_daily_raw"
output_folder = "validation_stock_data_daily_prepared"
models_folder = "models"

# Cutoff date (default was September 30, 2024) aka (2024, 9, 30)
cutoff_date = datetime(2024, 10, 31)

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each file in the input folder
for filename in os.listdir(input_folder):
    # Only process CSV files
    if filename.endswith(".csv"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        try:
            # Read the CSV file into a DataFrame, parse datetime column
            df = pd.read_csv(input_file_path, parse_dates=["datetime"])

            # Remove timezone information from the 'datetime' column
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

            # Filter rows based on the cutoff date
            df_filtered = df[df["datetime"] <= cutoff_date]

            # Check if the filtered DataFrame is not empty
            if not df_filtered.empty:
                # Save the filtered DataFrame to the output folder
                df_filtered.to_csv(output_file_path, index=False)
                print(f"Filtered data saved for {filename} in {output_folder}")
            else:
                print(f"No data before the cutoff date in {filename}. Skipping.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
            
#run integrity check to only create data for missing folders
def integrity_check(csv_files, models_folder="models"):
    """
    Check the integrity of the folder structure for each CSV file.

    Args:
        csv_files (list): List of CSV filenames.
        models_folder (str): Path to the 'models' folder.

    Returns:
        List of missing CSV files (where folder structure does not exist).
    """
    missing_files = []

    for filename in os.listdir(csv_files):
        # Prepare model folder paths
        model_folder_name = filename[:-4]  # Remove ".csv" extension
        model_folder_path = os.path.join(models_folder, model_folder_name)
        cut_folder_path = os.path.join(model_folder_path, "cut")
        val_folder_path = os.path.join(model_folder_path, "val")

        # Check if all required folders exist
        if os.path.isdir(model_folder_path) and os.path.isdir(cut_folder_path) and os.path.isdir(val_folder_path):
            print(f"Integrity check passed: All folders exist for {filename}.")
        else:
            print(f"Integrity check failed: Missing folders for {filename}.")
            missing_files.append(filename)

    return missing_files     

missing_files = integrity_check(output_folder, models_folder="models")

# Loop through each file in the folder
for filename in missing_files:
    # Only process CSV files
    if filename.endswith(".csv"):
        csv_file_path_cut = os.path.join(output_folder, filename)
        csv_file_path_val = os.path.join("validation_stock_data_daily_raw", filename)

        try:
            run_analysis("cut",csv_file_path_cut)
            run_analysis("val",csv_file_path_val)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
            
if missing_files == []:
    print("All files are in place")
    
    
#perform validation
def incremental_prediction(file_name, models_folder="models"):
    #how many days are we verifying
    simulation_days = 25
    dates = []
    values = []
    values_upper = []
    values_lower = []

    """
    Perform incremental prediction using the XGBoost model for a given stock data file.

    Args:
        file_name (str): The name of the CSV file (without .csv extension).
        models_folder (str): Path to the 'models' folder.

    Returns:
        None
    """
    
    #try:
    # Define paths
    model_folder_path = os.path.join(models_folder, file_name)
    cut_folder_path = os.path.join(model_folder_path, "cut")
    val_folder_path = os.path.join(model_folder_path, "val")

    # Load training data from 'cut' folder
    cut_data_undated = os.path.join(cut_folder_path, f"{file_name}_training_data.csv")
    cut_data_path = os.path.join(cut_folder_path, f"{file_name}_training_data_dated.csv")
    cut_data = pd.read_csv(cut_data_path, parse_dates=["datetime"])
    undated = pd.read_csv(cut_data_undated)

    # Load validation data from 'val' folder
    val_data_undated = os.path.join(val_folder_path, f"{file_name}_training_data.csv")
    val_data_path = os.path.join(val_folder_path, f"{file_name}_training_data_dated.csv")
    val_data = pd.read_csv(val_data_path, parse_dates=["datetime"])
    #val_undated = pd.read_csv(val_data_undated) we probably dont need this
    #we create a copy of val_data for the values we will extract later
    val_data_init = val_data.copy()
    #resolving timezone shit
    val_data_init["datetime"] = val_data_init["datetime"].dt.tz_localize(None)

    '''
    # Load the XGBoost model
    model_path = os.path.join(val_folder_path, "DayTrader_XGBoost.json")
    model = xgb.Booster()
    model.load_model(model_path)
    '''
    
    # Load all models from the "submodules" folder
    submodules_path = os.path.join(val_folder_path, "submodules")
    model_files = sorted([f for f in os.listdir(submodules_path) if f.startswith("model_") and f.endswith(".json")])
    
    def iteration_step(cut_data, val_data, undated, dates, values, values_upper, values_lower):

        # Set the working date as the last date in the 'cut' data
        working_date = cut_data["datetime"].max()

        # Remove timezone information from the 'datetime' column
        val_data["datetime"] = val_data["datetime"].dt.tz_localize(None)

        # Get the next date for prediction from the 'val' data
        next_data = val_data[val_data["datetime"] > working_date]
        if next_data.empty:
            print(f"No data available after the working date {working_date}. Skipping prediction.")
            return
        
        #compare the columns and drop the overlap
        # Identify columns that are unique to each DataFrame
        columns_to_drop = list(set(cut_data.columns) - set(undated.columns))
        #The target column needs to be removed as well, honestly this is messier than I wanted it to be
        columns_to_drop.append("target")

        predict_date = next_data["datetime"].min()
        print(f"Predicting for the next date: {predict_date}")

        # Prepare features for prediction (assuming all columns except 'datetime' are features)
        features = cut_data.drop(columns=columns_to_drop)
        
        #apparently we need to move around some columns for some reason
        # List of columns to move to the end
        cols_to_move = [
            'close_XOM_lag1',
            'close_XOM_lag2',
            'close_XOM_lag3',
            'close_XOM_lag4',
            'close_XOM_lag5'
        ]

        # Reorder the DataFrame
        features = features[[col for col in features.columns if col not in cols_to_move] + cols_to_move]
        
        #after rearranging the columns as they should be we can finally make a dmatrix and a prediction
        dmatrix = xgb.DMatrix(features)
        
        #print(features)

        # Collect predictions from all models
        simulation_results = []

        for model_file in model_files:
            model_path = os.path.join(submodules_path, model_file)
            model = xgb.Booster()
            model.load_model(model_path)
            
            # Perform prediction for the current model
            prediction = model.predict(dmatrix)
            simulation_results.append(prediction[-1])

        # Calculate statistics
        forecast_mean = np.mean(simulation_results)
        forecast_lower = np.percentile(simulation_results, 2.5)
        forecast_upper = np.percentile(simulation_results, 97.5)
        create_histogram(simulation_results, file_name, working_date)
        
        '''
        #This prints all predictions from the fit
        print("printing prediction matrix")
        print(prediction)
        '''

        print(f"Prediction for {predict_date}: Mean = {forecast_mean}, 2.5th Percentile = {forecast_lower}, 97.5th Percentile = {forecast_upper}")
        # Append the results
        dates.append(predict_date)
        values.append(forecast_mean)
        values_upper.append(forecast_upper)
        values_lower.append(forecast_lower)
        
        #we also append forecast_lower and forecast_upper
        return cut_data, val_data, undated, dates, values, predict_date, values_upper, values_lower
        
    for i in range(simulation_days):
        cut_data, val_data, undated, dates, values, predict_date, values_upper, values_lower = iteration_step(cut_data, val_data, undated, dates, values, values_upper, values_lower)
    
        #now remove the row with the date from val_data after appending it to cut_data
        # 1. Filter the row from 'val_data' based on 'predict_date'
        row_to_append = val_data[val_data["datetime"] == predict_date]

        # Check if the row exists (it should, but just in case)
        if row_to_append.empty:
            print(f"No row found for the prediction date {predict_date}. Skipping append.")
        else:
            # 2. Append the row to 'cut_data'
            cut_data = pd.concat([cut_data, row_to_append], ignore_index=True)
            print(f"Appended row for date {predict_date} to 'cut_data'.")

            # 3. Remove the row from 'val_data'
            val_data = val_data[val_data["datetime"] != predict_date]
            print(f"Removed row for date {predict_date} from 'val_data'.")
            

    #Creates a dataframe with the relevant headers
    validation_set = pd.DataFrame({
    "datetime": dates,
    "prediction": values,
    "pred_vs_act": np.nan, #Initialise prediction vs actual of previous day
    "actual_rel_diff": np.nan, #Initialise actual rel diff
    "rel_deviation": np.nan,  # Initialize 'rel_deviation' with NaN
    "direction": False,  # Initialize 'direction' with False
    "forecast_lower": values_lower, #Initialize lower bounds of forecast
    "forecast_upper": values_upper, #Initialize upper bounds of forecast
    "in_bounds": False  # Initialize 'in_bounds' with False
    })
    
    # Ensure 'datetime' columns are properly formatted
    val_data_init["datetime"] = pd.to_datetime(val_data_init["datetime"])
    validation_set["datetime"] = pd.to_datetime(validation_set["datetime"])
    
    # Step 4: Extract the 'close_XOM' column and fill the 'actual' column
    validation_set = pd.merge(
        validation_set,
        val_data_init[["datetime", "close_XOM"]],
        on="datetime",
        how="left"
    )

    # Rename the 'close_XOM' column to 'actual'
    validation_set.rename(columns={"close_XOM": "actual"}, inplace=True)
    
    #Evaluate the correct direction of the prediction
    # Set the first value of 'direction' to NaN
    validation_set.loc[0, "direction"] = np.nan
    
    # Loop through the DataFrame to evaluate the direction of change and the relative deviation
    for i in range(1, len(validation_set)):
        # Check direction for 'prediction' values
        prediction_direction = validation_set.loc[i, "prediction"] > validation_set.loc[i - 1, "actual"]      
        # Check direction for 'actual' values
        actual_direction = validation_set.loc[i, "actual"] > validation_set.loc[i - 1, "actual"]
        # Update 'direction' column: True if both directions are the same, False otherwise
        validation_set.loc[i, "direction"] = prediction_direction == actual_direction
        
        #Calculate deviation prediction vs actual of previous day
        prediction_diff = validation_set.loc[i, "prediction"] - validation_set.loc[i - 1, "actual"]
        # Update 'pred_vs_act' column
        validation_set.loc[i, "pred_vs_act"] = (prediction_diff / validation_set.loc[i - 1, "actual"])*100
        #Calculate deviation actual vs actual of previous day
        actual_diff = validation_set.loc[i, "actual"] - validation_set.loc[i - 1, "actual"]
        # Update 'actual_rel_diff' column
        validation_set.loc[i, "actual_rel_diff"] = (actual_diff / validation_set.loc[i - 1, "actual"])*100
        #create the difference and divide by the actual value of previous day
        rel_diff = np.abs(((actual_diff - prediction_diff) / validation_set.loc[i - 1, "actual"])*100)
        # Update 'rel_deviation' column
        validation_set.loc[i, "rel_deviation"] = rel_diff

        # check if the previous day's actual value is within forecast bounds
        validation_set.loc[i, "in_bounds"] = (
            validation_set.loc[i, "forecast_lower"] <= validation_set.loc[i - 1, "actual"] <= validation_set.loc[i, "forecast_upper"]
        )
        
        # Check if both forecast_upper and forecast_lower are either above or below the actual value of the previous day
        previous_actual = validation_set.loc[i - 1, "actual"]
        forecast_upper = validation_set.loc[i, "forecast_upper"]
        forecast_lower = validation_set.loc[i, "forecast_lower"]

        # Add a new column to indicate if the bounds are consistently above or below the previous actual value
        validation_set.loc[i, "consistent_bounds"] = (
            (forecast_upper > previous_actual and forecast_lower > previous_actual) or
            (forecast_upper < previous_actual and forecast_lower < previous_actual)
)
        
        
    print(validation_set)
    
    #Finally save the validation file:
    validation_folder_path = os.path.join(model_folder_path, "validation")
    os.makedirs(validation_folder_path, exist_ok=True)
    validation_csv_file_path = os.path.join(validation_folder_path, "validation_set.csv")
    validation_set.to_csv(validation_csv_file_path, index=False)
    
    # Plot the predictions and actual values
    plt.figure(figsize=(12, 6))
    plt.plot(validation_set["datetime"], validation_set["prediction"], color='red', label='Prediction')
    plt.plot(validation_set["datetime"], validation_set["actual"], color='blue', label='Actual')
    
    # Fill the area between forecast_lower and forecast_upper with light green color
    plt.fill_between(
        validation_set["datetime"],
        validation_set["forecast_lower"],
        validation_set["forecast_upper"],
        color='lightgreen',
        alpha=0.3,
        label='Forecast Bounds'
    )

    # Set labels, title, and legend
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.title("Prediction vs Actual Values Over Time")
    plt.legend()
    plt.grid()
    
    # Save the plot to the specified folder and close it
    chart_path = os.path.join(validation_folder_path, "chart.png")
    plt.savefig(chart_path)
    plt.close()

    print(f"Plot saved as {chart_path}")
    
def create_histogram(simulation_results, file_name, working_date):
    """
    Creates a histogram of the simulation results and saves it as an image file.

    Args:
        simulation_results (list): List of simulation results.
        file_name (str): Name of the file being processed.
        working_date (datetime): Date associated with the simulation.
    """
    # Ensure the histograms folder exists
    histograms_folder = "histograms"
    os.makedirs(histograms_folder, exist_ok=True)

    # Format the working date for the filename
    formatted_date = working_date.strftime("%Y-%m-%d")

    # Create the file path
    histogram_file_path = os.path.join(histograms_folder, f"{file_name}_{formatted_date}_histogram.png")

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(simulation_results, bins=40, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Simulation Results Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)

    # Save the histogram
    plt.savefig(histogram_file_path)
    plt.close()

    print(f"Histogram saved as {histogram_file_path}")

'''
    except Exception as e:
        print(f"Error during incremental prediction for {file_name}: {e}")
'''
        
#prediction = incremental_prediction(file_name, models_folder="models")
#prediction = incremental_prediction("ABBV-daily", models_folder="models")


# Iterate over all folders in the 'models' directory
for folder in os.listdir('models'):
    folder_path = os.path.join('models', folder)
    if os.path.isdir(os.path.join('models', folder)):
        # Check if it is a directory and the 'validation' folder does not exist
        if os.path.isdir(folder_path) and not os.path.exists(os.path.join(folder_path, 'validation')):
            incremental_prediction(folder, models_folder="models")
