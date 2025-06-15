import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import talib  # For technical indicators
from functools import reduce
import optuna  # For Bayesian optimization
import matplotlib.pyplot as plt
from DayTrader_tools import load_and_rename_data, create_lagged_features, add_technical_indicators

# Define file paths and tickers
file_paths = {
    'XOM': "stock_data_daily/Toyota_TOYOF-daily.csv",  # Replace with your actual file paths
    'AAPL': "stock_data_daily/Apple-daily.csv"
}

# Load and rename data

# Load data for all tickers
price_data_list = [load_and_rename_data(file_paths[ticker], ticker) for ticker in file_paths]
price_data = reduce(lambda left, right: pd.merge(left, right, on='datetime', how='inner'), price_data_list)

# Sort and fill missing values
price_data.sort_values(by='datetime', inplace=True)
price_data.fillna(method='ffill', inplace=True)

'''
# Create lagged features legacy
# def create_lagged_features(data, ticker, lag_periods):
#     for lag in lag_periods:
#         data[f'{ticker}_lag{lag}'] = data[f'close_{ticker}'].shift(lag)
#     return data
'''

# Function to create lagged features provided by DayTrader_tools

lag_periods = [1, 2, 3, 4, 5]
for ticker in file_paths:
    price_data = create_lagged_features(price_data, ticker, lag_periods)


# Add all technical indicators provided by DayTrader_tools

for ticker in file_paths:
    price_data = add_technical_indicators(price_data, ticker)

# Create target column (shifted 'close' price for next-day prediction)
def create_target_column(data, ticker):
    target_col = f'close_{ticker}'
    data['target'] = data[target_col].shift(-1)
    data.dropna(subset=['target'], inplace=True)
    return data

price_data = create_target_column(price_data, 'XOM')

# Trim dataset
#price_data = price_data.head(490)

# Prepare data for training
def prepare_data(data, ticker, lag_periods):
    """
    Prepares the features and target variable for training the model.
    Includes all technical indicators and lagged features.
    """
    # Define the full list of features
    features = [
        f'MA50_{ticker}', f'EMA20_{ticker}', f'RSI_{ticker}',
        f'upper_bb_{ticker}', f'lower_bb_{ticker}',
        f'ATR_{ticker}', f'MACD_{ticker}', f'MACD_signal_{ticker}',
        f'ADX_{ticker}', f'StochK_{ticker}', f'StochD_{ticker}',
        f'CCI_{ticker}', f'WilliamsR_{ticker}'
    ] + [f'close_{ticker}_lag{lag}' for lag in lag_periods]

    # Use the newly created target column ('target')
    target = 'target'

    # Drop rows with any missing values
    data = data.dropna()

    # Create feature matrix (X) and target vector (y)
    X = data[features]
    y = data[target]

    return X, y

X, y = prepare_data(price_data, 'XOM', lag_periods)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define columns to exclude
exclude_cols = ["datetime", "target", f"close_{ticker}"]

# Identify feature columns by excluding non-feature columns
feature_cols = [col for col in price_data.columns if col not in exclude_cols]

# Create feature matrix (X) and target vector (y)
X = price_data[feature_cols]
y = price_data['target']

# Check alignment between features and target for training data
assert X_train.shape[0] == y_train.shape[0], "Mismatch in row numbers between features and target in training data"

# Check alignment between features and target for test data
assert X_test.shape[0] == y_test.shape[0], "Mismatch in row numbers between features and target in test data"


# Convert training and test data to DMatrix
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter space
    param = {
        'booster': 'gbtree',
        'tree_method': 'hist',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'device': 'cuda',  # Use GPU if available
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5)
    }

    # Perform cross-validation with early stopping
    cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=500,
        nfold=5,
        early_stopping_rounds=10,
        metrics="rmse",
        as_pandas=True,
        verbose_eval=False
    )

    # Return the minimum RMSE (negative for Optuna's maximization)
    return cv_results['test-rmse-mean'].min()

# Set up and run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# Extract the best parameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Define final parameters using best_params
final_params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'device': 'cuda',  # Use GPU if available
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'max_depth': best_params['max_depth'],
    'eta': best_params['eta'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'min_child_weight': best_params['min_child_weight'],
    'gamma': best_params['gamma']
}

# Define a watchlist for monitoring training and validation performance
watchlist = [(dtrain, 'train'), (dtest, 'eval')]

# Train the final model using xgb.train
final_model = xgb.train(
    params=final_params,
    dtrain=dtrain,
    num_boost_round=300,
    evals=watchlist,
    early_stopping_rounds=10,
    verbose_eval=True
)

# Save the trained final model
final_model.save_model("final_xgb_model.json")
print("Final model saved as 'final_xgb_model.json'")

# Step 6.1: Make predictions on the test set
preds = final_model.predict(dtest)

# Step 6.2: Calculate RMSE for model evaluation
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Test RMSE: {rmse:.2f}")

# Step 6.3: Calculate R-squared for model evaluation
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)  # Total sum of squares
ss_residual = np.sum((y_test - preds) ** 2)         # Residual sum of squares
r_squared = 1 - (ss_residual / ss_total)
print(f"Test R-squared: {r_squared:.2f}")

# Step 6.4: Calculate feature importance
feature_cols = [col for col in X_test.columns]  # Get feature names
importance_matrix = final_model.get_score(importance_type='weight')

# Convert importance to DataFrame for easier visualization
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': [importance_matrix.get(col, 0) for col in feature_cols]
}).sort_values(by='Importance', ascending=False)

# Display the feature importance matrix
print(importance_df)

# Step 6.5: Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Top 15 Feature Importance')
plt.xlabel('Importance Score')
plt.show()

# Step 6.6: Combine test data with predictions for plotting
test_data = X_test.copy()
test_data['target'] = y_test
test_data['predicted'] = preds

# Ensure 'datetime' is included and sorted
test_data['datetime'] = price_data.loc[X_test.index, 'datetime']
test_data.sort_values(by='datetime', inplace=True)

# Plot the actual vs. predicted values over time
import matplotlib.pyplot as plt

# Step 6.7: Plot the actual vs. predicted values over time
plt.figure(figsize=(12, 6))
plt.plot(test_data['datetime'], test_data['target'], label='Actual', color='blue', linewidth=0.5)
plt.plot(test_data['datetime'], test_data['predicted'], label='Predicted', color='red', linewidth=0.5)
plt.title('Actual vs. Predicted Values Over Time')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Step 7.1: Set up a U.S. stock market calendar
us_calendar = mcal.get_calendar('NYSE')

# Define U.S. holidays (if you want to add custom holidays)
us_holidays = pd.to_datetime(["2024-01-01", "2024-12-25"])

# Step 7.2: Determine the last date in the dataset
last_date = pd.to_datetime(price_data['datetime'].max())

# Step 7.3: Generate only trading days for the forecast
# Create a date range for the next 30 days
date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Filter to include only trading days using the market calendar
trading_days = us_calendar.valid_days(start_date=last_date, end_date=last_date + pd.Timedelta(days=30))
trading_days = trading_days[~trading_days.isin(us_holidays)]

# Select only the first 5 trading days
future_dates = trading_days[:5]

# Step 7.4: Initialize `future_data` with the trading days
future_data = pd.DataFrame({'datetime': future_dates})

# Display the generated future trading days
print("Future Trading Days:")
print(future_data)

#line 316 until 370 in R code
# Initialize a list to store predictions
future_predictions = []

# Define the number of future days to predict
num_future_days = 5

# Copy the last known data for initialization
last_known_data = price_data.iloc[-1].copy()

# Define the target column (e.g., 'close_XOM')
target_col = 'close_XOM'

# Iterate over each future day
for i in range(num_future_days):
    if i == 0:
        # For the first day, use the last known data
        current_data = last_known_data.copy()
    else:
        # For subsequent days, use the previous day's data
        current_data = future_data.iloc[i - 1].copy()

    # Update lagged features
    for lag in lag_periods:
        lag_col = f'{target_col}_lag{lag}'
        if i < lag:
            # Use the last known lagged value from historical data
            current_data[lag_col] = price_data[lag_col].iloc[-1]
        else:
            # Use the previous predictions
            current_data[lag_col] = future_predictions[i - lag]

    # Recalculate technical indicators
    current_data[f'MA50_XOM'] = price_data[target_col].rolling(window=50).mean().iloc[-1]
    current_data[f'EMA20_XOM'] = price_data[target_col].ewm(span=20, adjust=False).mean().iloc[-1]
    current_data[f'RSI_XOM'] = talib.RSI(price_data[target_col], timeperiod=14).iloc[-1]

    upper_band, middle_band, lower_band = talib.BBANDS(price_data[target_col], timeperiod=20)
    current_data[f'upper_bb_XOM'] = upper_band.iloc[-1]
    current_data[f'lower_bb_XOM'] = lower_band.iloc[-1]

    current_data[f'ATR_XOM'] = talib.ATR(price_data['high_XOM'], price_data['low_XOM'], price_data[target_col], timeperiod=14).iloc[-1]
    macd, macd_signal, _ = talib.MACD(price_data[target_col], fastperiod=12, slowperiod=26, signalperiod=9)
    current_data[f'MACD_XOM'] = macd.iloc[-1]
    current_data[f'MACD_signal_XOM'] = macd_signal.iloc[-1]

    current_data[f'ADX_XOM'] = talib.ADX(price_data['high_XOM'], price_data['low_XOM'], price_data[target_col], timeperiod=14).iloc[-1]
    slowk, slowd = talib.STOCH(price_data['high_XOM'], price_data['low_XOM'], price_data[target_col], fastk_period=14, slowk_period=3, slowd_period=3)
    current_data[f'StochK_XOM'] = slowk.iloc[-1]
    current_data[f'StochD_XOM'] = slowd.iloc[-1]

    current_data[f'CCI_XOM'] = talib.CCI(price_data['high_XOM'], price_data['low_XOM'], price_data[target_col], timeperiod=20).iloc[-1]
    current_data[f'WilliamsR_XOM'] = talib.WILLR(price_data['high_XOM'], price_data['low_XOM'], price_data[target_col], timeperiod=14).iloc[-1]

    # Prepare the feature vector for prediction
    feature_vector = current_data[feature_cols].values.reshape(1, -1)

    # Convert the feature vector to DMatrix format
    dfuture = xgb.DMatrix(data=feature_vector, feature_names=feature_cols)

    # Make the prediction
    prediction = final_model.predict(dfuture)[0]

    # Store the prediction
    future_predictions.append(prediction)
    future_data.loc[i, 'target'] = prediction

    # Update the target column in future_data for use in lagged features
    future_data.loc[i, target_col] = prediction

# Display the future predictions
print("Future Predictions:", future_predictions)
print(future_data[['datetime', 'target']])

#R line 372 until 405
# Create a DataFrame for the forecast results
forecast_results = pd.DataFrame({
    'datetime': future_data['datetime'],
    'target': np.nan,
    'predicted': future_predictions
})

# Ensure 'datetime' columns are in datetime format
forecast_results['datetime'] = pd.to_datetime(forecast_results['datetime'])
test_data['datetime'] = pd.to_datetime(test_data['datetime'])

# Ensure 'predicted' columns are numeric
test_data['predicted'] = pd.to_numeric(test_data['predicted'], errors='coerce')
forecast_results['predicted'] = pd.to_numeric(forecast_results['predicted'], errors='coerce')

# Combine the test data and forecast results
combined_results = pd.concat([
    test_data[['datetime', 'target', 'predicted']],
    forecast_results[['datetime', 'target', 'predicted']]
], ignore_index=True)

# Step 2: Plot the Actual vs. Predicted Values Including Forecast
plt.figure(figsize=(12, 6))
plt.plot(combined_results['datetime'], combined_results['target'], label='Actual', color='blue', linewidth=0.5)
plt.plot(combined_results['datetime'], combined_results['predicted'], label='Predicted', color='red', linewidth=0.5)
plt.title("Actual vs. Predicted Values Including Forecast")
plt.xlabel("Datetime")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

#R line 409 until 494
# Step 1: Set the number of simulations for bootstrapping
num_simulations = 100

# Initialize an array to store the forecasts from each simulation
simulation_results = np.empty((5, num_simulations))

# Define the same parameter bounds for training in each iteration
bootstrap_params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'device': 'cuda',
    'max_depth': best_params['max_depth'],
    'eta': best_params['eta'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'min_child_weight': best_params['min_child_weight'],
    'gamma': best_params['gamma']
}

# Step 2: Run simulations with model re-training
for sim in range(num_simulations):
    # Re-sample the training data with replacement
    train_data_sample = X_train.sample(n=len(X_train), replace=True)
    y_train_sample = y_train.loc[train_data_sample.index]
    dtrain_sample = xgb.DMatrix(data=train_data_sample, label=y_train_sample, feature_names=feature_cols)

    # Train a new model on the re-sampled training data
    model = xgb.train(params=bootstrap_params, dtrain=dtrain_sample, num_boost_round=200, verbose_eval=False)

    # Reinitialize future_data with last known data
    future_data = pd.DataFrame({'datetime': future_dates})
    for col in feature_cols:
        future_data[col] = np.nan
    # Filter feature_cols to include only existing columns in last_known_data
    existing_cols = [col for col in feature_cols if col in last_known_data.index]

    # Assign values individually
    for col in existing_cols:
        future_data.at[0, col] = last_known_data[col]

    # Store predictions for this simulation
    future_predictions = []

    # Forecast for each future day using the new model
    for i in range(5):
        if i == 0:
            current_data = last_known_data.copy()
        else:
            current_data = future_data.iloc[i - 1].copy()

        # Update lagged features
        for lag in lag_periods:
            lag_col = f'close_XOM_lag{lag}'
            if i < lag:
                current_data[lag_col] = price_data[lag_col].iloc[-1]
            else:
                current_data[lag_col] = future_predictions[i - lag]

        # Recalculate technical indicators
        current_data[f'MA50_XOM'] = price_data['close_XOM'].rolling(window=50).mean().iloc[-1]
        current_data[f'EMA20_XOM'] = price_data['close_XOM'].ewm(span=20, adjust=False).mean().iloc[-1]

        # Prepare the feature vector for prediction
        feature_vector = current_data[feature_cols].values.reshape(1, -1)
        dfuture = xgb.DMatrix(data=feature_vector, feature_names=feature_cols)

        # Make the prediction
        prediction = model.predict(dfuture)[0]
        future_predictions.append(prediction)
        future_data.loc[i, 'close_XOM'] = prediction

    # Store the predictions from this simulation in the matrix
    simulation_results[:, sim] = future_predictions

# Step 3: Calculate the Mean and 95% Confidence Interval
forecast_means = np.mean(simulation_results, axis=1)
forecast_lower = np.percentile(simulation_results, 2.5, axis=1)
forecast_upper = np.percentile(simulation_results, 97.5, axis=1)

# Step 4: Combine Results into a DataFrame
forecast_df = pd.DataFrame({
    'datetime': future_dates,
    'mean_forecast': forecast_means,
    'lower_bound': forecast_lower,
    'upper_bound': forecast_upper
})

# Display the forecast with confidence intervals
print(forecast_df)

# Step 5: Plot the Forecast with Confidence Intervals
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['datetime'], forecast_df['mean_forecast'], label='Mean Forecast', color='red')
plt.fill_between(forecast_df['datetime'], forecast_df['lower_bound'], forecast_df['upper_bound'], color='gray', alpha=0.3, label='95% Confidence Interval')
plt.title("Bootstrapped Forecast with 95% Confidence Interval")
plt.xlabel("Datetime")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

#R line 496 until 548
# Step 6.1: Combine Results for Plotting
forecast_confidence = pd.DataFrame({
    'datetime': future_dates,
    'forecast_mean': forecast_means,
    'lower_95': forecast_lower,
    'upper_95': forecast_upper
})

# Rename the 'forecast_mean' column to 'predicted'
forecast_confidence.rename(columns={'forecast_mean': 'predicted'}, inplace=True)

# Ensure 'datetime' columns are in datetime format
forecast_confidence['datetime'] = pd.to_datetime(forecast_confidence['datetime'])
test_data['datetime'] = pd.to_datetime(test_data['datetime'])

# Step 6.2: Combine Test Data with Forecasted Confidence Intervals
combined_results = pd.concat([
    test_data[['datetime', 'target', 'predicted']],
    forecast_confidence[['datetime', 'predicted']].assign(target=np.nan)
], ignore_index=True)

# Step 6.3: Plot the Forecast with 95% Confidence Intervals
plt.figure(figsize=(12, 6))
# Plot actual data
plt.plot(combined_results['datetime'], combined_results['target'], label='Actual', color='blue', linewidth=0.5)
# Plot forecasted mean
plt.plot(forecast_confidence['datetime'], forecast_confidence['predicted'], label='Forecasted Mean', color='red', linewidth=0.5)
# Plot confidence interval
plt.fill_between(forecast_confidence['datetime'], forecast_confidence['lower_95'], forecast_confidence['upper_95'], color='green', alpha=0.3, label='95% Confidence Interval')

plt.title("Actual vs. Forecasted Values with 95% Confidence Interval")
plt.xlabel("Datetime")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Step 6.4: Calculate Expected Return Based on Forecast
# Get the closing price on the last day of historical data
last_closing_price = test_data['target'].iloc[-1]

# Get the forecasted mean, lower, and upper values for the first forecasted day
first_forecast_mean = forecast_means[0]
first_forecast_lower = forecast_lower[0]
first_forecast_upper = forecast_upper[0]

# Calculate expected return based on mean forecast
expected_return_mean = (first_forecast_mean - last_closing_price) / last_closing_price * 100

# Calculate expected return range based on confidence interval
expected_return_lower = (first_forecast_lower - last_closing_price) / last_closing_price * 100
expected_return_upper = (first_forecast_upper - last_closing_price) / last_closing_price * 100

# Print the results
print(f"Expected Return (Mean): {expected_return_mean:.2f}%")
print(f"Expected Return Range (95% Confidence Interval): {expected_return_lower:.2f}% to {expected_return_upper:.2f}%")