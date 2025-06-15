# Load necessary libraries
library(dplyr)
library(TTR)          # For technical indicators
library(xgboost)      # For the XGBoost model
library(ParBayesianOptimization)  # For Bayesian Optimization
library(tidyr)
library(purrr)        # For reduce function (merging data frames)
library(zoo)
library(modeltime)
library(tidymodels)
library(quantmod)
library(bizdays)


# Define file paths and tickers
file_paths <- list(
  XOM = "G:/Meine Ablage/LM DayTrader/stock_data_daily/Toyota_TOYOF-daily.csv")

# Initialize an empty list to store each stock or currency data frame
price_data_list <- list()

# Load each file, add a stock identifier, and prefix columns
for (ticker in names(file_paths)) {
  # Read CSV and assign to data frame
  df <- read.csv(file_paths[[ticker]])
  
  # Add a stock identifier to each relevant column
  df <- df %>%
    rename(
      !!paste0("volume_", ticker) := v,
      !!paste0("vw_", ticker) := vw,
      !!paste0("open_", ticker) := o,
      !!paste0("close_", ticker) := c,
      !!paste0("high_", ticker) := h,
      !!paste0("low_", ticker) := l,
      !!paste0("trades_", ticker) := n
    ) %>%
    select(datetime, starts_with("volume"), starts_with("vw"), starts_with("open"),
           starts_with("close"), starts_with("high"), starts_with("low"), starts_with("trades"))
  
  # Store each processed data frame in the list
  price_data_list[[ticker]] <- df
}

# Merge all data frames on datetime using a full join
price_data <- reduce(price_data_list, inner_join, by = "datetime")

# Sort by datetime to ensure chronological order
price_data <- price_data %>% arrange(datetime)

# Fill any missing values forward (if needed for continuous time series analysis)
price_data <- price_data %>% fill(everything(), .direction = "down") |> head(490)

# Set the target column and the lag/feature settings
lag_periods <- c(1, 2, 3, 4, 5)        # Lag periods in minutes
sma_windows <- c(1, 5, 10, 20)       # Moving average windows in minutes
target_col <- "close_XOM"         # Use closing price of Exxon Mobil as target

### Step 1: Feature Engineering

# Function to create lagged features for a given stock
create_lagged_features <- function(df, stock, lag_periods) {
  for (lag in lag_periods) {
    df[[paste0(stock, "_lag", lag)]] <- dplyr::lag(df[[stock]], lag)
  }
  return(df)
}

add_quantmod_indicators <- function(df, stock) {
  # Convert to xts format using the datetime column
  df_xts <- xts(df[[stock]], order.by = as.POSIXct(df$datetime, format = "%Y-%m-%d %H:%M:%S"))
  
  # Ensure necessary columns are available for each indicator
  if (all(c(paste0("high_", stock), paste0("low_", stock), paste0("close_", stock)) %in% colnames(df))) {
    # Create an xts object with High, Low, and Close columns for indicators that need them
    HLC_xts <- xts(df[, c(paste0("high_", stock), paste0("low_", stock), paste0("close_", stock))],
                   order.by = as.POSIXct(df$datetime, format = "%Y-%m-%d %H:%M:%S"))
    
    # Bollinger Bands (requires close prices)
    bbands <- BBands(HLC_xts[, "close"])
    df[[paste0(stock, "_BB_upper")]] <- coredata(bbands$up)
    df[[paste0(stock, "_BB_middle")]] <- coredata(bbands$mavg)
    df[[paste0(stock, "_BB_lower")]] <- coredata(bbands$dn)
    
    # Average True Range (ATR)
    atr <- ATR(HLC_xts, n = 14)
    df[[paste0(stock, "_ATR")]] <- coredata(atr$atr)
  }
  
  # MACD (requires only closing price)
  macd <- MACD(df_xts, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
  df[[paste0(stock, "_MACD")]] <- coredata(macd$macd)
  df[[paste0(stock, "_MACD_signal")]] <- coredata(macd$signal)
  
  # ADX (requires high, low, close columns)
  if (exists("HLC_xts")) {
    adx <- ADX(HLC_xts, n = 14)
    df[[paste0(stock, "_ADX")]] <- coredata(adx$ADX)
  }
  
  # Stochastic Oscillator (requires high, low, close)
  if (exists("HLC_xts")) {
    stoch <- stoch(HLC_xts, nFastK = 14, nFastD = 3, nSlowD = 3)
    df[[paste0(stock, "_StochK")]] <- coredata(stoch$fastK)
    df[[paste0(stock, "_StochD")]] <- coredata(stoch$fastD)
  }
  
  # Commodity Channel Index (CCI)
  if (exists("HLC_xts")) {
    cci <- CCI(HLC_xts, n = 20)
    df[[paste0(stock, "_CCI")]] <- coredata(cci)
  }
  
  # Williams %R
  if (exists("HLC_xts")) {
    willr <- WPR(HLC_xts, n = 14)
    df[[paste0(stock, "_WilliamsR")]] <- coredata(willr)
  }
  
  return(df)
}

# Apply feature engineering for each stock
stocks <- setdiff(colnames(price_data), "datetime")  # Exclude datetime column
for (stock in stocks) {
  price_data <- create_lagged_features(price_data, stock, lag_periods)
  price_data <- add_quantmod_indicators(price_data, stock)
}

# Remove rows with NA values introduced by new indicators
price_data <- na.omit(price_data)

price_data <- price_data %>%
  mutate(target = lead(!!sym(target_col), 1)) %>%
  na.omit()  # Remove rows with NA in the target

### Step 2: Update Column Selection and Data Preparation

# Identify columns to keep (those that contain "_lag", "_SMA", "_BB", "_MACD", etc.)
columns_to_keep <- grep("(_lag|datetime|target|_SMA|_BB|_MACD|_ADX|_Stoch|_CCI|_ATR|_WilliamsR)", names(price_data), value = TRUE)

# Subset the data frame to retain only the columns we want
price_data <- price_data %>% select(all_of(columns_to_keep))

# Split data into training and test sets (80% train, 20% test)
train_index <- sample(seq_len(nrow(price_data)), size = 0.8 * nrow(price_data))
train_data <- price_data[train_index, ]
test_data <- price_data[-train_index, ]

# Step 3: Convert Data to DMatrix for XGBoost

# Remove non-feature columns
feature_cols <- setdiff(names(price_data), c("datetime", "target", target_col))

# Check if the rows of train_data and test_data align for features and target
if (nrow(train_data[, feature_cols]) != length(train_data$target)) {
  stop("Mismatch in row numbers between features and target in training data")
}

if (nrow(test_data[, feature_cols]) != length(test_data$target)) {
  stop("Mismatch in row numbers between features and target in test data")
}


# Convert training and test data to DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, feature_cols]), label = train_data$target)
dtest <- xgb.DMatrix(data = as.matrix(test_data[, feature_cols]), label = test_data$target)

### Step 4: Set Up Bayesian Optimization for Hyperparameter Tuning

# Define the objective function for Bayesian optimization
bayes_xgb_opt <- function(max_depth, eta, subsample, colsample_bytree, min_child_weight, gamma) {
  params <- list(
    booster = "gbtree",
    device = "cuda",
    tree_method = "hist",
    objective = "reg:squarederror",
    max_depth = as.integer(max_depth),
    eta = eta,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    eval_metric = "rmse"
  )
  
  xgb_cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 500,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  list(Score = -min(xgb_cv$evaluation_log$test_rmse_mean), Pred = 0)
}

# Define the parameter bounds for optimization
bounds <- list(
  max_depth = c(3L, 100L),
  eta = c(0.01, 0.3),
  subsample = c(0.5, 1.0),
  colsample_bytree = c(0.5, 1.0),
  min_child_weight = c(1, 10),
  gamma = c(0, 5)
)

# Run Bayesian optimization
opt_results <- bayesOpt(
  FUN = bayes_xgb_opt,
  bounds = bounds,
  initPoints = 20,
  iters.n = 10,
  acq = "ucb",
  kappa = 2.576
)

# Extract the best parameters
best_params <- getBestPars(opt_results)

### Step 5: Train Final XGBoost Model with Optimized Parameters

final_params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  device = "cuda",
  tree_method = "hist",
  max_depth = as.integer(best_params$max_depth),
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma,
  eval_metric = "rmse"
)

# Train the final model
final_model <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 300,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

### Step 6: Evaluate the Model

# Make predictions on the test set
preds <- predict(final_model, dtest)

# Calculate RMSE for model evaluation
rmse <- sqrt(mean((preds - test_data$target)^2))
cat("Test RMSE:", rmse, "\n")

# Calculate R-squared for model evaluation
ss_total <- sum((test_data$target - mean(test_data$target))^2)  # Total sum of squares
ss_residual <- sum((test_data$target - preds)^2)               # Residual sum of squares
r_squared <- 1 - (ss_residual / ss_total)
cat("Test R-squared:", r_squared, "\n")

# Calculate feature importance
importance_matrix <- xgb.importance(feature_names = feature_cols, model = final_model)

# Display the feature importance matrix
print(importance_matrix)

# Plot feature importance
xgb.plot.importance(importance_matrix, top_n = 15)  # Change top_n to display more/less features

# Combine test data with predictions for plotting
test_data$predicted <- preds

# Plot the actual vs. predicted values over time
ggplot(test_data, aes(x = as.POSIXct(datetime, format="%Y-%m-%d %H:%M:%S"))) +
  geom_line(aes(y = target, color = "Actual"), linewidth = 0.5) +
  geom_line(aes(y = predicted, color = "Predicted"), linewidth = 0.5) +
  labs(title = "Actual vs. Predicted Values",
       x = "Datetime",
       y = "Price",
       color = "Legend") +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))



# Step 1: Set Up a Business Calendar for Trading Days
# You can add holidays for the U.S. stock market as a vector (if needed)
us_holidays <- as.Date(c("2024-01-01", "2024-12-25"))  # Example holidays, add more as needed

# Create a business calendar that excludes weekends and U.S. holidays
create.calendar(name = "US_CAL", weekdays = c("saturday", "sunday"), holidays = us_holidays)

# Step 2: Determine the Last Date in the Dataset
last_date <- as.Date(tail(price_data$datetime, 1))

# Step 3: Generate Only Trading Days for the Forecast
future_dates <- bizseq(last_date + 1, last_date + 30, "US_CAL")  # Generate up to 30 days to ensure we get 5 trading days
future_dates <- future_dates[1:5]  # Select only the first 5 trading days

# Step 4: Initialize `future_data` with the Trading Days
future_data <- data.frame(datetime = future_dates)


# Get the last row of your price_data
last_known_data <- tail(price_data, 1)

# Initialize future_data with the last known values
for (col in setdiff(names(price_data), c("datetime", "target"))) {
  future_data[[col]] <- NA
  future_data[[col]][1] <- last_known_data[[col]]
}


# Prepare a vector to store predictions
future_predictions <- numeric(5)

# Loop over each future day
for (i in 1:5) {
  # For the first day, features are based on historical data
  if (i == 1) {
    current_data <- last_known_data
  } else {
    # For subsequent days, features are based on previous predictions
    current_data <- future_data[i - 1, ]
  }
  
  # Compute lagged features
  for (lag in lag_periods) {
    lag_col <- paste0(target_col, "_lag", lag)
    if (i - lag <= 0) {
      # Use last known lagged value from price_data
      value <- price_data[[lag_col]][nrow(price_data)]
    } else {
      value <- future_data[[target_col]][i - lag]
    }
    future_data[[lag_col]][i] <- value
  }
  
  # Compute technical indicators (e.g., SMA)
  for (window in sma_windows) {
    sma_col <- paste0(target_col, "_SMA", window)
    # Gather the required number of past values
    if (i <= window) {
      num_historical_needed <- window - (i - 1)
      past_values <- c(tail(price_data[[target_col]], num_historical_needed), future_data[[target_col]][1:(i - 1)])
    } else {
      past_values <- future_data[[target_col]][(i - window):(i - 1)]
    }
    future_data[[sma_col]][i] <- mean(past_values, na.rm = TRUE)
  }
  
  # Prepare the feature vector for prediction
  feature_vector <- future_data[i, feature_cols]
  
  # Ensure the feature vector is in the correct format
  feature_vector_matrix <- as.matrix(feature_vector)
  dfuture <- xgb.DMatrix(data = feature_vector_matrix)
  
  # Make the prediction
  prediction <- predict(final_model, dfuture)
  
  # Store the prediction
  future_data$target[i] <- prediction
  future_predictions[i] <- prediction
  
  # Update the target_col in future_data for use in lagged features
  future_data[[target_col]][i] <- prediction
}

# Step 4: Combine Predictions with Dates
forecast_results <- data.frame(
  datetime = future_data$datetime,
  target = NA_real_,
  predicted = future_predictions
)

# Convert datetime in forecast_results to POSIXct
forecast_results$datetime <- as.POSIXct(forecast_results$datetime, format = "%Y-%m-%d")

# Convert datetime in test_data to POSIXct
test_data$datetime <- as.POSIXct(test_data$datetime, format = "%Y-%m-%d %H:%M:%S")

# Ensure predicted columns are numeric
test_data$predicted <- as.numeric(test_data$predicted)
forecast_results$predicted <- as.numeric(forecast_results$predicted)

# Combine the data frames
combined_results <- bind_rows(
  test_data %>% select(datetime, target, predicted),
  forecast_results %>% select(datetime, target, predicted)
)

# Plot the Actual vs. Predicted Values Including Forecast
ggplot(combined_results, aes(x = datetime)) +
  geom_line(aes(y = target, color = "Actual"), linewidth = 0.5, na.rm = TRUE) +
  geom_line(aes(y = predicted, color = "Predicted"), linewidth = 0.5, na.rm = TRUE) +
  labs(title = "Actual vs. Predicted Values Including Forecast",
       x = "Datetime",
       y = "Price",
       color = "Legend") +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  scale_x_datetime(date_breaks = "90 days", date_labels = "%Y-%m")



# Set the number of simulations for bootstrapping
num_simulations <- 100

# Initialize a matrix to store the forecasts from each simulation
simulation_results <- matrix(NA, nrow = 5, ncol = num_simulations)

# Define the same parameter bounds for training in each iteration
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  tree_method = "hist",
  eval_metric = "rmse",
  device = "cuda",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma
)

# Run simulations with model re-training
for (sim in 1:num_simulations) {
  # Re-sample the training data with replacement
  train_data_sample <- train_data[sample(nrow(train_data), replace = TRUE), ]
  dtrain_sample <- xgb.DMatrix(data = as.matrix(train_data_sample[, feature_cols]), label = train_data_sample$target)
  
  # Train a new model on the re-sampled training data
  model <- xgb.train(params = params, data = dtrain_sample, nrounds = 200, verbose = 0)
  
  # Reinitialize future_data with last known data
  future_data <- data.frame(datetime = future_dates)
  for (col in setdiff(names(price_data), c("datetime", "target"))) {
    future_data[[col]] <- NA
    future_data[[col]][1] <- last_known_data[[col]]
  }
  
  # Store predictions for this simulation
  future_predictions <- numeric(5)
  
  # Forecast for each future day using the new model
  for (i in 1:5) {
    if (i == 1) {
      current_data <- last_known_data
    } else {
      current_data <- future_data[i - 1, ]
    }
    
    # Compute lagged features
    for (lag in lag_periods) {
      lag_col <- paste0(target_col, "_lag", lag)
      if (i - lag <= 0) {
        value <- price_data[[lag_col]][nrow(price_data)]
      } else {
        value <- future_data[[target_col]][i - lag]
      }
      future_data[[lag_col]][i] <- value
    }
    
    # Compute technical indicators (e.g., SMA)
    for (window in sma_windows) {
      sma_col <- paste0(target_col, "_SMA", window)
      num_historical_needed <- window - (i - 1)
      past_values <- if (i <= window) c(tail(price_data[[target_col]], num_historical_needed), future_data[[target_col]][1:(i - 1)]) else future_data[[target_col]][(i - window):(i - 1)]
      future_data[[sma_col]][i] <- mean(past_values, na.rm = TRUE)
    }
    
    # Prepare the feature vector for prediction
    feature_vector <- future_data[i, feature_cols]
    dfuture <- xgb.DMatrix(data = as.matrix(feature_vector))
    
    # Make the prediction
    prediction <- predict(model, dfuture)
    future_data$target[i] <- prediction
    future_predictions[i] <- prediction
    future_data[[target_col]][i] <- prediction
  }
  
  # Store the predictions from this simulation in the matrix
  simulation_results[, sim] <- future_predictions
}

# Step 5: Calculate the Mean and 95% Confidence Interval
forecast_means <- rowMeans(simulation_results, na.rm = TRUE)
forecast_lower <- apply(simulation_results, 1, quantile, probs = 0.025, na.rm = TRUE)
forecast_upper <- apply(simulation_results, 1, quantile, probs = 0.975, na.rm = TRUE)

# Step 6: Combine Results for Plotting
forecast_confidence <- data.frame(
  datetime = as.POSIXct(future_dates),
  forecast_mean = forecast_means,
  lower_95 = forecast_lower,
  upper_95 = forecast_upper
)

# Plotting the Forecast with Confidence Interval
forecast_confidence <- forecast_confidence %>% rename(predicted = forecast_mean)
forecast_confidence$datetime <- as.POSIXct(forecast_confidence$datetime, format = "%Y-%m-%d")
test_data$datetime <- as.POSIXct(test_data$datetime, format = "%Y-%m-%d %H:%M:%S")

# Combine test data with forecasted confidence intervals
combined_results <- bind_rows(
  test_data %>% select(datetime, target, predicted),
  forecast_confidence %>% select(datetime, predicted) %>% mutate(target = NA)
)

# Plot the forecast with 95% confidence intervals
ggplot() +
  geom_line(data = test_data, aes(x = datetime, y = target, color = "Actual"), linewidth = 0.5) +
  geom_line(data = forecast_confidence, aes(x = datetime, y = predicted, color = "Forecasted Mean"), linewidth = 0.5) +
  geom_ribbon(data = forecast_confidence, aes(x = datetime, ymin = lower_95, ymax = upper_95), fill = "green", alpha = 0.3) +
  labs(title = "Actual vs. Forecasted Values with 95% Confidence Interval",
       x = "Datetime",
       y = "Price",
       color = "Legend") +
  scale_color_manual(values = c("Actual" = "blue", "Forecasted Mean" = "red")) +
  theme_minimal() +
  scale_x_datetime(date_breaks = "90 days", date_labels = "%Y-%m")



# Get the closing price on the last day of historical data
last_closing_price <- tail(test_data$target, 1)

# Get the forecasted mean, lower, and upper values for the first forecasted day
first_forecast_mean <- forecast_means[1]
first_forecast_lower <- forecast_lower[1]
first_forecast_upper <- forecast_upper[1]

# Calculate expected return based on mean forecast
expected_return_mean <- (first_forecast_mean - last_closing_price) / last_closing_price * 100

# Calculate expected return range based on confidence interval
expected_return_lower <- (first_forecast_lower - last_closing_price) / last_closing_price * 100
expected_return_upper <- (first_forecast_upper - last_closing_price) / last_closing_price * 100

# Print the results
cat("Expected Return (Mean):", round(expected_return_mean, 2), "%\n")
cat("Expected Return Range (95% Confidence Interval):", 
    round(expected_return_lower, 2), "% to", round(expected_return_upper, 2), "%\n")





