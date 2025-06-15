import matplotlib.pyplot as plt


def plot_feature_importance(importance_df, top_n=15):
    """Plot feature importance as a horizontal bar chart."""
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:top_n], importance_df['Importance'][:top_n], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance Score')
    plt.show()


def plot_actual_vs_predicted(data, title='Actual vs. Predicted Values Over Time',
                             datetime_col='datetime', actual_col='target', predicted_col='predicted'):
    """Plot actual and predicted values over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(data[datetime_col], data[actual_col], label='Actual', color='blue', linewidth=0.5)
    plt.plot(data[datetime_col], data[predicted_col], label='Predicted', color='red', linewidth=0.5)
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def plot_forecast_with_confidence(forecast_df, title='Bootstrapped Forecast with 95% Confidence Interval',
                                  datetime_col='datetime', mean_col='mean_forecast',
                                  lower_col='lower_bound', upper_col='upper_bound'):
    """Plot forecast mean with confidence interval."""
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df[datetime_col], forecast_df[mean_col], label='Mean Forecast', color='red')
    plt.fill_between(forecast_df[datetime_col], forecast_df[lower_col], forecast_df[upper_col],
                     color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def plot_actual_vs_forecast(combined_results, forecast_confidence,
                            datetime_col='datetime', target_col='target', predicted_col='predicted',
                            title='Actual vs. Forecasted Values with 95% Confidence Interval'):
    """Plot actual data with forecasted mean and confidence interval."""
    plt.figure(figsize=(12, 6))
    plt.plot(combined_results[datetime_col], combined_results[target_col], label='Actual', color='blue', linewidth=0.5)
    plt.plot(forecast_confidence[datetime_col], forecast_confidence[predicted_col],
             label='Forecasted Mean', color='red', linewidth=0.5)
    plt.fill_between(forecast_confidence[datetime_col], forecast_confidence['lower_95'],
                     forecast_confidence['upper_95'], color='green', alpha=0.3,
                     label='95% Confidence Interval')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
