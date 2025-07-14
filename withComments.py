import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from prophet import Prophet
import datetime
import os
from scipy import stats
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style and figure size
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Function to load and preprocess data with row limit parameter
def load_and_preprocess_data(file_path, row_limit=None):
    """
    Load and preprocess the household power consumption data
    
    Parameters:
    file_path (str): Path to the dataset file
    row_limit (int, optional): Maximum number of rows to load from the dataset
    """
    print("Loading and preprocessing data...")
    print(f"Row limit set to: {'All rows' if row_limit is None else row_limit}")
    
    # Define column names for the dataset
    column_names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
                    'Voltage', 'Global_intensity', 'Sub_metering_1',
                    'Sub_metering_2', 'Sub_metering_3']
    
    # Load the dataset with correct column names and data types
    # We use sep=';' as the dataset is semicolon separated
    # Use row_limit parameter to limit the number of rows
    df = pd.read_csv(file_path, sep=';', header=0, names=column_names, 
                     parse_dates={'datetime': ['Date', 'Time']},
                     dayfirst=True, na_values=['?'], nrows=row_limit)
    
    # Convert 'datetime' column to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Check for missing values
    print(f"Missing values before interpolation:\n{df.isnull().sum()}")
    
    # Handle missing values using linear interpolation
    # This is appropriate for time-series data as it preserves the trend
    df = df.interpolate(method='linear')
    
    # Check if there are any remaining missing values
    print(f"Missing values after interpolation:\n{df.isnull().sum()}")
    
    # Calculate total consumption (excluding what's measured by submeters)
    df['Other_consumption'] = (df['Global_active_power'] * 1000 / 60) - (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
    
    # Replace any remaining NaN values with the median of the respective column
    for col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Filter out any outliers using z-score
    for col in df.select_dtypes(include=np.number).columns:
        z_scores = stats.zscore(df[col])
        abs_z_scores = np.abs(z_scores)
        df = df[abs_z_scores < 5]  # Keep only rows with z-score < 5
    
    # Show dataset info
    print(f"\nDataset shape after preprocessing: {df.shape}")
    print("\nDataset columns:", df.columns.tolist())
    print("\nDataset sample:\n", df.head())
    
    return df

# Feature Engineering
def engineer_features(df):
    """
    Create time-based features and lag features for the dataset
    """
    print("\nPerforming feature engineering...")
    
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Create season feature (1: Winter, 2: Spring, 3: Summer, 4: Fall)
    df['season'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else
                                  (2 if x in [3, 4, 5] else
                                   (3 if x in [6, 7, 8] else 4)))
    
    # Create part of day feature
    def get_part_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df['part_of_day'] = df.index.hour.map(get_part_of_day)
    
    # Create lag features for autoregressive modeling
    for lag in [1, 6, 12, 24]:
        df[f'Global_active_power_lag_{lag}h'] = df['Global_active_power'].shift(lag)
    
    # Create rolling window features (averages over different periods)
    for window in [6, 12, 24]:
        df[f'Global_active_power_rolling_{window}h'] = df['Global_active_power'].rolling(window=window).mean()
    
    # Drop NaN values that resulted from creating lag features
    df.dropna(inplace=True)
    
    print("Feature engineering complete.")
    print("Engineered features:", [col for col in df.columns if col not in ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                                                                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
                                                                       'Sub_metering_3', 'Other_consumption']])
    
    return df

# Exploratory Data Analysis
def perform_eda(df):
    """
    Perform exploratory data analysis on the preprocessed data
    """
    print("\nPerforming Exploratory Data Analysis...")
    
    # Create 'output' directory for plots if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # 1. Distribution of energy consumption
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Global_active_power'], kde=True)
    plt.title('Distribution of Global Active Power')
    plt.xlabel('Global Active Power (kilowatts)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Global_active_power'])
    plt.title('Boxplot of Global Active Power')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.tight_layout()
    plt.savefig('output/distribution_plots.png')
    
    # 2. Daily, weekly and monthly patterns
    # Daily patterns
    daily_consumption = df.groupby(df.index.hour)['Global_active_power'].mean()
    plt.figure(figsize=(12, 6))
    daily_consumption.plot(kind='line', marker='o')
    plt.title('Average Power Consumption by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.savefig('output/daily_pattern.png')
    
    # Weekly patterns
    weekly_consumption = df.groupby(df.index.dayofweek)['Global_active_power'].mean()
    plt.figure(figsize=(12, 6))
    weekly_consumption.plot(kind='bar')
    plt.title('Average Power Consumption by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.xticks(range(0, 7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.grid(True, axis='y')
    plt.savefig('output/weekly_pattern.png')
    
    # Monthly patterns
    monthly_consumption = df.groupby(df.index.month)['Global_active_power'].mean()
    plt.figure(figsize=(12, 6))
    monthly_consumption.plot(kind='bar')
    plt.title('Average Power Consumption by Month')
    plt.xlabel('Month')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, axis='y')
    plt.savefig('output/monthly_pattern.png')
    
    # 3. Correlation between features
    plt.figure(figsize=(14, 12))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    # Draw heatmap
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('output/correlation_matrix.png')
    
    # 4. Submeter breakdown - Energy distribution among different submeters
    # Create a pie chart showing distribution of energy consumption
    submeter_data = df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Other_consumption']].mean()
    plt.figure(figsize=(10, 10))
    plt.pie(submeter_data, labels=['Kitchen', 'Laundry Room', 'Water Heater & AC', 'Other'], 
            autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Distribution of Energy Consumption by Appliance Type')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.savefig('output/energy_distribution_pie.png')
    
    # 5. Time series plot for a sample week
    # Sample a week of data
    sample_week = df.iloc[-7*24*60:].resample('H').mean(numeric_only=True)  # Last week, hourly resampled

    
    plt.figure(figsize=(18, 10))
    plt.plot(sample_week.index, sample_week['Global_active_power'], label='Global Active Power')
    plt.plot(sample_week.index, sample_week['Sub_metering_1']/50, label='Kitchen')  # Scaled for visibility
    plt.plot(sample_week.index, sample_week['Sub_metering_2']/50, label='Laundry Room')  # Scaled for visibility
    plt.plot(sample_week.index, sample_week['Sub_metering_3']/50, label='Water Heater & AC')  # Scaled for visibility
    
    plt.title('Energy Consumption Over a Sample Week')
    plt.xlabel('Date')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis to show day of week
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/sample_week_consumption.png')
    
    # 6. Anomaly Detection - Find unusual consumption patterns
    # Compute statistical properties
    rolling_mean = df['Global_active_power'].rolling(window=24).mean()
    rolling_std = df['Global_active_power'].rolling(window=24).std()
    
    # Define anomalies as points beyond 3 standard deviations from the rolling mean
    upper_bound = rolling_mean + 3 * rolling_std
    lower_bound = rolling_mean - 3 * rolling_std
    anomalies = df[(df['Global_active_power'] > upper_bound) | (df['Global_active_power'] < lower_bound)]
    
    # Sample a month of data to visualize anomalies
    sample_month = df.iloc[-30*24*60:].resample('H').mean(numeric_only=True)  # Last month, hourly resampled
    sample_month_anomalies = anomalies.loc[sample_month.index[0]:sample_month.index[-1]].resample('H').mean(numeric_only=True)
    
    plt.figure(figsize=(18, 10))
    plt.plot(sample_month.index, sample_month['Global_active_power'], label='Global Active Power', color='blue', alpha=0.7)
    plt.scatter(sample_month_anomalies.index, sample_month_anomalies['Global_active_power'], 
                color='red', label='Anomalies', s=100, zorder=5)
    
    plt.title('Anomaly Detection in Energy Consumption')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/anomaly_detection.png')
    
    # 7. Energy consumption by season
    seasonal_consumption = df.groupby('season')['Global_active_power'].mean()
    plt.figure(figsize=(12, 6))
    season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}

    seasonal_consumption.index = [season_names[i] for i in seasonal_consumption.index]
    seasonal_consumption.plot(kind='bar', color='skyblue')
    plt.title('Average Power Consumption by Season')
    plt.xlabel('Season')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True, axis='y')
    plt.savefig('output/seasonal_consumption.png')
    
    # 8. Consumption patterns by part of day
    part_day_consumption = df.groupby('part_of_day')['Global_active_power'].mean().reindex(['Morning', 'Afternoon', 'Evening', 'Night'])
    plt.figure(figsize=(12, 6))
    part_day_consumption.plot(kind='bar', color='lightgreen')
    plt.title('Average Power Consumption by Part of Day')
    plt.xlabel('Part of Day')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True, axis='y')
    plt.savefig('output/part_of_day_consumption.png')
    
    # 9. Weekend vs. Weekday Consumption
    weekday_weekend = df.groupby('is_weekend')['Global_active_power'].mean()
    plt.figure(figsize=(10, 6))
    weekday_weekend.index = ['Weekday', 'Weekend']
    weekday_weekend.plot(kind='bar', color=['lightsalmon', 'lightblue'])
    plt.title('Weekday vs. Weekend Power Consumption')
    plt.xlabel('')
    plt.ylabel('Global Active Power (kilowatts)')
    plt.grid(True, axis='y')
    plt.savefig('output/weekday_weekend_consumption.png')
    
    print("EDA completed. Plots saved in the 'output' directory.")
    
    return anomalies

# Feature Scaling
def scale_features(df):
    """
    Scale numerical features using MinMaxScaler
    """
    print("\nScaling features...")
    
    # Select numerical columns for scaling
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Scale numerical features
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("Features scaled successfully.")
    
    return df_scaled, scaler

# Prepare data for modeling
def prepare_for_modeling(df, target_col='Global_active_power', forecast_horizon=24):
    """
    Prepare data for time series modeling
    """
    print("\nPreparing data for modeling...")
    
    # For simplicity, we'll use the target column and its engineered features
    relevant_cols = [col for col in df.columns if (col.startswith(target_col) or 
                                                 col in ['hour', 'day_of_week', 'month', 'season', 'is_weekend'])]
    
    # Create feature set
    X = df[relevant_cols]
    y = df[target_col]
    
    # For time series, split based on time
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, df):
    """
    Train multiple models and evaluate their performance
    """
    print("\nTraining and evaluating models...")
    
    # Model 1: XGBoost Regressor
    print("\n1. Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Train the model
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    xgb_preds = xgb_model.predict(X_test)
    
    # Evaluate XGBoost model
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)
    
    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    print(f"XGBoost MAE: {xgb_mae:.4f}")
    print(f"XGBoost R²: {xgb_r2:.4f}")
    
    # Plot feature importance for XGBoost
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(xgb_model, max_num_features=15, height=0.8)
    plt.title('Feature Importance in XGBoost Model')
    plt.tight_layout()
    plt.savefig('output/xgboost_feature_importance.png')
    
    # Model 2: ARIMA for time series forecasting
    # Use a subset of recent data for ARIMA modeling
    print("\n2. Training ARIMA model...")
    # Resample to hourly data for ARIMA modeling (to reduce computational complexity)
    hourly_data = df['Global_active_power'].resample('H').mean()
    
    # Split into train and test
    train_size = int(len(hourly_data) * 0.8)
    train_arima = hourly_data[:train_size]
    test_arima = hourly_data[train_size:]
    
    # Fit ARIMA model
    try:
        arima_model = ARIMA(train_arima, order=(2,1,2))
        arima_fit = arima_model.fit()
        
        # Make forecast
        forecast_steps = len(test_arima)
        arima_forecast = arima_fit.forecast(steps=forecast_steps)
        
        # Evaluate ARIMA model
        arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_forecast))
        arima_mae = mean_absolute_error(test_arima, arima_forecast)
        
        print(f"ARIMA RMSE: {arima_rmse:.4f}")
        print(f"ARIMA MAE: {arima_mae:.4f}")
        
        # Plot ARIMA forecast
        plt.figure(figsize=(14, 7))
        plt.plot(train_arima.index[-100:], train_arima.values[-100:], label='Historical Data')
        plt.plot(test_arima.index, test_arima.values, label='Actual Values')
        plt.plot(test_arima.index, arima_forecast, label='ARIMA Forecast', color='red')
        plt.title('ARIMA Time Series Forecast')
        plt.xlabel('Date')
        plt.ylabel('Global Active Power (kilowatts)')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/arima_forecast.png')
    except:
        print("ARIMA model fitting failed. This could be due to non-stationarity or other time series issues.")
    
    # Model 3: Prophet model for time series forecasting
    print("\n3. Training Facebook Prophet model...")
    try:
        # Prepare data for Prophet (requires specific format)
        prophet_data = hourly_data.reset_index()
        prophet_data.columns = ['ds', 'y']
        
        # Split into train and test
        prophet_train = prophet_data.iloc[:train_size]
        prophet_test = prophet_data.iloc[train_size:]
        
        # Initialize and fit Prophet model
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        prophet_model.fit(prophet_train)
        
        # Create future dataframe for prediction
        future = prophet_model.make_future_dataframe(periods=len(prophet_test), freq='H')
        forecast = prophet_model.predict(future)
        
        # Extract predictions for test period
        prophet_preds = forecast.iloc[-len(prophet_test):]['yhat'].values
        
        # Evaluate Prophet model
        prophet_rmse = np.sqrt(mean_squared_error(prophet_test['y'], prophet_preds))
        prophet_mae = mean_absolute_error(prophet_test['y'], prophet_preds)
        
        print(f"Prophet RMSE: {prophet_rmse:.4f}")
        print(f"Prophet MAE: {prophet_mae:.4f}")
        
        # Plot Prophet forecast components
        prophet_fig = prophet_model.plot_components(forecast)
        plt.savefig('output/prophet_components.png')
        
        # Plot Prophet forecast
        plt.figure(figsize=(14, 7))
        plt.plot(prophet_train['ds'].iloc[-100:], prophet_train['y'].iloc[-100:], label='Historical Data')
        plt.plot(prophet_test['ds'], prophet_test['y'], label='Actual Values')
        plt.plot(prophet_test['ds'], prophet_preds, label='Prophet Forecast', color='green')
        plt.title('Prophet Time Series Forecast')
        plt.xlabel('Date')
        plt.ylabel('Global Active Power (kilowatts)')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/prophet_forecast.png')
    except:
        print("Prophet model fitting failed.")
    
    # Compare model performances
    print("\nModel Performance Comparison:")
    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    try:
        print(f"ARIMA RMSE: {arima_rmse:.4f}")
        print(f"Prophet RMSE: {prophet_rmse:.4f}")
    except:
        pass
    
    # Return the best model (XGBoost for this example)
    return xgb_model, xgb_preds, y_test

# Generate appliance-specific insights
def generate_appliance_insights(df):
    """
    Generate insights about energy consumption patterns of specific appliances
    """
    print("\nGenerating appliance-specific insights...")
    
    # Calculate daily averages for each submeter
    daily_submeter = df.resample('D').mean(numeric_only=True)

    
    # Create a dataframe for appliance insights
    appliance_df = pd.DataFrame({
        'Kitchen': daily_submeter['Sub_metering_1'],
        'Laundry': daily_submeter['Sub_metering_2'],
        'HVAC': daily_submeter['Sub_metering_3'],
        'Other': daily_submeter['Other_consumption']
    })
    
    # Calculate percentage contribution of each appliance
    total_consumption = appliance_df.sum(axis=1)
    for col in appliance_df.columns:
        appliance_df[f'{col}_pct'] = (appliance_df[col] / total_consumption) * 100
    
    # Identify the most energy-consuming appliance for each day
    appliance_df['Highest_Consumer'] = appliance_df[['Kitchen', 'Laundry', 'HVAC', 'Other']].idxmax(axis=1)
    
    # Plot time series of percentage contribution
    plt.figure(figsize=(16, 8))
    for col in ['Kitchen_pct', 'Laundry_pct', 'HVAC_pct', 'Other_pct']:
        plt.plot(appliance_df.index, appliance_df[col], label=col.split('_')[0])
    
    plt.title('Daily Percentage Contribution of Each Appliance to Total Consumption')
    plt.xlabel('Date')
    plt.ylabel('Percentage of Total Consumption (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/appliance_contribution_percentage.png')
    
    # Analyze hourly patterns for each appliance
    hourly_submeter = df.groupby(df.index.hour).mean(numeric_only=True)
    
    plt.figure(figsize=(16, 8))
    plt.plot(hourly_submeter.index, hourly_submeter['Sub_metering_1'], label='Kitchen')
    plt.plot(hourly_submeter.index, hourly_submeter['Sub_metering_2'], label='Laundry')
    plt.plot(hourly_submeter.index, hourly_submeter['Sub_metering_3'], label='HVAC')
    plt.plot(hourly_submeter.index, hourly_submeter['Other_consumption'], label='Other')
    
    plt.title('Average Hourly Consumption by Appliance Type')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.savefig('output/hourly_appliance_consumption.png')
    
    # Identify peak usage times for each appliance
    kitchen_peak = hourly_submeter['Sub_metering_1'].idxmax()
    laundry_peak = hourly_submeter['Sub_metering_2'].idxmax()
    hvac_peak = hourly_submeter['Sub_metering_3'].idxmax()
    other_peak = hourly_submeter['Other_consumption'].idxmax()
    
    print(f"Peak usage times by appliance:")
    print(f"Kitchen: {kitchen_peak}:00")
    print(f"Laundry: {laundry_peak}:00")
    print(f"HVAC: {hvac_peak}:00")
    print(f"Other appliances: {other_peak}:00")
    
    # Calculate correlation between appliance usage
    appliance_correlation = df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Other_consumption']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(appliance_correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Between Appliance Usage')
    plt.tight_layout()
    plt.savefig('output/appliance_correlation.png')
    
    # Check for seasonal variations in appliance usage
    season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    seasonal_appliance = df.groupby('season')[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Other_consumption']].mean()
    # Map only the seasons present in the data
    season_labels = [season_names.get(s, str(s)) for s in seasonal_appliance.index]
    seasonal_appliance.index = season_labels
    
    plt.figure(figsize=(14, 10))
    seasonal_appliance.plot(kind='bar')
    plt.title('Seasonal Variation in Appliance Energy Consumption')
    plt.xlabel('Season')
    plt.ylabel('Average Power Consumption')
    plt.legend(['Kitchen', 'Laundry', 'HVAC', 'Other'])
    plt.grid(True, axis='y')
    plt.savefig('output/seasonal_appliance_usage.png')
    
    # Calculate energy efficiency opportunities
    # Identify potential savings for each appliance type
    kitchen_potential_savings = hourly_submeter['Sub_metering_1'].max() - hourly_submeter['Sub_metering_1'].min()
    laundry_potential_savings = hourly_submeter['Sub_metering_2'].max() - hourly_submeter['Sub_metering_2'].min()
    hvac_potential_savings = hourly_submeter['Sub_metering_3'].max() - hourly_submeter['Sub_metering_3'].min()
    other_potential_savings = hourly_submeter['Other_consumption'].max() - hourly_submeter['Other_consumption'].min()
    
    print(f"\nPotential energy savings by optimizing usage times:")
    print(f"Kitchen appliances: {kitchen_potential_savings:.2f} units per hour")
    print(f"Laundry appliances: {laundry_potential_savings:.2f} units per hour")
    print(f"HVAC system: {hvac_potential_savings:.2f} units per hour")
    print(f"Other appliances: {other_potential_savings:.2f} units per hour")
    
    return kitchen_peak, laundry_peak, hvac_peak, other_peak

# Generate optimization recommendations
def generate_recommendations(df, kitchen_peak, laundry_peak, hvac_peak, other_peak):
    """
    Generate recommendations for energy optimization
    """
    print("\nGenerating energy optimization recommendations...")
    
    laundry_savings = 0
    kitchen_savings = 0
    vampire_savings = 0
    # Off-peak hours (typically early morning and late night)
    off_peak_hours = list(range(0, 7)) + list(range(22, 24))
    
    # Calculate average tariff difference between peak and off-peak hours
    # This is an assumption - in a real scenario, you'd use actual tariff data
    assumed_peak_tariff = 0.20  # $ per kWh during peak hours
    assumed_off_peak_tariff = 0.10  # $ per kWh during off-peak hours
    tariff_difference = assumed_peak_tariff - assumed_off_peak_tariff
    
    # Calculate potential savings by shifting loads
    # 1. Laundry optimization
    laundry_consumption = df['Sub_metering_2'].mean()
    if laundry_peak not in off_peak_hours:
        laundry_savings = laundry_consumption * tariff_difference * 30  # monthly savings
        print(f"1. Shifting laundry activities from {laundry_peak}:00 to off-peak hours (22:00-7:00)")
        print(f"   could save approximately ${laundry_savings:.2f} per month.")
    else:
        print("1. Laundry activities are already optimized for off-peak usage.")
    
     # 2. HVAC optimization
    winter_hvac = df[df['season'] == 1]['Sub_metering_3'].mean()
    summer_hvac = df[df['season'] == 3]['Sub_metering_3'].mean()
    seasonal_difference = abs(winter_hvac - summer_hvac)

    # 3. Kitchen usage optimization
    if kitchen_peak in range(17, 21):  # If peak is during evening peak hours
        kitchen_savings = df['Sub_metering_1'].mean() * 0.15 * 30  # Assuming 15% reduction
        print(f"3. Kitchen appliances are heavily used at {kitchen_peak}:00 (peak tariff time).")
        print("   Consider meal prepping during weekends or using energy-efficient cooking methods.")
        print(f"   Potential monthly savings: ${kitchen_savings:.2f}")

    # 4. Vampire power detection
    night_consumption = df[(df.index.hour >= 1) & (df.index.hour <= 4)]['Other_consumption'].mean()
    if night_consumption > 1:  # If standby power consumption is high
        vampire_savings = night_consumption * 0.7 * assumed_off_peak_tariff * 30  # Assuming 70% reduction
        print(f"4. Standby power consumption is {night_consumption:.2f} units during 1:00-4:00 AM.")
        print("   Consider using smart power strips to cut power to devices when not in use.")
        print(f"   Potential monthly savings: ${vampire_savings:.2f}")
    
    # 5. Behavioral recommendations based on weekday/weekend patterns
    weekday_consumption = df[df['is_weekend'] == 0]['Global_active_power'].mean()
    weekend_consumption = df[df['is_weekend'] == 1]['Global_active_power'].mean()
    
    if weekend_consumption > weekday_consumption * 1.2:  # If weekend usage is 20% higher
        print("5. Weekend energy consumption is significantly higher than weekdays.")
        print("   Consider reviewing weekend activities for energy optimization.")
        print("   Potential strategies: Batch cooking, combined laundry loads, outdoor activities.")
    
    # 6. Anomaly-based recommendations
    # Calculate z-scores to identify anomalous consumption
    df['zscore'] = stats.zscore(df['Global_active_power'])
    anomalies = df[df['zscore'] > 3]  # More than 3 standard deviations
    
    if len(anomalies) > 0:
        anomaly_percentage = (len(anomalies) / len(df)) * 100
        print(f"6. Detected {len(anomalies)} anomalous consumption patterns ({anomaly_percentage:.2f}% of time periods).")
        print("   These could indicate inefficient appliance usage or malfunctions.")
        print("   Recommendation: Investigate appliances during these specific time periods.")
    
    # 7. General energy efficiency tips
    print("\nGeneral Energy Efficiency Recommendations:")
    print(" - Replace incandescent bulbs with LED lighting (up to 75% energy savings for lighting)")
    print(" - Ensure proper sealing on refrigerator and freezer doors")
    print(" - Unplug chargers and appliances when not in use")
    print(" - Consider smart thermostats for optimized HVAC control")
    print(" - Regularly clean or replace HVAC filters (can improve efficiency by 5-15%)")
    
    # Plot potential savings chart
    try:
        savings_categories = ['Laundry Timing', 'HVAC Optimization', 'Kitchen Usage', 'Standby Power']
        estimated_savings = [
            laundry_savings,
            seasonal_difference * assumed_peak_tariff * 30 * 0.1,
            kitchen_savings,
            vampire_savings
        ]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(savings_categories, estimated_savings, color=['skyblue', 'lightgreen', 'salmon', 'purple'])

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${height:.2f}', ha='center', va='bottom')

        plt.title('Estimated Monthly Savings by Optimization Category')
        plt.ylabel('Estimated Savings ($)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('output/estimated_savings.png')
        
    except Exception as e:
        print(f"Couldn't generate savings chart due to missing data: {e}")

# Main function with row limit parameter
def main(row_limit=None):
    """
    Main function to orchestrate the entire workflow
    
    Parameters:
    row_limit (int, optional): Maximum number of rows to load from the dataset
    """
    print("Energy Consumption Analysis and Prediction Pipeline")
    print("=" * 50)
    
    # Check if the dataset file exists in the current directory
    file_path = 'household_power_consumption.csv'
    if not os.path.exists(file_path):
        print(f"Error: Dataset file '{file_path}' not found.")
        print("Please make sure the 'Individual Household Electric Power Consumption' dataset is in the current directory.")
        return
    
    # Load and preprocess data with row limit
    df = load_and_preprocess_data(file_path, row_limit=row_limit)
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Perform Exploratory Data Analysis
    anomalies = perform_eda(df_engineered)
    
    # Scale features
    df_scaled, scaler = scale_features(df_engineered)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_for_modeling(df_engineered)
    
    # Train and evaluate models
    best_model, predictions, actuals = train_and_evaluate_models(X_train, X_test, y_train, y_test, df_engineered)
    
    # Generate appliance-specific insights
    kitchen_peak, laundry_peak, hvac_peak, other_peak = generate_appliance_insights(df_engineered)
    
    # Generate optimization recommendations
    generate_recommendations(df_engineered, kitchen_peak, laundry_peak, hvac_peak, other_peak)
    
    print("\nAnalysis completed successfully. Results and visualizations are available in the 'output' directory.")

if __name__ == "__main__":
    # You can specify the number of rows to process here
    # For example, use 100000 rows (about 5% of the dataset) for testing
    # Set to None to process the entire dataset
    main(row_limit=500000)  # Change this value as needed