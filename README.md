# Smart Home Energy Analysis

## Overview
This project analyzes household power consumption data to uncover patterns, trends, and insights for energy optimization. It leverages advanced data science techniques, including time series analysis, machine learning, and statistical modeling, to provide actionable recommendations for reducing energy usage and improving efficiency in smart homes.

## Features
- **Data Preprocessing:** Handles missing values, interpolates gaps, removes outliers, and calculates total consumption.
- **Feature Engineering:** Creates time-based, seasonal, and lag features for improved modeling.
- **Exploratory Data Analysis (EDA):** Generates visualizations for consumption distribution, daily/weekly/monthly patterns, correlations, appliance breakdowns, anomaly detection, and more.
- **Modeling:** Implements XGBoost, ARIMA, and Prophet models for forecasting and evaluation.
- **Appliance Insights:** Calculates daily and hourly contributions of kitchen, laundry, HVAC, and other appliances.
- **Optimization Recommendations:** Suggests ways to reduce energy consumption based on detected patterns.

## Project Structure

```
├── LICENSE
├── main.ipynb                # Jupyter notebook for interactive analysis
├── main.py                   # Main Python script for data processing and modeling
├── withComments.py           # Python script with detailed comments for learning
├── README.md                 # Project documentation
├── output/                   # Directory containing generated plots and visualizations
│   ├── anomaly_detection.png
│   ├── appliance_contribution_percentage.png
│   ├── appliance_correlation.png
│   ├── correlation_matrix.png
│   ├── daily_pattern.png
│   ├── distribution_plots.png
│   ├── energy_distribution_pie.png
│   ├── estimated_savings.png
│   ├── hourly_appliance_consumption.png
│   ├── monthly_pattern.png
│   ├── part_of_day_consumption.png
│   ├── sample_week_consumption.png
│   ├── seasonal_appliance_usage.png
│   ├── seasonal_consumption.png
│   ├── weekday_weekend_consumption.png
│   ├── weekly_pattern.png
│   └── xgboost_feature_importance.png
```

## Data Preprocessing
- Loads household power consumption data (semicolon-separated CSV).
- Handles missing values using linear interpolation and median imputation.
- Removes outliers using z-score filtering.
- Calculates 'Other_consumption' (energy not measured by submeters).

## Feature Engineering
- Extracts time-based features: hour, day, month, year, day_of_week, is_weekend, season, part_of_day.
- Creates lag and rolling window features for autoregressive modeling.

## Exploratory Data Analysis (EDA)
- Distribution plots (histogram, boxplot) of global active power.
- Daily, weekly, and monthly consumption patterns.
- Correlation matrix heatmap of all numerical features.
- Pie chart of energy distribution among appliances.
- Time series plots for sample week and month.
- Anomaly detection using rolling mean and standard deviation.
- Seasonal and part-of-day consumption analysis.
- Weekday vs. weekend consumption comparison.

## Modeling
- **XGBoost Regressor:** Predicts energy consumption using engineered features.
- **ARIMA:** Time series forecasting on hourly aggregated data.
- **Prophet:** Advanced time series forecasting and trend analysis.
- Model evaluation metrics: RMSE, MAE, R².
- Feature importance visualization for XGBoost.

## Appliance Insights
- Calculates daily and hourly averages for kitchen, laundry, HVAC, and other appliances.
- Computes percentage contribution of each appliance to total consumption.
- Identifies highest energy-consuming appliance per day.
- Visualizes appliance contribution over time.

## Optimization Recommendations
- Analyzes peak usage times for each appliance.
- Suggests actionable strategies to reduce energy consumption and costs.

## Output Visualizations
All generated plots are saved in the `output/` directory. Key visualizations include:
- Distribution and boxplots of energy consumption
- Daily, weekly, monthly, and seasonal patterns
- Correlation matrix
- Appliance energy breakdown (pie chart)
- Sample week and month time series
- Anomaly detection plots
- Feature importance (XGBoost)

## Dependencies
The project uses the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- prophet
- xgboost

Install dependencies using pip:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels prophet xgboost
```

## Usage
1. Place the household power consumption dataset in the project directory.
2. Run `main.py` for full analysis and modeling:
   ```powershell
   python main.py
   ```
3. Alternatively, use `main.ipynb` for interactive exploration in Jupyter Notebook.
4. View generated plots in the `output/` folder.

## Customization
- Adjust row limits, feature engineering, or modeling parameters in `main.py` as needed.
- Extend analysis by adding new models or visualizations.

## License
This project is licensed under the terms of the LICENSE file in this repository.

## Contact
For questions, suggestions, or contributions, please contact the repository owner via GitHub.
