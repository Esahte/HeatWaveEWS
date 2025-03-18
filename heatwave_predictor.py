"""
Antigua Heatwave Early Warning System

This module implements a machine learning-based system for predicting
heatwaves in Antigua using historical temperature data. The system combines
time series forecasting with classification models to predict both future
temperatures and the probability of heatwave conditions.

A heatwave in Antigua is defined as 2 or more consecutive days with
temperatures exceeding 31.5°C.

Models used:
- Prophet: For time series forecasting of temperatures
- Classification models: Logistic Regression, SVM, KNN, XGBoost, Naive Bayes

Author: June Douglas, Esahtengang Asonganyi
University of the West Indies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# For time series modeling
from prophet import Prophet

# For classification models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb


def analyze_heatwaves(data):
    """
    Analyze patterns of heatwaves in the dataset

    Parameters:
    -----------
    data : pandas.DataFrame
        Preprocessed dataframe with heatwave labels

    Returns:
    --------
    dict
        Dictionary with heatwave statistics
    """
    print("Analyzing heatwave patterns...")

    try:
        # Count total heatwave days
        heatwave_days = data['heatwave'].sum()

        # Count total heatwave events
        # A new event starts when the current day is a heatwave day but the previous day wasn't
        data['new_heatwave'] = (data['heatwave'] == 1) & (data['heatwave'].shift(1) != 1)
        heatwave_events = data['new_heatwave'].sum()

        # Calculate average duration of heatwaves
        heatwave_durations = []
        current_duration = 0

        for i in range(len(data)):
            if data['heatwave'].iloc[i] == 1:
                current_duration += 1
            elif current_duration > 0:
                heatwave_durations.append(current_duration)
                current_duration = 0

        # Don't forget to add the last heatwave if it extends to the end of the dataset
        if current_duration > 0:
            heatwave_durations.append(current_duration)

        avg_duration = np.mean(heatwave_durations) if heatwave_durations else 0

        # Calculate monthly distribution of heatwaves
        try:
            # Create a month column to avoid index issues
            temp_data = data[data['heatwave'] == 1].copy()
            temp_data['month'] = temp_data.index.month
            monthly_heatwaves = temp_data.groupby('month').size()
        except Exception as e:
            print(f"Error calculating monthly distribution: {e}")
            # Fallback to empty series
            monthly_heatwaves = pd.Series(dtype='int64')

        # Calculate yearly distribution of heatwaves
        try:
            # Create a year column to avoid index issues
            temp_data = data[data['heatwave'] == 1].copy()
            temp_data['year'] = temp_data.index.year
            yearly_heatwaves = temp_data.groupby('year').size()
        except Exception as e:
            print(f"Error calculating yearly distribution: {e}")
            # Fallback to empty series
            yearly_heatwaves = pd.Series(dtype='int64')

        # Add debug info
        print(f"Data shape: {data.shape}")
        print(f"Number of heatwave days detected: {heatwave_days}")
        print(f"Number of heatwave events detected: {heatwave_events}")
        print(f"Average duration: {avg_duration}")

        # Store results in a dictionary
        results = {
            'total_days': len(data),
            'heatwave_days': heatwave_days,
            'heatwave_events': heatwave_events,
            'heatwave_percentage': (heatwave_days / len(data)) * 100 if len(data) > 0 else 0,
            'avg_duration': avg_duration,
            'max_duration': max(heatwave_durations) if heatwave_durations else 0,
            'monthly_distribution': monthly_heatwaves.to_dict() if not monthly_heatwaves.empty else {},
            'yearly_distribution': yearly_heatwaves.to_dict() if not yearly_heatwaves.empty else {}
        }

        print(f"Analysis complete. Found {heatwave_events} heatwave events spanning {heatwave_days} days.")
        return results

    except Exception as e:
        print(f"Error analyzing heatwaves: {e}")
        # Return basic results on error
        return {
            'total_days': len(data) if isinstance(data, pd.DataFrame) else 0,
            'heatwave_days': 0,
            'heatwave_events': 0,
            'heatwave_percentage': 0,
            'avg_duration': 0,
            'max_duration': 0,
            'monthly_distribution': {},
            'yearly_distribution': {}
        }


def load_data(filepath):
    """
    Load temperature data from CSV file

    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing temperature data

    Returns:
    --------
    pandas.DataFrame
        Processed dataframe with dates and temperatures
    """
    print(f"Loading data from {filepath}...")

    try:
        df = pd.read_csv(filepath)

        # Print column names to ensure we're accessing the right ones
        print(f"Columns in dataset: {df.columns.tolist()}")

        # Check and prepare the data format
        if 'Temperature' in df.columns and 'Dates' in df.columns:
            # Assuming standard format
            df = df[['Dates', 'Temperature']]
        else:
            # Try to determine columns by position
            df = df.iloc[:, [1, 0]]  # Assuming second column is date, first is temperature
            df.columns = ['Dates', 'Temperature']

        # Convert dates to datetime
        df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')

        # Drop rows with NaN dates or temperatures
        df = df.dropna()

        # Sort by date
        df = df.sort_values('Dates')

        # Set date as index
        df = df.set_index('Dates')

        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


class HeatwavePredictor:
    def __init__(self, temperature_threshold=31.5, consecutive_days=2):
        """
        Initialize the Heatwave Predictor model

        Parameters:
        -----------
        temperature_threshold : float
            Temperature threshold in Celsius above which a day is considered unusually hot
        consecutive_days : int
            Number of consecutive days above threshold to classify as a heatwave
        """
        self.temperature_threshold = temperature_threshold
        self.consecutive_days = consecutive_days
        self.time_series_model = None
        self.classification_model = None
        self.scaler = None
        self.feature_names = None
        self.model_scores = {}

    def preprocess_data(self, df):
        """
        Preprocess temperature data for modeling

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with temperature data

        Returns:
        --------
        pandas.DataFrame
            Processed dataframe with additional features
        """
        print("Preprocessing data...")

        try:
            # Create a copy to avoid modifying the original
            data = df.copy()

            # Resample to daily data in case there are multiple readings per day
            data = data.resample('D').mean()

            # Handle missing days by forward filling (limited to 2 days)
            data = data.asfreq('D').fillna(method='ffill', limit=2)

            # If there are still missing values, interpolate
            data = data.interpolate(method='linear')

            # Create lag features (previous days' temperatures)
            for i in range(1, 8):
                data[f'temp_lag_{i}'] = data['Temperature'].shift(i)

            # Create rolling average features
            data['rolling_mean_3d'] = data['Temperature'].rolling(window=3).mean()
            data['rolling_mean_7d'] = data['Temperature'].rolling(window=7).mean()
            data['rolling_std_3d'] = data['Temperature'].rolling(window=3).std()

            # Add month and day of year as cyclical features
            data['month'] = data.index.month
            data['day_of_year'] = data.index.dayofyear

            # Convert month and day of year to cyclical features using sine and cosine transformations
            data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
            data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_year']/365)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_year']/365)

            # Create binary labels for heat wave days
            data['above_threshold'] = (data['Temperature'] >= self.temperature_threshold).astype(int)

            # Create heat wave labels (1 if current day is part of a heat wave)
            data['heatwave'] = 0
            for i in range(len(data) - self.consecutive_days + 1):
                if all(data['above_threshold'].iloc[i:i+self.consecutive_days] == 1):
                    data['heatwave'].iloc[i:i+self.consecutive_days] = 1

            # Drop rows with NaN values created by the lag and rolling features
            data = data.dropna()

            print(f"Data preprocessing complete. Shape after preprocessing: {data.shape}")
            return data

        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return df  # Return original DataFrame on error

    def train_time_series_model(self, data, forecast_days=7):
        """
        Train time series model for temperature forecasting

        Parameters:
        -----------
        data : pandas.DataFrame
            Preprocessed dataframe with temperature data
        forecast_days : int
            Number of days to forecast into the future

        Returns:
        --------
        tuple
            (model, forecast) - Trained time series model and its forecast
        """
        print("Training time series forecasting model...")

        try:
            # Prepare Prophet data format
            prophet_data = data.reset_index()[['Dates', 'Temperature']]
            prophet_data.columns = ['ds', 'y']

            # Initialize and train Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )

            model.fit(prophet_data)

            # Generate forecast
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            # Store the model
            self.time_series_model = model

            print("Time series model training complete.")
            return model, forecast

        except Exception as e:
            print(f"Error training time series model: {e}")
            return None, None

    def train_classification_model(self, data, target_days_ahead=3):
        """
        Train a classification model to predict heatwave occurrences

        Parameters:
        -----------
        data : pandas.DataFrame
            Preprocessed dataframe with heatwave labels
        target_days_ahead : int
            Number of days ahead to predict heatwaves

        Returns:
        --------
        model
            Trained classification model
        """
        print(f"Training classification model to predict heatwaves {target_days_ahead} days ahead...")

        try:
            # Create target variable: will there be a heatwave in the next X days?
            data[f'heatwave_in_{target_days_ahead}d'] = 0

            for i in range(len(data) - target_days_ahead):
                if data['heatwave'].iloc[i:i+target_days_ahead].sum() > 0:
                    data[f'heatwave_in_{target_days_ahead}d'].iloc[i] = 1

            # Prepare features and target
            features = [
                'Temperature', 'rolling_mean_3d', 'rolling_mean_7d',
                'rolling_std_3d', 'month_sin', 'month_cos',
                'day_sin', 'day_cos'
            ]

            # Add lag features
            for i in range(1, 8):
                if f'temp_lag_{i}' in data.columns:
                    features.append(f'temp_lag_{i}')

            X = data[features]
            y = data[f'heatwave_in_{target_days_ahead}d']

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Split data using a simple train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, shuffle=False
            )

            # Initialize models
            models = {
                'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
                'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'Naive Bayes': GaussianNB()
            }

            best_model = None
            best_score = 0
            self.model_scores = {}

            # Train and evaluate each model
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    # Custom score that weights recall higher than precision (detecting heatwaves is more important)
                    custom_score = (0.3 * precision + 0.7 * recall)
                    self.model_scores[name] = custom_score

                    print(f"{name} scores: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Custom={custom_score:.4f}")

                    if custom_score > best_score:
                        best_score = custom_score
                        best_model = model

                except Exception as e:
                    print(f"Error evaluating {name}: {e}")

            # If no model worked, use the simplest model as fallback
            if best_model is None:
                print("All model evaluations failed. Using Logistic Regression as fallback.")
                best_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
                best_model.fit(X_scaled, y)
            else:
                print(f"Best model: {[name for name, model in models.items() if model == best_model][0]} with score: {best_score:.4f}")

            # Store model metadata
            self.classification_model = best_model
            self.feature_names = features

            print("Classification model training complete.")
            return best_model

        except Exception as e:
            print(f"Error training classification model: {e}")
            return None

    def predict_next_days(self, recent_data, days=7):
        """
        Generate temperature and heatwave predictions for the next few days

        Parameters:
        -----------
        recent_data : pandas.DataFrame
            Recent temperature data
        days : int
            Number of days to predict

        Returns:
        --------
        pandas.DataFrame
            DataFrame with predictions
        """
        print(f"Generating predictions for the next {days} days...")

        try:
            if self.time_series_model is None:
                raise ValueError("Time series model has not been trained yet.")

            if self.classification_model is None:
                raise ValueError("Classification model has not been trained yet.")

            # Get the latest date in the data
            last_date = recent_data.index.max()

            # Generate temperature forecasts using Prophet
            future = pd.DataFrame({
                'ds': [last_date + timedelta(days=i) for i in range(1, days+1)]
            })
            forecast = self.time_series_model.predict(future)

            # Extract predicted temperatures
            temp_predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            temp_predictions.columns = ['date', 'pred_temp', 'temp_lower', 'temp_upper']
            temp_predictions = temp_predictions.set_index('date')

            # Prepare data for classification prediction
            latest_data = recent_data.iloc[-1:].copy()

            # Create features for each day in the prediction horizon
            all_predictions = []

            for i in range(days):
                day_data = latest_data.copy()
                prediction_date = last_date + timedelta(days=i+1)

                # Update with forecasted temperature
                day_data['Temperature'] = temp_predictions['pred_temp'].iloc[i]

                # Update date-based features
                day_data['month'] = prediction_date.month
                day_data['day_of_year'] = prediction_date.timetuple().tm_yday
                day_data['month_sin'] = np.sin(2 * np.pi * day_data['month']/12)
                day_data['month_cos'] = np.cos(2 * np.pi * day_data['month']/12)
                day_data['day_sin'] = np.sin(2 * np.pi * day_data['day_of_year']/365)
                day_data['day_cos'] = np.cos(2 * np.pi * day_data['day_of_year']/365)

                # Update lag features if possible
                if i > 0:
                    for j in range(1, min(i+1, 8)):
                        lag_idx = i - j
                        if f'temp_lag_{j}' in self.feature_names:
                            day_data[f'temp_lag_{j}'] = all_predictions[lag_idx]['pred_temp']

                # Extract features for the classification model
                X_features = day_data[self.feature_names].values
                X_scaled = self.scaler.transform(X_features)

                # Predict heatwave probability
                try:
                    heatwave_prob = self.classification_model.predict_proba(X_scaled)[0][1]
                except:
                    # Some models might not support predict_proba, use a binary prediction and convert
                    heatwave_pred = self.classification_model.predict(X_scaled)[0]
                    heatwave_prob = float(heatwave_pred)

                # Create prediction record
                prediction = {
                    'date': prediction_date,
                    'pred_temp': temp_predictions['pred_temp'].iloc[i],
                    'temp_lower': temp_predictions['temp_lower'].iloc[i],
                    'temp_upper': temp_predictions['temp_upper'].iloc[i],
                    'heatwave_probability': heatwave_prob,
                    'heatwave_warning': heatwave_prob > 0.5
                }

                all_predictions.append(prediction)

            # Convert to DataFrame
            predictions_df = pd.DataFrame(all_predictions)
            predictions_df = predictions_df.set_index('date')

            print("Predictions generated successfully.")
            return predictions_df

        except Exception as e:
            print(f"Error generating predictions: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def visualize_forecast(self, historical_data, forecast_data, filename='heatwave_forecast.png'):
        """
        Create and save visualization of temperature forecast and heatwave probabilities

        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical temperature data
        forecast_data : pandas.DataFrame
            Forecast data from predict_next_days method
        filename : str
            Name of the output file

        Returns:
        --------
        None
        """
        print(f"Creating forecast visualization and saving to {filename}...")

        try:
            # Set up the figure
            plt.figure(figsize=(14, 10))

            # Plot historical temperatures
            plt.subplot(2, 1, 1)
            recent_hist = historical_data.iloc[-30:]['Temperature']
            plt.plot(recent_hist.index, recent_hist, 'b-', label='Historical Temperature')

            # Plot forecasted temperatures
            plt.plot(forecast_data.index, forecast_data['pred_temp'], 'r-', label='Forecasted Temperature')
            plt.fill_between(
                forecast_data.index,
                forecast_data['temp_lower'],
                forecast_data['temp_upper'],
                color='r',
                alpha=0.2
            )

            # Add threshold line
            plt.axhline(y=self.temperature_threshold, color='orange', linestyle='--',
                       label=f'Heatwave Threshold ({self.temperature_threshold}°C)')

            plt.title('Temperature Forecast')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot heatwave probabilities
            plt.subplot(2, 1, 2)
            bars = plt.bar(
                forecast_data.index,
                forecast_data['heatwave_probability'],
                color=forecast_data['heatwave_warning'].map({True: 'red', False: 'blue'})
            )

            plt.axhline(y=0.5, color='black', linestyle='--', label='Warning Threshold')
            plt.title('Heatwave Probability Forecast')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Format the figure
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

            print(f"Visualization saved as {filename}")

        except Exception as e:
            print(f"Error creating visualization: {e}")

    def save_models(self, time_series_path='time_series_model.pkl',
                   classification_path='classification_model.pkl'):
        """
        Save trained models to disk

        Parameters:
        -----------
        time_series_path : str
            Path to save the time series model
        classification_path : str
            Path to save the classification model

        Returns:
        --------
        None
        """
        try:
            if self.time_series_model is not None:
                with open(time_series_path, 'wb') as f:
                    pickle.dump(self.time_series_model, f)
                print(f"Time series model saved to {time_series_path}")

            if self.classification_model is not None:
                # Save classification model and metadata
                with open(classification_path, 'wb') as f:
                    pickle.dump({
                        'model': self.classification_model,
                        'scaler': self.scaler,
                        'feature_names': self.feature_names,
                        'model_scores': self.model_scores
                    }, f)
                print(f"Classification model saved to {classification_path}")

        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self, time_series_path='time_series_model.pkl',
                   classification_path='classification_model.pkl'):
        """
        Load trained models from disk

        Parameters:
        -----------
        time_series_path : str
            Path to the saved time series model
        classification_path : str
            Path to the saved classification model

        Returns:
        --------
        None
        """
        try:
            if os.path.exists(time_series_path):
                with open(time_series_path, 'rb') as f:
                    self.time_series_model = pickle.load(f)
                print(f"Time series model loaded from {time_series_path}")

            if os.path.exists(classification_path):
                with open(classification_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.classification_model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.feature_names = model_data['feature_names']
                    self.model_scores = model_data.get('model_scores', {})
                print(f"Classification model loaded from {classification_path}")

        except Exception as e:
            print(f"Error loading models: {e}")


# Example usage
if __name__ == '__main__':
    # Initialize predictor
    predictor = HeatwavePredictor(temperature_threshold=31.5, consecutive_days=2)

    # Load data
    data = load_data('Major_Research_Proj_Data.csv')

    if data.empty:
        print("Failed to load data. Exiting.")
    else:
        # Preprocess data
        processed_data = predictor.preprocess_data(data)

        # Analyze heatwave patterns
        heatwave_stats = analyze_heatwaves(processed_data)
        print("\nHeatwave Statistics:")
        for key, value in heatwave_stats.items():
            if not isinstance(value, dict):
                print(f"- {key}: {value}")

        # Load existing models or train new ones
        predictor.load_models()

        if predictor.time_series_model is None:
            ts_model, forecast = predictor.train_time_series_model(processed_data)

        if predictor.classification_model is None:
            clf_model = predictor.train_classification_model(processed_data)

        # Save models
        predictor.save_models()

        # Generate predictions for the next 7 days
        recent_data = processed_data.iloc[-14:]  # Use the last 14 days as context
        predictions = predictor.predict_next_days(recent_data, days=7)

        # Visualize results
        predictor.visualize_forecast(processed_data, predictions)

        # Print predictions
        print("\nHeatwave Predictions for Next 7 Days:")
        for date, row in predictions.iterrows():
            warning = "⚠️ HEATWAVE WARNING" if row['heatwave_warning'] else "No heatwave expected"
            print(f"{date.date()}: {row['pred_temp']:.1f}°C ({row['heatwave_probability']:.1%} chance) - {warning}")