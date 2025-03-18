# Implementation Guide: Antigua Heatwave Early Warning System

This guide provides instructions for setting up and running the Heatwave Early Warning System developed for Antigua. The system consists of two main components:

1. A predictive model for forecasting temperatures and heatwave probabilities
2. A web dashboard for visualizing predictions and historical patterns

## Prerequisites

Before setting up the system, ensure you have the following installed:

- Python 3.8 or newer
- pip (Python package manager)

## Installation

### Step 1: Clone or download the project files

Ensure you have all the necessary files in your project directory:
- `heatwave_predictor.py` (Main prediction model)
- `app.py` (Streamlit dashboard)
- `Major_Research_Proj_Data.csv` (Temperature dataset)
- `requirements.txt` (Dependencies file)

### Step 2: Create a virtual environment (recommended)

```bash
# Create a virtual environment
python -m venv heatwave_env

# Activate the virtual environment
# On Windows:
heatwave_env\Scripts\activate
# On macOS/Linux:
source heatwave_env/bin/activate
```

### Step 3: Install dependencies

Create a `requirements.txt` file with the following content:

```
pandas==2.0.0
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
statsmodels==0.13.5
prophet==1.1.3
xgboost==1.7.5
streamlit==1.22.0
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the System

### Step 1: Train the prediction models

The first time you run the system, you'll need to train the prediction models. This can be done by running the `heatwave_predictor.py` script:

```bash
python heatwave_predictor.py
```

This will:
1. Load and preprocess the temperature data
2. Train the time series forecasting model (Prophet)
3. Train the classification model for heatwave prediction
4. Save the trained models to disk for future use
5. Generate a sample forecast

### Step 2: Launch the dashboard

Once the models are trained, you can launch the Streamlit dashboard:

```bash
streamlit run app.py
```

This will start the web server and open the dashboard in your default browser. If it doesn't open automatically, you can access it at `http://localhost:8501`.

## System Components

### 1. Heatwave Predictor (`heatwave_predictor.py`)

This is the core module that handles:
- Data preprocessing
- Time series forecasting
- Heatwave classification
- Prediction generation

The main class `HeatwavePredictor` provides methods for loading data, training models, and generating predictions.

#### Key Methods:

- `load_data(filepath)`: Loads temperature data from a CSV file
- `preprocess_data(df)`: Prepares data for modeling
- `train_time_series_model(data)`: Trains a Prophet model for temperature forecasting
- `train_classification_model(data)`: Trains a classification model for heatwave prediction
- `predict_next_days(recent_data, days)`: Generates predictions for the specified number of days
- `visualize_forecast(historical_data, forecast_data)`: Creates a visualization of the forecast

### 2. Dashboard (`app.py`)

The Streamlit dashboard provides a user-friendly interface for:
- Viewing current heatwave status
- Exploring temperature forecasts
- Analyzing historical heatwave patterns
- Getting recommendations based on current risk level

## Customization Options

### Adjusting Heatwave Definition

You can modify the heatwave definition by changing the `temperature_threshold` and `consecutive_days` parameters when initializing the `HeatwavePredictor`:

```python
predictor = HeatwavePredictor(temperature_threshold=31.5, consecutive_days=2)
```

### Adding New Data

To update the system with new temperature data:

1. Append the new data to `Major_Research_Proj_Data.csv` following the same format
2. Retrain the models by running `heatwave_predictor.py` again

### Deploying to a Server

For production deployment:

1. Set up a server with Python and the required dependencies
2. Configure a service to run the Streamlit app (e.g., using systemd)
3. Consider using nginx as a reverse proxy for better security

## Extending the System

### Adding Additional Data Sources

To enhance the system with additional data sources (e.g., humidity):

1. Modify the `load_data()` method to include the new data
2. Update the feature engineering in `preprocess_data()`
3. Retrain the models with the enhanced dataset

### Implementing Alert Notifications

To add automatic alert notifications:

1. Create a new module for handling notifications (e.g., email, SMS)
2. Add a function in `app.py` to trigger alerts when risk exceeds threshold
3. Schedule regular runs of the prediction model to update alerts

## Troubleshooting

### Missing Modules

If you encounter "module not found" errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Model Training Errors

If model training fails:
1. Check the format of your CSV file
2. Ensure there are no missing values or outliers
3. Try with a smaller dataset first to confirm functionality

### Dashboard Display Issues

If the dashboard doesn't display properly:
1. Check browser console for errors
2. Ensure all Python libraries are up to date
3. Try restarting the Streamlit server

## Contact and Support

For assistance with the implementation or to report issues, please contact:

- June Douglas and Esahtengang Asonganyi
- Computer Science Department
- University of the West Indies

## Acknowledgements

This system was developed as part of a research project under the supervision of ILDEPHONCE, Ilenius at the University of the West Indies.
