import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os

# Import the predictor class
from heatwave_predictor import HeatwavePredictor, load_data, analyze_heatwaves

# Set page config
st.set_page_config(
    page_title="Antigua Heatwave Early Warning System",
    page_icon="🌡️",
    layout="wide"
)

# Add CSS
st.markdown("""
<style>
    .warning-box {
        background-color: #FFCCCC;
        border-left: 5px solid #FF0000;
        padding: 10px;
        border-radius: 5px;
    }
    .safe-box {
        background-color: #CCFFCC;
        border-left: 5px solid #00CC00;
        padding: 10px;
        border-radius: 5px;
    }
    .alert-box {
        background-color: #FFEDCC;
        border-left: 5px solid #FFA500;
        padding: 10px;
        border-radius: 5px;
    }
    .header-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: #003366;
        color: white;
        text-align: center;
    }
    .metric-box {
        background-color: #EEF0F2;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_predictor():
    """Initialize the heatwave predictor"""
    predictor = HeatwavePredictor(temperature_threshold=31.5, consecutive_days=2)

    # Check if models exist, otherwise train new ones
    if os.path.exists('time_series_model.pkl') and os.path.exists('classification_model.pkl'):
        predictor.load_models()
    else:
        # Load and process data
        data = load_data('Major_Research_Proj_Data.csv')
        processed_data = predictor.preprocess_data(data)

        # Train models
        predictor.train_time_series_model(processed_data)
        predictor.train_classification_model(processed_data)

        # Save models
        predictor.save_models()

    return predictor


def generate_current_status(predictions):
    """Generate current heatwave status based on predictions"""
    # Check if any of the next 3 days have heatwave probability > 0.7
    high_risk_days = predictions.iloc[:3][predictions.iloc[:3]['heatwave_probability'] > 0.7]

    if len(high_risk_days) > 0:
        status = "HIGH RISK"
        status_desc = "High probability of heatwave conditions in the next 3 days"
        status_class = "warning-box"
    elif predictions.iloc[0]['heatwave_probability'] > 0.4:
        status = "MODERATE RISK"
        status_desc = "Moderate heatwave risk detected"
        status_class = "alert-box"
    else:
        status = "LOW RISK"
        status_desc = "No immediate heatwave risk detected"
        status_class = "safe-box"

    return status, status_desc, status_class


def get_recommendations(status):
    """Get heatwave recommendations based on current status"""
    if status == "HIGH RISK":
        return [
            "Stay indoors during peak heat hours (10am-4pm)",
            "Use air conditioning or fans to keep cool",
            "Drink plenty of water, even if not thirsty",
            "Wear lightweight, light-colored, loose-fitting clothing",
            "Check on vulnerable family members and neighbors",
            "Avoid strenuous activities",
            "Be alert for symptoms of heat-related illness"
        ]
    elif status == "MODERATE RISK":
        return [
            "Stay hydrated by drinking plenty of water",
            "Limit outdoor activities during peak heat",
            "Wear appropriate clothing and use sunscreen",
            "Take regular breaks in shaded or air-conditioned areas",
            "Monitor for signs of heat stress"
        ]
    else:
        return [
            "Stay hydrated",
            "Use sunscreen when outdoors",
            "Monitor weather reports for changes"
        ]


def plot_temperature_forecast(historical_data, forecast_data, threshold=31.5):
    """Plot temperature forecast with heatwave threshold"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot historical temperatures
    recent_hist = historical_data.iloc[-30:]['Temperature']
    ax.plot(recent_hist.index, recent_hist, 'b-', label='Historical Temperature')

    # Plot forecasted temperatures
    ax.plot(forecast_data.index, forecast_data['pred_temp'], 'r-', label='Forecasted Temperature')
    ax.fill_between(
        forecast_data.index,
        forecast_data['temp_lower'],
        forecast_data['temp_upper'],
        color='r',
        alpha=0.2
    )

    # Add threshold line
    ax.axhline(y=threshold, color='orange', linestyle='--',
               label=f'Heatwave Threshold ({threshold}°C)')

    ax.set_title('Temperature Forecast')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_heatwave_probability(forecast_data):
    """Plot heatwave probability forecast"""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Define colors based on probability
    colors = ['green' if x < 0.4 else 'orange' if x < 0.7 else 'red' for x in forecast_data['heatwave_probability']]

    # Create bar chart
    bars = ax.bar(
        forecast_data.index,
        forecast_data['heatwave_probability'],
        color=colors
    )

    # Add horizontal line at 0.5 threshold
    ax.axhline(y=0.5, color='black', linestyle='--', label='Warning Threshold')

    # Add labels and formatting
    ax.set_title('Heatwave Probability Forecast')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def create_metrics_section(stats):
    """Create metrics section displaying key heatwave statistics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Total Heatwave Days", f"{stats['heatwave_days']} days")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Heatwave Events", f"{stats['heatwave_events']} events")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Average Duration", f"{stats['avg_duration']:.1f} days")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Heatwave Percentage", f"{stats['heatwave_percentage']:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)


def plot_monthly_distribution(stats):
    """Plot monthly distribution of heatwaves"""
    monthly_data = pd.Series(stats['monthly_distribution'])

    # Add month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_data.index = [month_names[i - 1] for i in monthly_data.index]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(monthly_data.index, monthly_data.values, color='orangered')

    ax.set_title('Monthly Distribution of Heatwave Days')
    ax.set_ylabel('Number of Days')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_yearly_trend(stats):
    """Plot yearly trend of heatwaves"""
    yearly_data = pd.Series(stats['yearly_distribution']).sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly_data.index, yearly_data.values, marker='o', linestyle='-', color='firebrick')

    ax.set_title('Yearly Trend of Heatwave Days')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Days')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown("<div class='header-box'><h1>Antigua Heatwave Early Warning System</h1></div>", unsafe_allow_html=True)

    # Initialize predictor
    try:
        predictor = initialize_predictor()

        # Load data
        data = load_data('Major_Research_Proj_Data.csv')
        processed_data = predictor.preprocess_data(data)

        # Generate predictions for the next 7 days
        recent_data = processed_data.iloc[-14:]  # Use the last 14 days as context
        predictions = predictor.predict_next_days(recent_data, days=7)

        # Analyze heatwave patterns
        heatwave_stats = analyze_heatwaves(processed_data)

        # Generate current status
        status, status_desc, status_class = generate_current_status(predictions)

        # Display current status
        st.markdown(f"<div class='{status_class}'><h2>{status}</h2><p>{status_desc}</p></div>", unsafe_allow_html=True)

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Forecast", "Historical Analysis", "About"])

        with tab1:
            # Create columns for forecast details
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("7-Day Temperature Forecast")
                temp_fig = plot_temperature_forecast(processed_data, predictions)
                st.pyplot(temp_fig)

                st.subheader("Heatwave Probability")
                prob_fig = plot_heatwave_probability(predictions)
                st.pyplot(prob_fig)

            with col2:
                st.subheader("Forecast Details")

                # Display daily forecast
                for date, row in predictions.iterrows():
                    warning_emoji = "🔴" if row['heatwave_warning'] else "🟢"
                    prob_formatted = f"{row['heatwave_probability']:.1%}"
                    temp_formatted = f"{row['pred_temp']:.1f}°C"

                    st.markdown(
                        f"**{date.strftime('%a, %b %d')}**: {temp_formatted} - {prob_formatted} risk {warning_emoji}")

                # Recommendations
                st.subheader("Recommended Actions")
                recommendations = get_recommendations(status)
                for rec in recommendations:
                    st.markdown(f"- {rec}")

                # Add info box
                st.info("Heatwave definition: 2 or more consecutive days with temperature above 31.5°C")

        with tab2:
            st.subheader("Historical Heatwave Analysis")

            # Display key metrics
            create_metrics_section(heatwave_stats)

            # Create columns for charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Monthly Distribution")
                monthly_fig = plot_monthly_distribution(heatwave_stats)
                st.pyplot(monthly_fig)

            with col2:
                st.subheader("Yearly Trend")
                yearly_fig = plot_yearly_trend(heatwave_stats)
                st.pyplot(yearly_fig)

            # Temperature heatmap by year and month
            st.subheader("Temperature Patterns")

            # Resample data to monthly averages
            monthly_temps = processed_data['Temperature'].resample('M').mean()
            monthly_df = monthly_temps.reset_index()
            monthly_df['Year'] = monthly_df['Dates'].dt.year
            monthly_df['Month'] = monthly_df['Dates'].dt.month

            # Pivot data for heatmap
            pivot_df = monthly_df.pivot(index='Year', columns='Month', values='Temperature')
            pivot_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_df, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=.5, ax=ax)
            ax.set_title('Monthly Average Temperatures by Year (°C)')
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.subheader("About this System")
            st.write("""
            This heatwave early warning system provides forecasts and alerts for Antigua based on historical temperature data. 
            The system uses machine learning to predict temperatures and the likelihood of heatwave conditions up to 7 days in advance.
            
            ### How It Works
            
            1. **Data Collection**: The system uses historical temperature data from Antigua (1991-2024).
            2. **Time Series Forecasting**: A Prophet model predicts temperature values for the next 7 days.
            3. **Classification Model**: A machine learning model assesses the probability of heatwave conditions.
            4. **Risk Assessment**: Combines both models to generate a risk status and recommendations.
            
            ### Definition
            
            In Antigua, a heatwave is defined as 2 or more consecutive days with temperatures exceeding 31.5°C.
            
            ### Data Sources
            
            - Historical temperature data from Antigua Meteorological Service
            - Additional regional climate data from Caribbean Climate Outlook Forum (CariCOF)
            """)

            st.subheader("Research Team")
            st.write("""
            - June Douglas
            - Esahtengang Asonganyi
            
            Computer Science Department  
            University of the West Indies
            
            Project Supervisor: ILDEPHONCE, Ilenius
            """)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please make sure the data file 'Major_Research_Proj_Data.csv' is available and correctly formatted.")


if __name__ == "__main__":
    main()
