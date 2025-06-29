from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


app = Flask(__name__)
ridge_model = joblib.load('lgb_model.pkl')  
scaler = joblib.load('scaler.pkl') 
poly_transform = joblib.load('poly_transform.pkl')
temp_model = joblib.load('temperature_model.pkl')
wind_model = joblib.load('wind_speed_model.pkl')
hum_model = joblib.load('humidity_model.pkl')
rain_model = joblib.load('rain_shower_model.pkl')
scaler_wea = joblib.load('scaler_wea.pkl') 
holiday_data = pd.read_csv('indian_holidays_and_weekends_2020_2044.csv')
holiday_data['Date'] = pd.to_datetime(holiday_data['Date'], format='%Y-%m-%d')

def is_holiday(date):
    holiday_row = holiday_data[holiday_data['Date'] == date]
    if not holiday_row.empty and holiday_row['is_holiday'].values[0] == 1:
        return True, holiday_row.iloc[0]['holiday_name']
    return False, "Working Day"

def apply_seasonal_adjustments(month, demand):
    if month in [5, 6]:  
        demand *= 1.006
        season = "Summer"
    elif month in [7, 8]:
        demand *= 1.001
        season = "Monsoon"
    elif month in [9, 10]:
        demand *= 1.00
        season = "Post-Monsoon"
    elif month in [11, 12, 1]:
        demand *= 1.002
        season = "Winter"
    elif month in [2, 3, 4]:
        demand *= 1.003
        season = "Pre-Summer"
    return demand, season

def apply_load_growth(date, demand):
    load_growth_rate = 0.03
    load_decrease_rate = -0.03
    growth_start_date = datetime(2024, 1, 1)
    decrease_date = datetime(2020, 1, 1)

    if date > growth_start_date:
        years_passed = (date - growth_start_date).days / 365.25
        growth_factor = (1 + load_growth_rate) ** years_passed
        demand *= growth_factor
    elif date < decrease_date:
        years_passed = (decrease_date - date).days / 365.25
        decrease_factor = (1 + load_decrease_rate) ** years_passed
        demand *= decrease_factor

    return demand

def predict_weather_factors(X_weather_scaled):
    weather_df = pd.DataFrame({
        'Temperature': temp_model.predict(X_weather_scaled),
        'Wind Speed': wind_model.predict(X_weather_scaled),
        'Humidity': hum_model.predict(X_weather_scaled),
        'rain/shower': rain_model.predict(X_weather_scaled)
    })
    return weather_df

def predict_demand(weather_df):
    weather_features = ['Hour', 'Day','Weekday' , 'Month', 'Year' , 'Temperature', 'Humidity', 'Wind Speed', 'rain/shower']
    X_demand = weather_df[weather_features]
    X_demand_poly = poly_transform.transform(X_demand)
    weather_df['Demand'] = ridge_model.predict(X_demand_poly)
    return weather_df

def apply_event_adjustment(demand, events, date_str, hour):
    matching_events = events[
    (events['Date'] == date_str) &
    (events['Start_Hour'] <= hour) &
    (events['End_Hour'] >= hour)
]
    intensity_multiplier = {'low': 1.02, 'medium': 1.06, 'high': 1.09}
    for _, event in matching_events.iterrows():
        demand *= intensity_multiplier.get(event['Intensity'], 1.0)
    return demand,matching_events['Event_Name'].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_events', methods=['GET'])
def load_events():
    if os.path.exists('event.xlsx'):
        events = pd.read_excel('event.xlsx')
        events = events.fillna('')
        return jsonify(events.to_dict(orient='records'))
    return jsonify([])

@app.route('/save_events', methods=['POST'])
def save_events():
    try:
        events = request.get_json()
        columns = ["Date", "Start_Hour", "End_Hour", "Intensity", "Event_Name"]
        if not events:
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.DataFrame(events, columns=columns)

        df.to_excel("event.xlsx", index=False) 
        return jsonify({"message": "Events saved successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    try:
        input_date = pd.to_datetime(data['date'], format='%Y-%m-%d')
        input_hour = int(data['hour'])
        is_holiday_flag, holiday_name = is_holiday(input_date)
        events = pd.read_excel('event.xlsx')
        if not events.empty:
            events['Date'] = pd.to_datetime(events['Date'], format='%Y-%m-%d')

        demand_24_hours = []
        adjusted_demand_24_hours = []
        temperature_24_hours = []
        humidity_24_hours = []
        wind_speed_24_hours = []
        rain_shower_24_hours = []

        for hour in range(24):
            weather_features = {
                'day': [input_date.day],
                'hour': [hour],
                'month': [input_date.month],
                'Year': [input_date.year]
            }
            
            weather_df = pd.DataFrame(weather_features)

            X_weather_scaled = scaler_wea.transform(weather_df[['day', 'hour', 'month', 'Year']])

            weather_factors = predict_weather_factors(X_weather_scaled)

            demand_features = {
                'Hour': [hour],
                'Day': [input_date.day],
                'Weekday': [input_date.weekday()],
                'Month': [input_date.month],
                'Year': [input_date.year],
                'Temperature': weather_factors['Temperature'].values,
                'Humidity': weather_factors['Humidity'].values,
                'Wind Speed': weather_factors['Wind Speed'].values,
                'rain/shower': weather_factors['rain/shower'].values
            }

            demand_df = pd.DataFrame(demand_features)
            demand_df = predict_demand(demand_df)
            predicted_demand = demand_df['Demand'].mean()
            adjusted_demand = apply_load_growth(input_date, predicted_demand) 
            adjusted_demand1, season = apply_seasonal_adjustments(input_date.month, adjusted_demand)
            #adjusted_demand1*=1.002
            if is_holiday_flag and 8 <= input_hour <= 18:
                if input_hour>=10 :
                    adjusted_demand1*=1.003
                elif 16<=input_hour<=18:
                    adjusted_demand1*=1.002
            adjusted_demand2, event_names = apply_event_adjustment(adjusted_demand1, events, input_date , hour)
            demand_24_hours.append(predicted_demand)
            adjusted_demand_24_hours.append(adjusted_demand2)
            temperature_24_hours.append(weather_factors['Temperature'].values[0])
            humidity_24_hours.append(weather_factors['Humidity'].values[0])
            wind_speed_24_hours.append(weather_factors['Wind Speed'].values[0])
            rain_shower_24_hours.append(weather_factors['rain/shower'].values[0])

        peak_demand = max(adjusted_demand_24_hours)
        peak_hour = adjusted_demand_24_hours.index(peak_demand)
        predicted_hour_demand = demand_24_hours[input_hour]
        adjusted_hour_demand = adjusted_demand_24_hours[input_hour]
        response = {
            'predicted_demand': predicted_hour_demand, 
            'adjusted_demand': adjusted_hour_demand,  
            'is_active_hour': 8 <= input_hour <= 18,
            'season': season,
            'peak_demand': peak_demand,
            'peak_hour': peak_hour,
            'temperature': temperature_24_hours[input_hour],
            'humidity': humidity_24_hours[input_hour],
            'wind_speed': wind_speed_24_hours[input_hour],
            'rain': rain_shower_24_hours[input_hour],
            'holiday': is_holiday_flag,
            'holiday_name': holiday_name,
            'prediction_date': data['date'],
            'prediction_hour': input_hour,
            'demand_24_hours': adjusted_demand_24_hours,
            'adjusted_demand_24_hours': adjusted_demand_24_hours,  
            'Temperature_24_hours': temperature_24_hours,
            'Humidity_24_hours': humidity_24_hours,
            'Wind_Speed_24_hours': wind_speed_24_hours,
            'rain_shower_24_hours': rain_shower_24_hours,
            'event_names': ', '.join(event_names) if event_names else 'None'
        }

    except Exception as e:
        response = {
            'error': str(e)
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)