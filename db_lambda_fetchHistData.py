import requests
import pandas as pd
import datetime
import awswrangler as wr
from decimal import Decimal
import numpy as np

def lambda_handler(event, context):
    date_today=datetime.date.today()
    date_minus_2 = date_today - datetime.timedelta(days=2)
    date = date_minus_2.strftime('%Y-%m-%d')  # Formats as "YYYY-MM-DD"
    start_date="2024-01-01"
    urls = [
        f'https://archive-api.open-meteo.com/v1/archive?latitude=51.0501&longitude=-114.0853&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=49.2497&longitude=-123.1193&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=43.7001&longitude=-79.4163&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=45.5088&longitude=-73.5878&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=53.5501&longitude=-113.4687&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=45.4112&longitude=-75.6981&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=49.8844&longitude=-97.147&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=46.8123&longitude=-71.2145&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=43.2501&longitude=-79.8496&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration',
        f'https://archive-api.open-meteo.com/v1/archive?latitude=44.6464&longitude=-63.5729&start_date={start_date}&end_date={date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,apparent_temperature,rain,snowfall,snow_depth,surface_pressure,cloud_cover,weather_code,wind_direction_10m,wind_speed_10m,is_day,sunshine_duration'
    ]
    city=['Calgary','Vancouver','Toronto','Montreal','Edmonton','Ottawa','Winnipeg','Quebec','Hamilton','Halifax']
    counter=0
    weather_data = []
    
    for url in urls:
        response = requests.get(url)
        print(f"{city[counter]} Request returned {response.status_code} : '{response.reason}'")
        data = response.json()  # Parse `response.text` into JSON
        hourly_data = data['hourly']
    
    # Loop through the hourly data and append the relevant details
        for i in range(len(hourly_data['time'])):
            details = {
                'Location' : city[counter],
                'time': hourly_data['time'][i],
                'temp': hourly_data['temperature_2m'][i],
                'precipitation': hourly_data['precipitation'][i],
                'humidity': hourly_data['relative_humidity_2m'][i],
                'dew_point': hourly_data['dew_point_2m'][i],
                'apparent_temp': hourly_data['apparent_temperature'][i],
                'rain': hourly_data['rain'][i],
                'snowfall': hourly_data['snowfall'][i],
                'snow_depth': hourly_data['snow_depth'][i],
                'surface_pressure_info': hourly_data['surface_pressure'][i],
                'cloud_cover': hourly_data['cloud_cover'][i],
                'wind_speed': hourly_data['wind_speed_10m'][i],
                'wind_direction': hourly_data['wind_direction_10m'][i],
                'is_day_info': hourly_data['is_day'][i],
                'sunshine_duration_info': hourly_data['sunshine_duration'][i],
                'weather_code':hourly_data['weather_code'][i]
            }
            weather_data.append(details)
        counter+=1
    df = pd.DataFrame(weather_data)    
    df['snow_depth'] = df['snow_depth'].fillna(0)
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    def classify_weather(weather_code):
        if weather_code == 0:
            return 'Sunny'  # Clear sky
        elif weather_code in [1, 2, 3]:
            return 'Cloudy'  # Mainly clear, partly cloudy, and overcast
        elif weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
            return 'Rain'  # Drizzle, Rain showers, etc.
        elif weather_code in [71, 73, 75, 77, 85, 86]:
            return 'Snow'  # Snowfall, Snow showers
        elif weather_code in [45, 48, 66, 67, 95, 96, 99]:
            return 'Windy'  # Fog, Freezing Rain, Thunderstorm
        else:
            return 'Unknown'  # For any unknown codes
    
    # Apply the function to the 'weather_code' column
    df['weather_description'] = df['weather_code'].apply(classify_weather)
    df['time'] = df['time'].astype(str)


    def convert_floats_to_decimal(df):
        for col in df.select_dtypes(include=[np.float64]).columns:
            df[col] = df[col].apply(lambda x: Decimal(str(x)) if not pd.isna(x) else None)
        return df

    print("Before Conversion:", df.dtypes)
    df = convert_floats_to_decimal(df)
    print("After Conversion:", df.dtypes)
    print(df.head())
    print(df['dew_point'].mean())
    
    try:
        wr.config.region = "us-east-1"  # Explicitly set the AWS region for awswrangler
        wr.dynamodb.put_df(df=df, table_name='project608')
        print("Data successfully written to DynamoDB.")
    except Exception as e:
        print(f"Error writing to DynamoDB: {str(e)}")
