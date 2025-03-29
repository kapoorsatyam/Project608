import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import OneHotEncoder
import warnings
import boto3

warnings.filterwarnings("ignore")

# Load the dfset
s3_client = boto3.client('s3')
BUCKET_NAME = "kapoors3bucket" 

# Initialize the DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('project608')

# Initialize pagination variables
all_items = []
last_evaluated_key = None

while True:
    if last_evaluated_key:
        response = table.scan(ExclusiveStartKey=last_evaluated_key)
    else:
        response = table.scan()

    # Append fetched items
    all_items.extend(response['Items'])

    # Print progress
    print(f'Fetched {len(all_items)} records...', end="\r")

    # Check if there is more df to fetch
    last_evaluated_key = response.get('LastEvaluatedKey')
    if not last_evaluated_key:
        break  # Exit loop when no more df is found

# Convert to Pandas DataFrame
df = pd.DataFrame(all_items)

# Convert numeric columns
numeric_cols = ['temp', 'precipitation', 'humidity', 'cloud_cover']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Ensure 'time' column is datetime and set it as the index
df['time'] = pd.to_datetime(df['time'])
df['time'] = df['time'] + pd.Timedelta(hours=14)
df.set_index('time', inplace=True)

# Define the forecast steps
forecast_steps = 24

# Prepare a DataFrame to store all the forecasts
all_forecasts = []

# List of unique locations (assuming there is a 'location' column)
locations = df['Location'].unique()
print(locations)

# Function to forecast an exogenous variable using ARIMA
def forecast_exog(series, steps):
    model = ARIMA(series, order=(2, 1, 2))  # Example ARIMA parameters; tune as needed
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(hours=1),
                                   periods=steps, freq='h')
    return pd.Series(forecast, index=forecast_index)

# Loop through each location (city) and train a model
for location in locations:
    # Filter df for the current location
    location_df = df[df['Location'] == location]

    # Select relevant columns
    df = location_df[['temp', 'precipitation', 'humidity', 'cloud_cover', 'season', 'weather_description']].copy()

    # Preprocess categorical variables (e.g., season and weather_description)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(df[['season', 'weather_description']])
    encoded_columns = encoder.get_feature_names_out(['season', 'weather_description'])

    # Combine with the original DataFrame
    df_encoded = pd.concat(
        [df[['temp', 'precipitation', 'humidity', 'cloud_cover']],
         pd.DataFrame(encoded_features, columns=encoded_columns, index=df.index)],
        axis=1
    )

    # Separate the target variable (temp) and exogenous variables
    target = df_encoded['temp']
    exog = df_encoded.drop(columns=['temp'])

    # Forecast each exogenous variable
    exog_forecasts = {}
    for col in exog.columns:
        exog_forecasts[col] = forecast_exog(exog[col], forecast_steps)

    # Combine forecasted exogenous variables into a DataFrame
    future_exog = pd.DataFrame(exog_forecasts)

    # Fit ARIMAX model using historical df
    p, d, q = 2, 1, 3
    model = ARIMA(target, order=(p, d, q), exog=exog)
    model_fit = model.fit()

    # Forecast the target variable (temp) using the future exogenous variables
    temp_forecast = model_fit.forecast(steps=forecast_steps, exog=future_exog)

    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq="h")[1:]

    forecast_df = pd.DataFrame({
        'time': forecast_index,
        'Location': location,
        'forecast_temp': temp_forecast
    })

    # Append the forecast to the results list
    all_forecasts.append(forecast_df)
    print(f"{location} model trained and forecasted.")

# Combine all forecasts into a single DataFrame
final_forecasts_df = pd.concat(all_forecasts)

# Reset index and ensure 'time' is properly set
final_forecasts_df.reset_index(drop=True, inplace=True)

final_forecasts_df['time'] = final_forecasts_df['time'] + pd.Timedelta(days=1)

final_forecasts_df.to_csv("forecast_df.csv", index=False)

# Set your S3 bucket name
FILE_PATH = "forecast_df.csv"  # Change to your actual file path
S3_KEY = "forecast_df.csv"  # This is how the file will be named in S3

try:
    # Upload the file to S3
    s3_client.upload_file(FILE_PATH, BUCKET_NAME, S3_KEY)
    print(f"File '{FILE_PATH}' successfully uploaded to 's3://{BUCKET_NAME}/{S3_KEY}'")
except Exception as e:
    print(f"Error uploading file: {e}")