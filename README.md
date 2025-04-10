# Data608 Project

This repository contains several scripts designed to fetch, process, and store weather data from an API, as well as interact with AWS services such as S3, DynamoDB, and Lambda. These scripts are structured to automate the data pipeline and facilitate easy access to historical and daily weather data.

### Note:
- **Primary Data Source**: All data is primarily stored in **DynamoDB** for efficient and scalable retrieval.

## Scripts Overview

### 1. **`db_lambda_fetchHisData.py`**
- **Purpose**: This Lambda function fetches weather data from the API and stores it in DynamoDB.
- **How it works**: The script preprocesses, cleans, and transforms the data before storing it in a **DynamoDB table** for further use.

### 2. **`db_lambda_automate.py`**
- **Purpose**: This Lambda function automates the fetching of new weather data from the API via an EventBridge scheduler.
- **How it works**: It regularly fetches new data, performs transformations and cleaning, and appends the updated data to the existing records in **DynamoDB**.

### 3. **`ec2_dbConnect.py`**
- **Purpose**: This Python script is designed to fetch paginated data from DynamoDB to test connectivity between an EC2 instance and the database.
- **How it works**: The script checks whether data retrieval from DynamoDB works correctly on the EC2 instance, which will eventually connect to a Streamlit dashboard for data visualizations.

### 4. **`model_db.py`**
- **Purpose**: This script uses an ARIMA model to forecast temperature for the next 24 hours using historical weather data.
- **How it works**: It fetches data from **DynamoDB**, preprocesses it, encodes categorical features, and trains an ARIMAX model for each location. Forecast results are saved as a CSV and uploaded to **S3**.

### 5. **`streamlit_db.py`**
- **Purpose**: This Streamlit app displays an interactive dashboard for visualizing weather data and temperature forecasts.
- **How it works**: It fetches historical data from **DynamoDB** and forecasted data from **S3**, processes them, and displays insights such as trends, weather distributions, correlations, and 24-hour temperature forecasts using **Plotly** visualizations.

## Requirements
- Python_version >= "3.9"
- AWS SDK for Python (`boto3`)
- Streamlit (for interactive dashboard)
- Jupyter Notebook (for data exploration)
- Required libraries:
  - `pandas`
  - `numpy`
  - `requests`
  - `boto3`
  - `statsmodels`
  - `scikit-learn`
  - `plotly`
  - `scipy`
