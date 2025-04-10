import boto3
import pandas as pd

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

    # Check if there is more data to fetch
    last_evaluated_key = response.get('LastEvaluatedKey')
    if not last_evaluated_key:
        break  # Exit loop when no more data is found

# Convert to Pandas DataFrame
df = pd.DataFrame(all_items)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())
