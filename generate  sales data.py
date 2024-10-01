import pandas as pd
import numpy as np
from faker import Faker
import datetime

# Initialize Faker for generating fake data
faker = Faker()

# Number of records
n_records = 1000

# Generate random sales data
data = {
    'Date': [faker.date_between(start_date=datetime.date(2024, 1, 1), end_date=datetime.date(2024, 12, 31)) for _ in range(n_records)],
    'Product': np.random.choice(['Product A', 'Product B', 'Product C'], n_records),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
    'Sales': np.random.uniform(500, 2000, n_records),
    'Quantity': np.random.randint(1, 20, n_records),
    'Profit': np.random.uniform(100, 500, n_records),
    'Target': np.random.choice([0, 1], n_records)  # Example target column
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('sales_data.csv', index=False)

print("CSV file 'sales_data.csv' has been generated.")

