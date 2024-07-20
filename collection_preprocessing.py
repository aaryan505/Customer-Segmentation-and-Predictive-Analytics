import pandas as pd
import sqlite3

# Connect to SQLite database (assuming a database named 'customers.db')
conn = sqlite3.connect('customers.db')

# SQL query to extract customer data
query = '''
SELECT *
FROM customer_data
'''
# Load data into a DataFrame
customer_data = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Data Preprocessing
# Handle missing values
customer_data.dropna(inplace=True)

# Normalize numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
customer_data[['age', 'income']] = scaler.fit_transform(customer_data[['age', 'income']])

# Encode categorical variables
customer_data = pd.get_dummies(customer_data, columns=['gender', 'region'])
