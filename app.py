import json
from urllib import request
from flask import Flask, render_template
import requests
import yfinance as yf
from flask import Flask, render_template, request, jsonify

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle

# Create a new Flask app
app = Flask(__name__)

# Replace YOUR_API_KEY with your own Alpha Vantage API key
api_key = "K973YJ8KE4LDXPPS."

# Define the URL for the API request
symbol = "AAPL" # Replace with any stock symbol of your choice
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}"

# Send a request to the Alpha Vantage API and convert the response to a Pandas DataFrame
response = requests.get(url)

data = response.json()["Time Series (Daily)"]
df = pd.DataFrame.from_dict(data, orient="index")

# Convert the DataFrame to the appropriate data types and add a target variable
df = df.astype(float)
df["target"] = df["4. close"].shift(-1)

# Split the data into training and testing sets
X = df.iloc[:-1, :-1]
y = df.iloc[:-1, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit a linear regression model to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print(f"Model score: {score}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model using regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"Root mean squared error: {rmse:.2f}")
print(f"Mean absolute error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")


    # Alpha Vantage API endpoint for retrieving live data for a specific stock
    # url = 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=K973YJ8KE4LDXPPS'

@app.route('/dashboard')
def dashboard():
    # Alpha Vantage API endpoint for retrieving live data for a specific stock
    url = 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=K973YJ8KE4LDXPPS'

    # Send a GET request to the API endpoint
    response = requests.get(url)

    # Retrieve the required data from the API response in JSON format
    data = response.json()['Global Quote']

    # Render the template with the data
    return render_template('dashboard.html', data=data)


# @app.route('/predict')
# def predict():
# #------------------------------------------
# # Make predictions on the testing data
#     y_pred = model.predict(X_test)

#     # Evaluate the model using regression metrics
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
# #------------------------------------------

#     # Return the accuracy to the user
#     return 'mean_squared_error: {}'.format(mse)

@app.route('/predict')
def predict():
    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model using regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create a DataFrame to store the evaluation metrics
    eval_metrics = pd.DataFrame({'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error'],
                                 'Value': [mse, rmse, mae]})

    # Return the evaluation metrics as an HTML table
    return render_template('eval_metrics.html', eval_metrics=eval_metrics.to_html(index=False))

if __name__ == '__main__':
    app.run(debug = True)

