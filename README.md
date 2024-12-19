**Simple Linear Regression - HR Analytics**
# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt

# Loading the dataset
ctc = pd.read_csv('ctc_data.csv')

# Selecting relevant columns
ctc_new = ctc[['CTCoffered', 'LastCTC']]

# Splitting the data into training and testing sets
x = ctc_new[['LastCTC']]  # Independent variable
y = ctc_new[['CTCoffered']]  # Dependent variable

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# Building a linear regression model
lr = LinearRegression()
model = lr.fit(x_train, y_train)

# Calculating the slope and intercept
slope = model.coef_
intercept = model.intercept_
print(f"Slope (m): {slope}")
print(f"Intercept (c): {intercept}")

# Predicting on the test data
y_test['Predicted CTC'] = model.predict(x_test)

# Display the first few predictions
print(y_test.head())

# Calculating RMSE (Root Mean Squared Error)
y_test['Error'] = y_test['CTCoffered'] - y_test['Predicted CTC']
y_test['Squared_Error'] = y_test['Error'] ** 2
mse = y_test['Squared_Error'].mean()
rmse = sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Example prediction for a new candidate
new_last_ctc = 12  # Example input for the Last CTC
predicted_ctc = model.predict([[new_last_ctc]])
upper_band = predicted_ctc + rmse
lower_band = predicted_ctc - rmse
print(f"Predicted CTC: {predicted_ctc[0][0]:.2f} Lakhs")
print(f"Prediction Range: {lower_band[0][0]:.2f} to {upper_band[0][0]:.2f} Lakhs")
