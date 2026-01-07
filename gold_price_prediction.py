# ================================================================#
# Gold Price Prediction using Machine Learning
# ================================================================#

# Work Flow:
# 1. Import Libraries
# 2. Load Dataset
# 3. Data Preprocessing
# 4. Data Analysis
# 5. Data Splitting
# 6. Model Training
# 7. Model Evaluation

# DataSet Structure:
# Date - Date of the record
# SPX - S&P 500 index value in that date in USD
# GLD - Gold price in USD
# USO - Crude Oil price in USD
# SLV - Silver price in USD
# EUR/USD - Euro to USD exchange rate
# ================================================================#
# 1. Import Libraries
# ================================================================#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ================================================================#
# 2. Load Dataset
# ================================================================#

data_df = pd.read_csv("gld_price_data.csv")

# Print the first few rows of the dataset
print(data_df.head())


# ================================================================#
# 3. Data Preprocessing
# ================================================================#

# Covert 'Date' column to date format
data_df["Date"] = pd.to_datetime(data_df["Date"])
# Print the shape of the dataset
print(data_df.shape)

# Print the column information
print(data_df.info())

# Print the summary statistics of the dataset
print(data_df.describe())

# Checking missing values
print(data_df.isnull().sum())

# ===============================================================#
# 4. Data Analysis
# ================================================================#

# Correlation Matrix
correlation = data_df.corr()

print(correlation)

# Visualizing the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="YlGnBu", cbar=True, square=True)
plt.title("Correlation Matrix")
plt.show()

# Correlation with GLD
print(correlation["GLD"].sort_values(ascending=False))

# Distribution of GLD prices
plt.figure(figsize=(8, 6))
sns.distplot(data_df["GLD"], color="blue")
plt.title("Distribution of Gold Prices (GLD)")
plt.xlabel("Gold Price (GLD)")
plt.ylabel("Density")
plt.show()

# ================================================================#
# 5. Data Splitting
# ================================================================#

X = data_df.drop(["Date", "GLD"], axis=1)
y = data_df["GLD"]

print(X)
print(y)


# Splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)

# ================================================================#
# 6. Model Training
# ================================================================#

model = RandomForestRegressor(n_estimators=100)

# Training the model

model.fit(X_train, y_train)

# ================================================================#
# 7. Model Evaluation
# ================================================================#

# Making predictions
test_prediction = model.predict(X_test)

print(test_prediction)

# R Squared Error

r2_score = r2_score(y_test, test_prediction)
print("R Squared Error:", r2_score)

# Conclusions:
# R2_Score: 0.9894240233298174
# This indicates that the model explains approximately 98.94%
# of the variance in the gold prices based on the input features.

# Mean Absolute Error
mae = mean_absolute_error(y_test, test_prediction)
print("Mean Absolute Error:", mae)

# Conclusions:
# MAE: 1.3265309298253254
# This indicates that, on average, the model's predictions
# are off by approximately $1.33 from the actual gold prices.

# Mean Squared Error
mse = mean_squared_error(y_test, test_prediction)
print("Mean Squared Error:", mse)

# Conclusions:
# MSE: 5.578230157751912
# This indicates that the average squared difference between
# the predicted and actual gold prices is approximately 5.58.


# Visualizing the actual vs predicted prices

y_test = list(y_test)
plt.figure(figsize=(8, 6))
plt.plot(y_test, color="blue", label="Actual Gold Price")
plt.plot(test_prediction, color="red", label="Predicted Gold Price")
plt.title("Actual vs Predicted Gold Prices")
plt.xlabel("Samples")
plt.ylabel("Gold Price")
plt.legend()
plt.show()
