import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

# This script performs house price prediction using a Random Forest Regressor.

# Load the dataset
df = pd.read_csv('data/train.csv')

#Data Exploration
# print(df.head())
# print(df.info())
# print(df.describe())

numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
# print(correlation['SalePrice'].sort_values(ascending=False))

# Correlation with SalePrice results:
# OverallQual      0.790982
# GrLivArea        0.708624
# GarageCars       0.640409
# GarageArea       0.623431
# TotalBsmtSF      0.613581
# 1stFlrSF         0.605852
# FullBath         0.560664
# TotRmsAbvGrd     0.533723
# YearBuilt        0.522897
# YearRemodAdd     0.507101

# Data Visualization
# top_corr_features = correlation['SalePrice'].sort_values(ascending=False).head(10).index
# plt.figure(figsize=(10, 8))
# sns.heatmap(numeric_df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap of Top Features with SalePrice')
# plt.show()

# Data Preprocessing
features = ['GrLivArea', 'GarageCars', 'OverallQual', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
data = df[features + ['SalePrice']].dropna()
X = data[features]
y = data['SalePrice']

print(df[features + ['SalePrice']].describe())


# Split the dataset into training and testing sets
# 20% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
# print(f"RMSE: {rmse:.2f}")
# print(f"R² Score: {r2:.4f}")

# Save the model
joblib.dump(model, 'model/house_price_model.pkl')

# Test the saved model with a sample input
loaded_model = joblib.load('model/house_price_model.pkl')
sample_prediction = loaded_model.predict([[1500, 2, 7, 800, 2, 2005]])  # GrLivArea, GarageCars, OverallQual, TotalBsmtSF, FullBath, YearBuilt
print(f"Predicted SalePrice: ${sample_prediction[0]:,.2f}")