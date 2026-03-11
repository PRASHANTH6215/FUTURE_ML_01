# Sales Demand Forecasting using Machine Learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# 2. Load Dataset
import os

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "DATASET.csv")
df = pd.read_csv(r"C:\Users\PRASHANTH\OneDrive\文档\Desktop\FT TASk1\DATASET.csv", encoding='latin1')
print("First 5 rows of dataset:")
print(df.head())

# 3. Data Cleaning
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Sort dataset by date
df = df.sort_values('Order Date')

# Remove missing values
df = df.dropna()

# 4. Feature Engineering
df['year'] = df['Order Date'].dt.year
df['month'] = df['Order Date'].dt.month
df['day'] = df['Order Date'].dt.day
df['day_of_week'] = df['Order Date'].dt.dayofweek

# 5. Aggregate Sales by Date
sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()
print("\nAggregated Sales Data:")
print(sales_data.head())

# 6. Prepare Features & Target
X = df[['year','month','day','day_of_week']]
y = df['Sales']

# 7. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 8. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModel Training Completed")

# 9. Predictions
predictions = model.predict(X_test)
print("\nSample Predictions:")
print(predictions[:10])

# 10. Model Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("\nModel Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)

# 11. Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Sales")
plt.plot(predictions, label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.show()

# 12. Predict Future Sales
future_data = pd.DataFrame({
    "year":[2026],
    "month":[6],
    "day":[15],
    "day_of_week":[1]
})

future_sales = model.predict(future_data)

print("\nPredicted Future Sales:", future_sales)