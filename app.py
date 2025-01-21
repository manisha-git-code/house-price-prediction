# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv("data/train.csv")
print("Dataset Loaded Successfully!")

# Step 2: Explore the dataset
print("First 5 rows of data:")
print(data.head())

# Display basic statistics
print("\nDataset Summary:")
print(data.describe())

# Step 3: Preprocess the data
# Drop rows with missing values if any
data = data.dropna()

# Features (X) and Target (y)
X = data.drop("medv", axis=1)  # 'MEDV' is the target column
y = data["medv"]

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData Split Completed!")

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel Trained Successfully!")

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 8: Visualize predictions
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
