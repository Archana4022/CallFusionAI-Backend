import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Load the trained model
with open("optimized_voip_cost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load test dataset
df = pd.read_csv("D:/AI_VOIP_Cost_Optimizer/data/calls_data.csv")
df.columns = df.columns.str.strip()  # Remove unwanted spaces

# Define the same features used in training
features = ['Duration (s)', 'Latency (ms)', 'Carrier_Carrier B', 'Carrier_Carrier C', 
            'Carrier_Carrier D', 'Time of Day_Evening', 'Time of Day_Morning', 'Time of Day_Night']
target = "Cost ($)"

# Ensure categorical encoding matches the training phase
df = pd.get_dummies(df, columns=["Carrier", "Time of Day"], drop_first=True)

# Select only the features used in training
X_test = df[features]
y_test = df[target]

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Display sample predictions
df_results = pd.DataFrame({"Actual Cost": y_test, "Predicted Cost": y_pred})
print(df_results.head(10))  # Print first 10 results
