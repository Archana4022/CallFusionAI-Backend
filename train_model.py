import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("D:/AI_VOIP_Cost_Optimizer/data/calls_data.csv")
df.columns = df.columns.str.strip()  # Remove unwanted spaces

# Drop non-numeric columns that are not needed
df = df.drop(columns=["Caller ID", "Receiver ID", "Timestamp"])

# Define features and target
features = ['Duration (s)', 'Latency (ms)', 'Carrier', 'Time of Day']
target = "Cost ($)"

# Convert categorical features into numerical using One-Hot Encoding
df = pd.get_dummies(df, columns=["Carrier", "Time of Day"], drop_first=True)

# Define final feature set
final_features = [col for col in df.columns if col != target]
print(f"📌 Final Features Used for Training: {final_features}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df[final_features], df[target], test_size=0.2, random_state=42)

# Initialize and train model
model = lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Save the trained model
model_filename = "optimized_voip_cost_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved successfully as {model_filename}")
