from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import mysql.connector
from pydantic import BaseModel
import uvicorn

# Load trained AI model
model = joblib.load("voip_cost_model.pkl")

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Update your MySQL username
    password="test12",  # Update your MySQL password
    database="voip_optimizer"
)
cursor = db.cursor()

# FastAPI app instance
app = FastAPI(title="AI VOIP Cost Optimizer")

# Request model for predictions
class CallData(BaseModel):
    caller_id: str
    receiver_id: str
    duration: int
    carrier: str
    latency: float
    time_of_day: str

# ✅ Correct Route
@app.post("/predict-cost/")
async def predict_cost(call: CallData):
    try:
        input_data = pd.DataFrame([{
            "Duration (s)": call.duration,
            "Carrier": call.carrier,
            "Latency (ms)": call.latency,
            "Time of Day": call.time_of_day
        }])

        input_data = pd.get_dummies(input_data)
        predicted_cost = model.predict(input_data)[0]

        query = """INSERT INTO call_logs (caller_id, receiver_id, duration, carrier, latency, time_of_day, predicted_cost)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        values = (call.caller_id, call.receiver_id, call.duration, call.carrier, call.latency, call.time_of_day, predicted_cost)
        cursor.execute(query, values)
        db.commit()

        return {"predicted_cost": round(predicted_cost, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
