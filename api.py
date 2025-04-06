from fastapi import FastAPI, HTTPException, FastAPI, Query, Response, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from fastapi.responses import StreamingResponse, JSONResponse
import io
import csv
from datetime import datetime
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from collections import defaultdict

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Update your MySQL username
    password="test12",  # Update your MySQL password
    database="voip_optimizer"
)
cursor = db.cursor()

# Load environment variables from .env file
load_dotenv()

# Get database connection details
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Connect to the database
try:
    db = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="test12",
        database="voip_optimizer"
    )
    print("Successfully connected to the database!")
except mysql.connector.Error as err:
    print(f"Error: {err}")

# Load the trained model
try:
    with open("optimized_voip_cost_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Initialize FastAPI app
app = FastAPI(title="AI VOIP Cost Optimizer API", description="Predicts the cost of a VOIP call 📞💰")

# Allow frontend to access backend (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define root route to avoid "Not Found" error
@app.get("/")
def home():
    return JSONResponse(content={"message": "Welcome to AI VOIP Cost Optimizer API 🎯. Use /docs for API documentation."})

# Define allowed categories
valid_carriers = ["Carrier A", "Carrier B", "Carrier C", "Carrier D"]
valid_times = ["Morning", "Afternoon", "Evening", "Night"]

# Define input data structure with validation
class CallData(BaseModel):
    duration: float = Field(..., gt=0, description="Call duration must be greater than 0 seconds.")
    latency: float = Field(..., ge=0, description="Latency must be 0 or greater.")
    carrier: str = Field(..., description=f"Carrier must be one of: {', '.join(valid_carriers)}")
    time_of_day: str = Field(..., description=f"Time of day must be one of: {', '.join(valid_times)}")

@app.post("/predict_cost/")
async def predict_cost(data: CallData):
    """Predicts the cost of a VOIP call based on input data."""
    
    # ✅ Validate Carrier
    if data.carrier not in valid_carriers:
        raise HTTPException(status_code=400, detail={"error": "Invalid carrier", "valid_options": valid_carriers})
    
    # ✅ Validate Time of Day
    if data.time_of_day not in valid_times:
        raise HTTPException(status_code=400, detail={"error": "Invalid time of day", "valid_options": valid_times})

    # Convert input into DataFrame
    input_data = pd.DataFrame([{
        "Duration (s)": data.duration,
        "Latency (ms)": data.latency,
        "Carrier_Carrier B": 1 if data.carrier == "Carrier B" else 0,
        "Carrier_Carrier C": 1 if data.carrier == "Carrier C" else 0,
        "Carrier_Carrier D": 1 if data.carrier == "Carrier D" else 0,
        "Time of Day_Evening": 1 if data.time_of_day == "Evening" else 0,
        "Time of Day_Morning": 1 if data.time_of_day == "Morning" else 0,
        "Time of Day_Night": 1 if data.time_of_day == "Night" else 0,
    }])

    try:
        # Make prediction
        predicted_cost = model.predict(input_data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Prediction failed", "message": str(e)})

    # Return the result
    return {"predicted_cost": round(predicted_cost, 2)}

@app.post("/suggest-optimizations/")
async def suggest_optimizations(call: CallData):
    try:
        base_input = {
            "Duration (s)": call.duration,
            "Latency (ms)": call.latency,
            "Carrier": call.carrier,
            "Time of Day": call.time_of_day
        }

        suggestions = []

        # Try other carriers
        for carrier in ["Carrier A", "Carrier B", "Carrier C", "Carrier D"]:
            if carrier != call.carrier:
                temp = base_input.copy()
                temp["Carrier"] = carrier
                temp_df = pd.get_dummies(pd.DataFrame([temp]))
                cost = model.predict(temp_df)[0]
                suggestions.append({
                    "suggestion": f"Try using {carrier}",
                    "estimated_cost": round(cost, 2)
                })

        # Try different times of day
        for tod in ["Morning", "Afternoon", "Evening", "Night"]:
            if tod != call.time_of_day:
                temp = base_input.copy()
                temp["Time of Day"] = tod
                temp_df = pd.get_dummies(pd.DataFrame([temp]))
                cost = model.predict(temp_df)[0]
                suggestions.append({
                    "suggestion": f"Try calling in the {tod}",
                    "estimated_cost": round(cost, 2)
                })

        # Return top 3 suggestions sorted by lowest cost
        sorted_suggestions = sorted(suggestions, key=lambda x: x["estimated_cost"])[:3]

        return {"optimizations": sorted_suggestions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/call-history")
def get_call_history(
    search: str = Query(default="", description="Search by Caller ID or Carrier"),
    sort_by: str = Query(default="id", description="Sort by 'duration', 'predicted_cost', or 'id'"),
    order: str = Query(default="desc", description="Sort order: 'asc' or 'desc'"),
    limit: int = Query(default=10, description="Number of records to return"),
    offset: int = Query(default=0, description="Offset for pagination"),
    start_date: Optional[str] = Query(default=None, description="Filter by start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="Filter by end date (YYYY-MM-DD)"),
    format: str = Query(default="json", description="Return format: 'json' or 'csv'")
):
    # Allowed fields to sort
    sort_field_map = {
        "duration": "duration",
        "predicted_cost": "predicted_cost",
        "id": "id"
    }
    sort_field = sort_field_map.get(sort_by, "id")
    order = "ASC" if order.lower() == "asc" else "DESC"

    # SQL base
    base_query = """
        SELECT caller_id, receiver_id, duration, carrier, latency, time_of_day, predicted_cost, created_at
        FROM call_logs
        WHERE (caller_id LIKE %s OR carrier LIKE %s)
    """

    filters = []
    params = [f"%{search}%", f"%{search}%"]

    # Add optional filters
    if start_date:
        filters.append("DATE(created_at) >= %s")
        params.append(start_date)
    if end_date:
        filters.append("DATE(created_at) <= %s")
        params.append(end_date)

    # Append filters if any
    if filters:
        base_query += " AND " + " AND ".join(filters)

    # Final sort + pagination
    base_query += f" ORDER BY {sort_field} {order} LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    cursor.execute(base_query, tuple(params))
    rows = cursor.fetchall()
    columns = ["caller_id", "receiver_id", "duration", "carrier", "latency", "time_of_day", "predicted_cost", "created_at"]
    results = [dict(zip(columns, row)) for row in rows]

    if not results:
        return JSONResponse(content={"message": "No calls found."}, status_code=200)

    # 📥 Export to CSV
    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
        output.seek(0)
        return StreamingResponse(output, media_type="text/csv", headers={
            "Content-Disposition": "attachment; filename=call_history.csv"
        })

    # ✅ Return JSON by default
    return results

@app.get("/analytics")
def get_analytics():
    # 1. Cost Trend Over Time (daily total cost)
    cursor.execute("""
        SELECT DATE(created_at) as day, SUM(predicted_cost) as total_cost
        FROM call_logs
        GROUP BY day
        ORDER BY day ASC
    """)
    cost_trend = [{"date": str(row[0]), "total_cost": float(row[1])} for row in cursor.fetchall()]

    # 2. Latency Heatmap: Average latency by time_of_day
    cursor.execute("""
        SELECT time_of_day, AVG(latency) as avg_latency
        FROM call_logs
        GROUP BY time_of_day
    """)
    latency_heatmap = [{"time_of_day": row[0], "avg_latency": float(row[1])} for row in cursor.fetchall()]

    # 3. Duration vs Cost Scatter Data
    cursor.execute("""
        SELECT duration, predicted_cost
        FROM call_logs
    """)
    scatter_data = [{"duration": int(row[0]), "cost": float(row[1])} for row in cursor.fetchall()]

    return JSONResponse(content={
        "cost_trend": cost_trend,
        "latency_heatmap": latency_heatmap,
        "scatter_data": scatter_data
    })

