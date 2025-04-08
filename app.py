from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from enum import Enum
from typing import Optional
from datetime import datetime
import joblib
import pandas as pd
import mysql.connector
from mysql.connector import Error
from contextlib import closing
import csv
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load AI model and expected dummy columns
model = joblib.load("voip_cost_model.pkl")

# Connect to MySQL database
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

# FastAPI app
app = FastAPI(title="CallFusion AI - VOIP Cost Optimizer")

# CORS (Allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to CallFusion AI VOIP Cost Optimizer API"}

# Enums for strict validation
class CarrierEnum(str, Enum):
    carrier_a = "Carrier A"
    carrier_b = "Carrier B"
    carrier_c = "Carrier C"
    carrier_d = "Carrier D"

class TimeOfDayEnum(str, Enum):
    morning = "Morning"
    afternoon = "Afternoon"
    evening = "Evening"
    night = "Night"

class CallData(BaseModel):
    caller_id: str
    receiver_id: str
    duration: int
    carrier: CarrierEnum
    latency: float
    time_of_day: TimeOfDayEnum

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

        with closing(db.cursor()) as cursor:
            query = """
                INSERT INTO call_logs (caller_id, receiver_id, duration, carrier, latency, time_of_day, predicted_cost)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (call.caller_id, call.receiver_id, call.duration, call.carrier, call.latency, call.time_of_day, predicted_cost)
            cursor.execute(query, values)
            db.commit()

        return {"predicted_cost": round(predicted_cost, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        for carrier in CarrierEnum:
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
        for tod in TimeOfDayEnum:
            if tod != call.time_of_day:
                temp = base_input.copy()
                temp["Time of Day"] = tod
                temp_df = pd.get_dummies(pd.DataFrame([temp]))
                cost = model.predict(temp_df)[0]
                suggestions.append({
                    "suggestion": f"Try calling in the {tod}",
                    "estimated_cost": round(cost, 2)
                })

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
    try:
        sort_field_map = {
            "duration": "duration",
            "predicted_cost": "predicted_cost",
            "id": "id"
        }

        if sort_by not in sort_field_map:
            raise HTTPException(status_code=400, detail="Invalid sort_by value")

        order = "ASC" if order.lower() == "asc" else "DESC"
        sort_field = sort_field_map[sort_by]

        query = """
            SELECT caller_id, receiver_id, duration, carrier, latency, time_of_day, predicted_cost, created_at
            FROM call_logs
            WHERE (caller_id LIKE %s OR carrier LIKE %s)
        """
        params = [f"%{search}%", f"%{search}%"]

        if start_date:
            query += " AND DATE(created_at) >= %s"
            params.append(start_date)
        if end_date:
            query += " AND DATE(created_at) <= %s"
            params.append(end_date)

        query += f" ORDER BY {sort_field} {order} LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with closing(db.cursor()) as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

        columns = ["caller_id", "receiver_id", "duration", "carrier", "latency", "time_of_day", "predicted_cost", "created_at"]
        results = [dict(zip(columns, row)) for row in rows]

        if not results:
            return JSONResponse(content={"message": "No call records found."}, status_code=200)

        if format == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
            output.seek(0)
            return StreamingResponse(output, media_type="text/csv", headers={
                "Content-Disposition": "attachment; filename=call_history.csv"
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics():
    try:
        with closing(db.cursor()) as cursor:
            # 1. Cost trend over time
            cursor.execute("""
                SELECT DATE(created_at), SUM(predicted_cost)
                FROM call_logs
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at) ASC
            """)
            cost_trend = [{"date": str(row[0]), "total_cost": float(row[1])} for row in cursor.fetchall()]

            # 2. Latency heatmap
            cursor.execute("""
                SELECT time_of_day, AVG(latency)
                FROM call_logs
                GROUP BY time_of_day
            """)
            latency_heatmap = [{"time_of_day": row[0], "avg_latency": float(row[1])} for row in cursor.fetchall()]

            # 3. Duration vs Cost
            cursor.execute("SELECT duration, predicted_cost FROM call_logs")
            scatter_data = [{"duration": row[0], "cost": float(row[1])} for row in cursor.fetchall()]

        return JSONResponse(content={
            "cost_trend": cost_trend,
            "latency_heatmap": latency_heatmap,
            "scatter_data": scatter_data
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)
