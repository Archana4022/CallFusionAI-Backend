from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from pydantic import BaseModel, Field

# Load the trained model
try:
    with open("optimized_voip_cost_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Initialize FastAPI app
app = FastAPI(title="AI VOIP Cost Optimizer API", description="Predicts the cost of a VOIP call 📞💰")

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
