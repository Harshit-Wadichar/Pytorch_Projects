from fastapi import FastAPI
# Reload trigger
from pydantic import BaseModel
import torch
from model import StudentModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentData(BaseModel):
    hours_studied: float
    sleep_hours: float
    attendance_percent: float
    previous_score: float

model = StudentModel()
try:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
except FileNotFoundError:
    print("Model not found. Please run train.py first.")

@app.post("/predict")
def predict(data: StudentData):
    # Normalize input consistent with training
    hours = data.hours_studied / 24.0
    sleep = data.sleep_hours / 24.0
    attendance = data.attendance_percent / 100.0
    score = data.previous_score / 100.0
    
    input_tensor = torch.tensor([[hours, sleep, attendance, score]], dtype=torch.float32)
    
    
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_score = prediction.item() * 100.0
        
    # Clamp result between 0 and 100
    predicted_score = max(0.0, min(100.0, predicted_score))
        
    return {
        "predicted_score": round(predicted_score, 2)
    }

@app.get("/")
def home():
    return {"message": "Student Prediction API is running"}
