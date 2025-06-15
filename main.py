from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

app = FastAPI()

# Mount the static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates from templates directory
templates = Jinja2Templates(directory="templates")

# Load model, scaler, and label encoder
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    GPA: float = Form(...),
    No_of_Projects: int = Form(...),
    Internship_Experience: int = Form(...),
    Problem_Solving_Skill: int = Form(...),
    Hackathon_Participation: int = Form(...),
    Communication_Skills: int = Form(...),
    Leadership: int = Form(...),
    Skill_Python: int = Form(...),
    Skill_Java: int = Form(...),
    Skill_SQL: int = Form(...),
    Skill_Design: int = Form(...),
    Preferred_Tech_Track_AI: int = Form(...),
    Preferred_Tech_Track_Cyber: int = Form(...),
    Preferred_Tech_Track_Data: int = Form(...),
    Preferred_Tech_Track_UIUX: int = Form(...),
    Preferred_Tech_Track_Web_Dev: int = Form(...)
):
    # Prepare input for model
    input_data = [
        GPA,
        No_of_Projects,
        Internship_Experience,
        Problem_Solving_Skill,
        Hackathon_Participation,
        Communication_Skills,
        Leadership,
        Skill_Python,
        Skill_Java,
        Skill_SQL,
        Skill_Design,
        Preferred_Tech_Track_AI,
        Preferred_Tech_Track_Cyber,
        Preferred_Tech_Track_Data,
        Preferred_Tech_Track_UIUX,
        Preferred_Tech_Track_Web_Dev
    ]

    # Apply scaler
    input_scaled = scaler.transform([input_data])

    # Predict
    pred = model.predict(input_scaled)

    # Decode predicted label
    pred_label = label_encoder.inverse_transform([pred[0]])[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": pred_label
    })