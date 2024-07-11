from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Carga el modelo
model = joblib.load('model/gradient_boosting_model.pkl')

# Inicializa la aplicación FastAPI
app = FastAPI()

# Configura Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Endpoint para renderizar la página principal
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint para la predicción
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, 
                  RoundStartingEquipmentValue: float = Form(...), 
                  MatchHeadshots: float = Form(...), 
                  MatchAssists: float = Form(...), 
                  PrimaryAssaultRifle: int = Form(...), 
                  TeamStartingEquipmentValue: float = Form(...)):
    try:
        # Prepara los datos para la predicción
        X = np.array([[RoundStartingEquipmentValue, MatchHeadshots, MatchAssists, PrimaryAssaultRifle, TeamStartingEquipmentValue]])
        
        # Realiza la predicción
        prediction = model.predict(X)
        
        # Convertir la predicción a un entero
        prediction_int = int(round(prediction[0]))
        
        # Renderiza la respuesta con la predicción
        prediction_text = f"Muertes Predichas en la Partida: {prediction_int}"
        return templates.TemplateResponse("index.html", {"request": request, "prediction_text": prediction_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
