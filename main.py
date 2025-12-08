# import des librairies
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from app.schemas import ImagePrediction

from app.image_preprocessing import preprocess_image
from app.image_model import load_model, predict


# creation de l'instance fastAPI
app = FastAPI()

# chargement du modèle au démarrage de l'Application
model = load_model("models/mobilenetv2_flowers_finetuned.keras")


@app.get('/healthcheck')
def health_check():
    return {
        "status": "ok", 
        "message": "API is running smoothly."
    }


@app.post("/predict_violence", response_model=ImagePrediction)
async def predict_image(file: UploadFile = File(...)):
    try:
        image = await file.read()
        preprocessed_image = preprocess_image(image)
        prediction = predict(model, preprocessed_image)
        round_prediction = {
            i: round(p, 2) for i, p in enumerate(prediction[0])
        }

        return {
            "Number": np.argmax(prediction[0]),
            "Proba": round_prediction
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Error reading image file")
