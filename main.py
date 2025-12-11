# import des librairies
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from app.schemas import ImagePrediction

from app.image_model import ViolencePredictionCVModel, get_model


# creation de l'instance fastAPI
app = FastAPI()

@app.get('/healthcheck')
def health_check(model: ViolencePredictionCVModel = Depends(get_model)):
    if not model:
        raise HTTPException(
            status_code=500,
            detail={ 
                "status": "error", 
                "message": "Model not loaded."
            }
        )

    return {
        "status": "ok", 
        "message": "API is running smoothly."
    }


@app.post("/predict_violence", response_model=ImagePrediction)
async def predict_image(
    file: UploadFile = File(...),
    model: ViolencePredictionCVModel = Depends(get_model)
):
    try:
        image = await file.read()
        prediction = model.predict(image)
        probas = {
            "0": float(1 - prediction[0][0]),
            "1": float(prediction[0][0])
        }

        return {
            "Number": int(np.round(prediction[0][0])),
            "Proba": probas
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))
