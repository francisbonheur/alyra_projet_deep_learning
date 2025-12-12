# import des librairies
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from app.schemas import ImagePrediction

from app.models import PredictionModel, get_models


# creation de l'instance fastAPI
app = FastAPI()


@app.get('/healthcheck')
def health_check(models: list[PredictionModel] = Depends(get_models)):
    """
    Endpoint de vérification de l'état de chargement
    des modèles de computer vision

    :param models: liste des modèles chargés
    """
    if len(models) == 0:
        raise HTTPException(
            status_code=500,
            detail={ 
                "status": "error",
                "message": "Models not loaded."
            }
        )

    return {
        "status": "ok", 
        "message": "API is running smoothly."
    }


@app.post("/compliance", response_model=ImagePrediction)
async def predict_image(
    file: UploadFile = File(...),
    models: list[PredictionModel] = Depends(get_models)
):
    """
    Docstring for predict_image
    
    :param file: Description
    :type file: UploadFile
    :param models: Description
    :type models: list[PredictionModel]
    """
    try:
        predicted_class = 0
        predictions = []

        image = await file.read()
        for model in models:
            prediction = model.predict(image)
            predicted_class = int(np.round(prediction[0][0]))

            predictions.append({
                "model_path": model.get_model_path(),
                "probabilites": {
                    "0": float(1 - prediction[0][0]),
                    "1": float(prediction[0][0])
                }
            })

            if predicted_class == 1:
                break

        return {
            "predicted_class": predicted_class,
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))
