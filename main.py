# import des librairies
import asyncio

import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from app.api.schemas import ImagePrediction

from app.computervision.models import PredictionModel, get_models

from app.services.predictions import PredictionService, get_prediction_service


# creation de l'instance fastAPI
app = FastAPI()


async def check_compliance(name, queue):
    """
    Vérifier la conformité de façon asynchrone

    :param prediction: Description
    :type prediction: Prediction
    :param models: Description
    :type models: list[PredictionModel]
    :param service: Description
    :type service: PredictionService
    """
    prediction_requests, models, service = await queue.get()

    try:
        predicted_class = 0
        predictions = []

        for model in models:
            prediction = model.predict(prediction_requests.image)
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

        prediction_result = {
            "predicted_class": predicted_class,
            "predictions": predictions
        }

        service.update_prediction(
            prediction_requests.id,
            str(prediction_result)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))


queue = asyncio.Queue()
workers = []
for i in range(4):
    worker = asyncio.create_task(check_compliance(f'Worker-{i}', queue))
    workers.append(worker)


@app.get('/image/healthcheck')
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


@app.post("/image/compliance", response_model=ImagePrediction)
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


@app.post("/image/compliance/request")
async def create_compliance_check_request(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
    models: list[PredictionModel] = Depends(get_models)
):
    """
    Créer une request de vérification de la conformité
    de l'image

    :param file: Description
    :type file: UploadFile
    """
    try:
        image = await file.read()
        prediction_request = service.add_prediction(image)
        await queue.put((prediction_request, models, service))

        return {
            "id": prediction_request.id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))


