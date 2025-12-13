# import des librairies
import asyncio
import json

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends

from app.services.predictions import PredictionService, get_prediction_service

from app.computervision.registry import get_prediction_models
import app.computervision.models as cv_models

from app.api.schema import PredictionResult


# creation de l'instance fastAPI
app = FastAPI()


async def check_compliance(name, queue):
    """
    Tache de vérification de la conformité d'une image
     - Dépile la file d'attente (queue) : récupération de l'image, des modèles
    de computer vision ainsi que le service de manipulation
    des demandes de prédiction.
     - Appelle la méthode predict pour chaque modèle
     - S'arrête au premier modèle prédisant une non-conformité

    :param name: nom de la tache
    :param queue: file d'attente contenant les images à traiter
    """
    prediction_requests, models, service = await queue.get()

    try:
        prediction_result = PredictionResult.model_construct()
        prediction_result.predicted_class = 0

        for model in models:
            prediction = model.predict(prediction_requests.image)
            prediction_result.predicted_class = int(np.round(prediction[0][0]))

            prediction_result.add_prediction(
                model.get_model_path(), 
                float(1 - prediction[0][0]),
                float(prediction[0][0])
            )

            if prediction_result.predicted_class == 1:
                break

        result_as_json = prediction_result.model_dump(mode="json")
        service.update_prediction(
            prediction_requests.id,
            json.dumps(result_as_json)
        )

    except Exception as e:
        print(f"Error: {str(e)}")


# Initialisation des workers de traitement des images
queue = asyncio.Queue()
workers = []
for i in range(4):
    worker = asyncio.create_task(check_compliance(f'Worker-{i}', queue))
    workers.append(worker)


@app.get('/image/healthcheck')
def health_check(
    models: list[cv_models.PredictionModel] = Depends(get_prediction_models)
):
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


@app.post("/image/compliance/request")
async def create_compliance_check_request(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
    models: list[cv_models.PredictionModel] = Depends(get_prediction_models)
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


@app.get("/image/compliance/request/{id}", response_model=PredictionResult)
async def get_compliance_check_result(
    id: int,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Récupérer le résultat de la vérification de conformité
    
    :param id: Description
    :type id: int
    :param service: Description
    :type service: PredictionService
    """
    try:
        prediction_request = service.get(id)

        prediction_dict = json.loads(
            prediction_request.result
        )
        return PredictionResult(**prediction_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))


