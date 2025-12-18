# import des librairies
import asyncio
import json
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends

from app.services.predictions import PredictionService

from app.computervision.registry import get_prediction_models
import app.computervision.models as cv_models

from app.api.schema import PredictionResult
from app.api.worker import ComplianceCheckWorker


# creation de l'instance fastAPI
app = FastAPI()

# Initialisation des workers de traitement des images
queue = asyncio.Queue()
tasks = []
DEFAULT_WORKER_POOL_MAX_SIZE = "10"
WORKER_POOL_MAX_SIZE = os.getenv(
    "WORKER_POOL_MAX_SIZE",
    DEFAULT_WORKER_POOL_MAX_SIZE
)

for i in range(int(WORKER_POOL_MAX_SIZE)):
    worker = ComplianceCheckWorker(f'Worker-{i}')
    task = asyncio.create_task(worker.check_compliance(queue))
    tasks.append(task)


@app.get('/image/compliance/healthcheck')
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

    stats = get_prediction_models.cache_info()
    return {
        "status": "ok",
        "message": "API is running smoothly.",
        "cache": {
            "hits": stats.hits,
            "misses": stats.misses
        }
    }


@app.post("/image/compliance/request")
async def create_compliance_check_request(
    file: UploadFile = File(...),
    service: PredictionService = Depends(PredictionService),
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
    service: PredictionService = Depends(PredictionService)
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
        prediction_result = PredictionResult.model_construct()

        if prediction_request:
            if prediction_request.status == "done":
                prediction_dict = json.loads(
                    prediction_request.result
                )
                prediction_result = PredictionResult(**prediction_dict)

        return prediction_result

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error: " + str(e))
