import json
import numpy as np

from app.api.schema import PredictionResult


class ComplianceCheckWorker():
    def __init__(self, name):
        self.name = name

    async def check_compliance(self, queue):
        """
        Tache de vérification de la conformité d'une image
         - Dépile la file d'attente (queue) : récupération de l'image, 
        des modèles de computer vision ainsi que le service de manipulation
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
                prediction_result.predicted_class = int(
                    np.round(prediction[0][0])
                )

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
