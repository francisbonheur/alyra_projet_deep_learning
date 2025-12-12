from pydantic import BaseModel


class ImagePrediction(BaseModel):
    """
    Format de la réponse de prédiction d'image
    """
    predicted_class: int
    predictions: list[dict]
