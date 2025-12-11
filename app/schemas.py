from pydantic import BaseModel


class ImagePrediction(BaseModel):
    predicted_class: int
    predictions: list[dict]
