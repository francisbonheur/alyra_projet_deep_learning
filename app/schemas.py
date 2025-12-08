from pydantic import BaseModel


class ImagePrediction(BaseModel):
    Number: int
    Proba: dict
