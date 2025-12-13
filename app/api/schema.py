
from pydantic import BaseModel


class Probabilities(BaseModel):
    negatif: float
    positif: float


class Prediction(BaseModel):
    model_path: str
    probabilites: Probabilities


class PredictionResult(BaseModel):
    predicted_class: int
    predictions: list[Prediction] = []

    def add_prediction(
        self,
        model_path: str,
        negative_proba: float,
        positive_proba: float
    ):
        probabilities = Probabilities.model_construct()
        probabilities.negatif = negative_proba
        probabilities.positif = positive_proba
        prediction = Prediction.model_construct()
        prediction.model_path = model_path
        prediction.probabilites = probabilities
        self.predictions.append(prediction)


