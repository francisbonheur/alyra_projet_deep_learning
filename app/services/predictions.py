from app.database.repository import PredictionRepository, PredictionRepositoryImpl
from app.database.model import PredictionRequest


class PredictionService:
    """
    Service métier de manipulation des demandes de prédictions
    """
    def __init__(self, repository: PredictionRepository):
        self.repository = repository

    def add_prediction(self, image: bytes) -> PredictionRequest | None:
        created_prediction = self.repository.add(PredictionRequest(image))
        return created_prediction

    def get(self, id: int) -> PredictionRequest | None:
        prediction_request = self.repository.get(id)
        return prediction_request

    def get_all(self) -> list[PredictionRequest]:
        prediction_requests = self.repository.get_all()
        return prediction_requests

    def update_prediction(self, id: int, result: str) -> PredictionRequest | None:
        prediction_request = self.repository.update(id, result, "done")
        return prediction_request


def get_prediction_service() -> PredictionService:
    return PredictionService(PredictionRepositoryImpl())
