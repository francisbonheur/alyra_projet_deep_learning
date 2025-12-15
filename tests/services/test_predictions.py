from unittest.mock import MagicMock

from app.services.predictions import PredictionService
from app.database.repository import PredictionRepository

from app.database.model import PredictionRequest


mock_repository = MagicMock(spec=PredictionRepository)
prediction_service = PredictionService(mock_repository)


def test_add_prediction():
    image = '0'
    prediction_request: PredictionRequest = PredictionRequest(image)
    prediction_request.result = "result"
    prediction_request.status = "pending"

    mock_repository.add.return_value = prediction_request

    created_prediction: PredictionRequest = prediction_service.add_prediction(
        image
    )

    assert created_prediction.status == prediction_request.status
    assert created_prediction.result == prediction_request.result


def test_get():
    image = '0'
    prediction_request: PredictionRequest = PredictionRequest(image)
    prediction_request.result = "result"
    prediction_request.status = "pending"

    mock_repository.get.return_value = prediction_request

    prediction: PredictionRequest = prediction_service.get(1)

    mock_repository.get.assert_called_with(1)
    assert prediction.status == prediction_request.status
    assert prediction.result == prediction_request.result


def test_get_all():
    image = '0'
    prediction_request_1: PredictionRequest = PredictionRequest(image)
    prediction_request_1.result = "result"
    prediction_request_1.status = "pending"

    prediction_request_2: PredictionRequest = PredictionRequest(image)
    prediction_request_2.result = "resultat"
    prediction_request_2.status = "done"

    mock_repository.get_all.return_value = [
        prediction_request_1,
        prediction_request_2
    ]

    predictions: PredictionRequest = prediction_service.get_all()

    mock_repository.get_all.assert_called()
    assert predictions[0].status == "pending"
    assert predictions[0].result == "result"

    assert predictions[1].status == "done"
    assert predictions[1].result == "resultat"


def test_update_prediction():
    image = '0'
    prediction_request: PredictionRequest = PredictionRequest(image)
    prediction_request.result = "result"
    prediction_request.status = "pending"

    mock_repository.update.return_value = prediction_request

    prediction: PredictionRequest = prediction_service.update_prediction(
        1,
        "result"
    )

    mock_repository.update.assert_called_with(1, "result", "done")
    assert prediction.status == prediction_request.status
    assert prediction.result == prediction_request.result
