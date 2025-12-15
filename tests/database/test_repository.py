from unittest.mock import MagicMock, patch, Mock

from sqlmodel import Session
from app.database.repository import PredictionRepositoryImpl
from app.database.model import PredictionRequest


mock_session = MagicMock(spec=Session)
repository = PredictionRepositoryImpl(mock_session)


def test_add():
    image = 'O'
    prediction_request = PredictionRequest(image)
    repository.add(prediction_request)
    mock_session.add.assert_called_with(prediction_request)
    mock_session.commit.assert_called()
    mock_session.refresh.assert_called_with(prediction_request)


def test_update():
    image = 'O'
    prediction_request = PredictionRequest(image)

    with patch.object(repository, 'get') as mock_get:
        with patch.object(repository, 'add') as mock_add:
            # Configure the mock's return value
            mock_get.return_value = prediction_request
            updated_request = repository.update(1, "result", "done")

            mock_get.assert_called_with(1)
            mock_add.assert_called_with(prediction_request)

            assert updated_request.result == "result"
            assert updated_request.status == "done"


def test_get():
    image = '0'
    prediction_request = PredictionRequest(image)
    prediction_request.result = "resultat"
    prediction_request.status = "pending"

    mock_result = Mock()
    mock_result.first.return_value = prediction_request
    mock_session.exec.return_value = mock_result

    assert repository.get(2).status == "pending"
    assert repository.get(2).result == "resultat"


def test_get_all():
    image = '0'
    prediction_request_1 = PredictionRequest(image)
    prediction_request_1.result = "resultat"
    prediction_request_1.status = "pending"

    prediction_request_2 = PredictionRequest(image)
    prediction_request_2.result = "result"
    prediction_request_2.status = "done"

    mock_result = Mock()
    mock_result.all.return_value = [prediction_request_1, prediction_request_2]
    mock_session.exec.return_value = mock_result

    assert repository.get_all()[0].status == "pending"
    assert repository.get_all()[0].result == "resultat"

    assert repository.get_all()[1].status == "done"
    assert repository.get_all()[1].result == "result"
