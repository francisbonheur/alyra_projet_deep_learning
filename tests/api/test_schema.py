from app.api.schema import PredictionResult


def test_prediction_result_add_prediction():
    prediction_result = PredictionResult.model_construct()
    prediction_result.add_prediction("model_path_1", 0.1, 0.9)
    prediction_result.add_prediction("model_path_2", 0.2, 0.8)

    assert prediction_result.predictions[0].model_path == "model_path_1"
    assert prediction_result.predictions[0].probabilites.negatif == 0.1
    assert prediction_result.predictions[0].probabilites.positif == 0.9

    assert prediction_result.predictions[1].model_path == "model_path_2"
    assert prediction_result.predictions[1].probabilites.negatif == 0.2
    assert prediction_result.predictions[1].probabilites.positif == 0.8
