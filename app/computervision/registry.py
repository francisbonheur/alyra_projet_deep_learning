MODELS = {}


def prediction_model(model_path):
    """
    Décorateur pour l'enregistrement des classes héritant
    de PredictionModel dans un registry
    
    :param model_path: Description
    """
    def decorator(cls):
        if model_path in MODELS:
            raise ValueError(f"'{model_path}' is already registered.")
        MODELS[model_path] = cls
        return cls
    return decorator


def get_prediction_models():
    """
    Renvoie un instance de chaque classe du registre
    """
    models = []
    for model_path, cls in MODELS.items():
        try:
            models.append(cls(model_path))
        except Exception as e:
            print(f"Error loading models: {e}")

    return models
