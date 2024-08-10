from keras.models import load_model
from fastapi import HTTPException, status


def load_keras_model(model_path: str):
    try:
        # Replace 'model_path' with the path to your model file
        model = load_model(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Model not found or failed to load: {str(e)}")
