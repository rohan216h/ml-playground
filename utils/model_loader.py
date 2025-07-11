import os
import joblib
import tensorflow as tf

MODEL_PATHS = {
    "classification": "models/classification",
    "regression": "models/regression",
    "image classification": "models/image classification"
}

def list_models(task_type):
    model_dir = MODEL_PATHS.get(task_type.lower())
    if not model_dir or not os.path.exists(model_dir):
        return []
    return [f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.h5'))]

def load_model(model_name, task_type):
    model_dir = MODEL_PATHS.get(task_type.lower())
    if not model_dir:
        raise ValueError(f"Unknown task type: {task_type}")

    model_path = os.path.join(model_dir, model_name)
    if model_name.endswith(".pkl"):
        return joblib.load(model_path)
    elif model_name.endswith(".h5"):
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError("Unsupported model format")
