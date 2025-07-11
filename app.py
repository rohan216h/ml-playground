import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from utils.preprocess import preprocess_data
from utils.model_loader import load_model, list_models
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Playground", layout="wide")

st.sidebar.title("🔧 Settings")

# Select task type
task_type = st.sidebar.radio("Select Task", ["Classification", "Regression", "Image Classification"])

# Load data
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])
data = None
if uploaded_file and task_type != "Image Classification":
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

# Model selection
model_name = st.sidebar.selectbox("Select Model", list_models(task_type))
model = load_model(model_name, task_type)

# Run prediction
if st.sidebar.button("Run Prediction") and data is not None:
    try:
        X, y = preprocess_data(data, task_type)
        y_pred = model.predict(X)

        st.write(f"### Results using {model_name}")
        if task_type == "Classification":
            acc = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            st.write(f"Accuracy: {acc:.2f}")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        else:
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            st.write(f"RMSE: {rmse:.2f}")
            st.line_chart(pd.DataFrame({"Actual": y, "Predicted": y_pred}))

    except Exception as e:
        st.error(f"Error in processing: {e}")

# Image Classification placeholder
if task_type == "Image Classification":
    image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if image and st.sidebar.button("Classify Image"):
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Placeholder prediction result
        st.success("Predicted Class: ExampleClass (placeholder)")
