import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df, task_type):
    df = df.copy()

    # Basic preprocessing: drop NA, encode categoricals
    df.dropna(inplace=True)

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode target if classification
    if task_type == "Classification":
        if y.dtype == 'object' or y.nunique() <= 10:
            le = LabelEncoder()
            y = le.fit_transform(y)

    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y
