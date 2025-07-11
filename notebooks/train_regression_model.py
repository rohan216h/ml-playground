import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset (example: boston.csv)
df = pd.read_csv('data/HousingData.csv')
df.dropna(inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs('models/regression', exist_ok=True)
joblib.dump(model, 'models/regression/linear_regression.pkl')
