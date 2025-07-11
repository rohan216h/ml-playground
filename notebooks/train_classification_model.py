import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load Titanic dataset
df = pd.read_csv('data/titanic.csv')

# Select relevant features and drop missing
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df.dropna(inplace=True)

# Encode categorical features
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Split features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
os.makedirs('models/classification', exist_ok=True)
joblib.dump(model, 'models/classification/titanic_rf.pkl')
