import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
data = pd.read_csv('dataset/student_data.csv')

# Encode categorical features
le = LabelEncoder()
data['Extracurricular'] = le.fit_transform(data['Extracurricular'])
data['Final_Result'] = le.fit_transform(data['Final_Result'])

# Split features and labels
X = data.drop(['Final_Result', 'Student_ID'], axis=1)
y = data['Final_Result']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural Network (MLP) model
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                      max_iter=500, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"\n MLP Neural Network Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
pickle.dump(model, open('trained_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("\n Model and scaler saved successfully!")
