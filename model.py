import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


# 1. Load dataset

# Make sure this file exists in your repo: dataset/student_data.csv
data = pd.read_csv("dataset/student_data.csv")

# Expect columns:
# Student_ID, Attendance, Internal_Marks, Study_Hours,
# Previous_GPA, Family_Income, Extracurricular, Final_Result


# 2. Encode categorical columns

le_result = LabelEncoder()
data["Final_Result"] = le_result.fit_transform(data["Final_Result"])
# (Optional) save label encoder if you need inverse_transform later

data["Extracurricular"] = data["Extracurricular"].map({"Yes": 1, "No": 0})


# 3. Split features / target

X = data.drop(["Final_Result", "Student_ID"], axis=1)
y = data["Final_Result"]


# 4. Scale features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Train / test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# 6. Define MLP (Neural Network)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

print(" Training MLP Neural Network...")
model.fit(X_train, y_train)


# 7. Evaluate model

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n Test Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"\n Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")


# 8. Save model + scaler

with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n New model and scaler saved successfully (Render-compatible).")
