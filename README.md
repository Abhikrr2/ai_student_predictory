# AI-Powered Student Performance Prediction and Analysis System

This Flask-based machine learning project predicts student academic performance using Random Forest algorithm.

## Features
- Upload CSV dataset and get predictions
- Dashboard showing Excellent, Average, and At-Risk students
- Student login to check personal result

## Run Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python model.py`
3. Run app: `python app.py`
4. Visit `http://127.0.0.1:5000/` in browser


AI Student Performance Predictor
A Machine Learning + Neural Network-based web platform to predict student performance and classify them as Excellent, Average, or At Risk.

Live Demo: Deployed on Render Cloud Hosting

Project Summary
Educational institutions often struggle to monitor student performance early and intervene before it’s too late. This application provides:

Predictive analytics using ML models

Neural Network for high-accuracy classification

Admin dashboard with interactive charts and insights

Student login to check personal results

Upload and process large CSV datasets (thousands of students)

Export comprehensive PDF reports

Goal: Enable teachers, administrators, and advisors to make early, data-driven decisions, supporting students effectively.

Machine Learning Models Used
Multiple models were examined, with the best performer selected for deployment:

Model	Test Accuracy	Cross-Validation Score	Notes
Logistic Regression	85%	79%	Fast but limited
SVM	80%	88%	High generalization
Random Forest	95%	92%	Best classical ML model
Neural Network (NN)	97%	94%	Final selected model
Final Deployment: Neural Network (Deep Learning)

Neural Network Architecture
Input Layer: Normalized data (6 inputs)

Hidden Layer 1: 64 neurons (ReLU)

Hidden Layer 2: 32 neurons (ReLU)

Output Layer: 3 classes (Softmax - Excellent, Average, At Risk)

Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 50

Batch Size: 32

Tech Stack
Backend:

Python 3.9

Flask (API, admin, student portal)

Gunicorn (Render hosting)

Machine Learning:

scikit-learn

TensorFlow / Keras

NumPy

Pandas

Frontend:

HTML5 & CSS3

Bootstrap 5

Plotly.js (interactive graphs)

Database / Storage:

CSV files for datasets

Pickle for model serialization (trained_model.pkl, scaler.pkl)

Deployment:

Render Cloud Hosting

Gunicorn server

runtime.txt (Python version)

requirements.txt (dependencies)

Dataset Structure
Input CSV columns:

Column Name	Description
Student_ID	Unique ID
Attendance	% attendance
Internal_Marks	Marks from internal exams
Study_Hours	Avg hours studied per day
Previous_GPA	Last semester GPA
Family_Income	Annual family income
Extracurricular	Yes/No
Predicted Categories: Excellent, Average, At Risk

Key Features
Authentication: Admin & student login, session-based security

Upload Dataset: Upload CSV, validate structure, clean & scale data, predict performance

Interactive Dashboard: Pie chart (performance distribution), bar/line charts (trends), summary tables

Download PDF: Generates performance report using FPDF2

Student Portal: Students can check their own category

High Accuracy: Neural Network model

Scalability: Handles thousands of student records per upload

System Workflow
Admin logs in and uploads student dataset (CSV)

Data cleaning and normalization

ML/NN model predicts performance per student

Results displayed in dashboard (charts, tables)

Export results to PDF

Search for individual students

Students log in to check predicted result

Folder Structure
text
AI_Student_Predictor/
│
├── app.py
├── model.py
├── trained_model.pkl
├── scaler.pkl
├── requirements.txt
├── runtime.txt
├── Procfile
│
├── static/
│   ├── css/
│   └── js/
│
└── templates/
    ├── login.html
    ├── dashboard.html
    ├── upload.html
    ├── student_login.html
    ├── student_result.html
Future Improvements
Integrate MongoDB / PostgreSQL database

Add parent login system

Generate performance improvement suggestions with AI

Student chatbot (AI assistant)

Automatic email notifications for parents

Early warning alerts for "At Risk" students

Author
Abhishek Kumar
MCA – Central University of Haryana
GitHub: https://github.com/Abhikrr2
Email: (you can add your contact here)

