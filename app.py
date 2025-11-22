from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "Abhishek_AI_Project_Secret"  # For session handling

# Load trained model & scaler

model = pickle.load(open('trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Label mapping
performance_labels = {0: "Excellent", 1: "Average", 2: "At Risk"}

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "12345"


# HOME PAGE – Choose Admin/Student

@app.route('/')
def home():
    return render_template('home.html')



# ADMIN LOGIN

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = username
            return redirect(url_for('upload'))
        else:
            return render_template('admin_login.html', error="❌ Invalid credentials. Try again.")

    return render_template('admin_login.html')


# ADMIN LOGOUT

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('home'))


# ADMIN UPLOAD PAGE

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        try:
            file = request.files['file']
            df = pd.read_csv(file)

            # Drop Final_Result if exists
            if 'Final_Result' in df.columns:
                df = df.drop(['Final_Result'], axis=1)

            # Encode extracurricular
            if 'Extracurricular' in df.columns:
                df['Extracurricular'] = df['Extracurricular'].map({'Yes': 1, 'No': 0})

            # Validate Student_ID
            if 'Student_ID' not in df.columns:
                return "⚠️ The dataset must contain a 'Student_ID' column."

            # Prepare and predict
            X = df.drop(['Student_ID'], axis=1)
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

            df['Predicted_Performance'] = [performance_labels[i] for i in y_pred]

            # Save results
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
            df.to_csv('dataset/predicted_results.csv', index=False)

            return redirect(url_for('dashboard'))

        except Exception as e:
            return f"❌ Error during upload or prediction: {str(e)}"

    return render_template('upload.html')



# DASHBOARD (Admin Only)

@app.route('/dashboard')
def dashboard():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    if not os.path.exists('dataset/predicted_results.csv'):
        return "⚠️ No predictions found. Please upload dataset first."

    df = pd.read_csv('dataset/predicted_results.csv')
    total = len(df)
    excellent = len(df[df['Predicted_Performance'] == "Excellent"])
    average = len(df[df['Predicted_Performance'] == "Average"])
    at_risk = len(df[df['Predicted_Performance'] == "At Risk"])

    # Plotly Charts
    bar_fig = px.bar(
        x=['Excellent', 'Average', 'At Risk'],
        y=[excellent, average, at_risk],
        color=['Excellent', 'Average', 'At Risk'],
        color_discrete_map={
            'Excellent': '#4CAF50',
            'Average': '#FFC107',
            'At Risk': '#F44336'
        },
        text=[excellent, average, at_risk],
        title="Student Performance Distribution"
    )
    bar_fig.update_traces(textposition='outside')
    bar_html = pio.to_html(bar_fig, full_html=False)

    pie_fig = px.pie(
        values=[excellent, average, at_risk],
        names=['Excellent', 'Average', 'At Risk'],
        color=['Excellent', 'Average', 'At Risk'],
        color_discrete_map={
            'Excellent': '#4CAF50',
            'Average': '#FFC107',
            'At Risk': '#F44336'
        },
        title="Performance Percentage",
        hole=0.3
    )
    pie_html = pio.to_html(pie_fig, full_html=False)

    return render_template(
        'dashboard.html',
        total=total,
        excellent=excellent,
        average=average,
        at_risk=at_risk,
        bar_chart=bar_html,
        pie_chart=pie_html
    )



# DOWNLOAD REPORTS (Admin Only)

@app.route('/download_excel')
def download_excel():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    if not os.path.exists('dataset/predicted_results.csv'):
        return "⚠️ No predictions available."

    df = pd.read_csv('dataset/predicted_results.csv')
    excel_path = 'dataset/predicted_results.xlsx'
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)


@app.route('/download_pdf')
def download_pdf():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    if not os.path.exists('dataset/predicted_results.csv'):
        return "⚠️ No predictions available."

    df = pd.read_csv('dataset/predicted_results.csv')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Student Performance Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, txt=f"Total Students: {len(df)}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "ID", 1)
    pdf.cell(70, 10, "Predicted Performance", 1)
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    for index, row in df.iterrows():
        pdf.cell(40, 10, str(row["Student_ID"]), 1)
        pdf.cell(70, 10, row["Predicted_Performance"], 1)
        pdf.ln(10)
        if index > 45:
            pdf.add_page()

    pdf_path = "dataset/predicted_results.pdf"
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)



# STUDENT LOGIN & RESULT

@app.route('/student', methods=['GET', 'POST'])
def student_login():
    """
    Student lookup for predicted performance.
    Handles GET (show form) and POST (search ID) safely.
    """
    if request.method == 'POST':
        try:
            raw_id = request.form.get('student_id', '').strip()
            if raw_id == '':
                # No ID entered
                return render_template('student_login.html',
                                       error="Please enter a Student ID.")

            # Don't force int – some IDs like A101 are strings
            student_id_val = raw_id

            # Check that predictions file exists
            csv_path = 'dataset/predicted_results.csv'
            if not os.path.exists(csv_path):
                return render_template('student_login.html',
                                       error="No predictions found. Please contact the admin.")

            df = pd.read_csv(csv_path)

            if 'Student_ID' not in df.columns:
                return render_template('student_login.html',
                                       error="Results file does not contain 'Student_ID' column.")

            # Normalize IDs as strings for safe comparison
            df['Student_ID_str'] = df['Student_ID'].astype(str).str.strip()
            sid = str(student_id_val).strip()

            student_row = df[df['Student_ID_str'] == sid]

            if student_row.empty:
                return render_template('student_login.html',
                                       error=f"Student ID {raw_id} not found.")
            else:
                student_row = student_row.iloc[0]
                remark = student_row.get('Predicted_Performance', 'N/A')

                return render_template(
                    'student_result.html',
                    student_id=student_row['Student_ID'],
                    remark=remark
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('student_login.html',
                                   error=f"Unexpected error: {str(e)}")

    # GET request → just show the form
    return render_template('student_login.html')

#pdf

@app.route('/student_result_pdf/<student_id>')
def student_result_pdf(student_id):
    """
    Generate a PDF report for a single student's predicted performance.
    """
    csv_path = 'dataset/predicted_results.csv'
    if not os.path.exists(csv_path):
        return "⚠️ No predictions found. Please contact Admin."

    df = pd.read_csv(csv_path)
    # Normalize types
    df['Student_ID_str'] = df['Student_ID'].astype(str).str.strip()
    sid = str(student_id).strip()

    row = df[df['Student_ID_str'] == sid]
    if row.empty:
        return f"❌ Student ID {student_id} not found."

    row = row.iloc[0]

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Student Performance Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(50, 8, txt=f"Student ID: {row['Student_ID']}", ln=True)
    pdf.cell(50, 8, txt=f"Predicted Performance: {row['Predicted_Performance']}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, txt="Input Features:", ln=True)

    pdf.set_font("Arial", "", 11)
    for col in df.columns:
        if col in ['Student_ID', 'Student_ID_str', 'Predicted_Performance']:
            continue
        pdf.cell(0, 7, txt=f"{col}: {row[col]}", ln=True)

    out_path = f"dataset/student_{sid}_result.pdf"
    pdf.output(out_path)

    return send_file(out_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
