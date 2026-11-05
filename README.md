Smart Electricity Usage Analyzer & Prediction System

Project Overview

The Smart Electricity Usage Analyzer is a Python-based application designed to track, analyze, and estimate household electricity consumption.

It helps users understand their electricity usage, estimate bills, identify unusual consumption, and predict future electricity requirements.

Features

- Store meter readings using SQLite.
- Validate meter readings before saving.
- Calculate daily and monthly electricity consumption.
- Estimate electricity bills.
- Configurable CESC-style bill estimation.
- Detect unusually high electricity usage.
- Analyze monthly consumption trends.
- Compare monthly electricity usage.
- Generate electricity usage insights.
- Predict next month's electricity consumption.
- Evaluate prediction model performance when sufficient data is available.
- Daily and monthly usage alerts.
- Export reports in CSV and Excel formats.
- Interactive dashboard with charts.

Technologies Used

- Python
- Streamlit
- SQLite
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Openpyxl

Project Structure

Smart_Electricity_Usage_Analyzer/
│
├── app.py
├── database.py
├── analysis.py
├── prediction.py
├── data/
├── reports/
├── requirements.txt
└── README.md

Installation

1. Clone or download the project.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install the required packages:

pip install -r requirements.txt

Run the Application

streamlit run app.py

Prediction Information

The prediction model uses Linear Regression to estimate next month's electricity consumption based on historical monthly usage.

At least three months of data are required for the prediction chart, and four months for the model evaluation.

Disclaimer

Electricity bill calculations are estimates and may differ from the actual electricity provider's bill.

Project Purpose

This project demonstrates practical applications of Python programming, database management, data analysis, visualization, and basic machine learning.