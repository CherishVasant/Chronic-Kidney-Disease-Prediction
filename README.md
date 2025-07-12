# Chronic Kidney Disease Prediction

This project is an end-to-end machine learning application for predicting Chronic Kidney Disease (CKD) using patient medical data. It includes data preprocessing, model training, performance evaluation, and a web application built with Flask to serve predictions in a user-friendly format.

## Project Structure

Project Initialization and Planning Phase/
Data Collection and Preprocessing Phase/
Model Development Phase/
Model Optimization and Tuning Phase/
CompleteCode/
│
├── static/ # Static files (CSS, JavaScript)
├── templates/ # HTML templates
├── CKDPrediction.ipynb # Jupyter notebook with full ML pipeline
├── app.py # Flask web application
├── chronickidneydisease.csv # Dataset used for training
├── requirements.txt # Python dependencies


## Features

- Data loading, cleaning, and transformation
- Exploratory Data Analysis (EDA)
- Model training using scikit-learn
- Hyperparameter tuning and optimization
- Interactive web application for real-time CKD prediction
- Risk factor analysis based on user input


## Dataset

- Source: UCI Machine Learning Repository – Chronic Kidney Disease Dataset
- Features:

| No. | Feature Name | Description |
|-----|--------------|-------------|
| 1   | `age`        | Age of the patient (in years) |
| 2   | `bp`         | Blood pressure (in mm/Hg) |
| 3   | `bgr`        | Blood glucose random (in mg/dL) |
| 4   | `bu`         | Blood urea (in mg/dL) |
| 5   | `sc`         | Serum creatinine (in mg/dL) |
| 6   | `sod`        | Sodium (in mEq/L) |
| 7   | `pot`        | Potassium (in mEq/L) |
| 8   | `hemo`       | Hemoglobin (in g/dL) |
| 9   | `pcv`        | Packed cell volume |
| 10  | `wc`         | White blood cell count (cells/cumm) |
| 11  | `rc`         | Red blood cell count (millions/cmm) |
| 12  | `sg`         | Specific gravity (`1.005`, `1.010`, `1.015`, `1.020`, `1.025`) |
| 13  | `al`         | Albumin levels (scale 0–5) |
| 14  | `su`         | Sugar levels (scale 0–5) |
| 15  | `rbc`        | Red blood cells (`normal` = 0, `abnormal` = 1) |
| 16  | `pc`         | Pus cell (`normal` = 0, `abnormal` = 1) |
| 17  | `pcc`        | Pus cell clumps (`notpresent` = 0, `present` = 1) |
| 18  | `ba`         | Bacteria (`notpresent` = 0, `present` = 1) |
| 19  | `htn`        | Hypertension (`no` = 0, `yes` = 1) |
| 20  | `dm`         | Diabetes mellitus (`no` = 0, `yes` = 1) |
| 21  | `cad`        | Coronary artery disease (`no` = 0, `yes` = 1) |
| 22  | `appet`      | Appetite (`good` = 0, `poor` = 1) |
| 23  | `pe`         | Pedal edema (`no` = 0, `yes` = 1) |
| 24  | `ane`        | Anemia (`no` = 0, `yes` = 1) |


## Technologies Used

- Python 3.12
- NumPy
- Pandas
- scikit-learn
- Flask
- Jupyter Notebook
- HTML/CSS/JavaScript


## How to Run the Project

1. Clone the repository or download the project files.

2. Navigate to the `CompleteCode` directory.

3. Run the Flask app:
    python app.py

4. Open your web browser and go to:
    http://127.0.0.1:5000/

5. Enter medical values in the form and submit to receive a CKD prediction and risk factor analysis.

## Prediction Output

- Prediction: High CKD Risk / Low CKD Risk
- Confidence Score: e.g., 0.89
- Risk Factors: A list of engineered features based on input features

## Report

The full project report is available in the root directory as `Project Final Report.pdf`.
