# ML Pro – Health Prediction Suite

## Overview

This repository contains a **Streamlit web application** that loads three pre‑trained scikit‑learn pipelines and provides an interactive UI for predicting:

- **Diabetes** risk (🩸)
- **Heart disease** risk (❤️)
- **Parkinson's disease** progression (🧠)

The models are stored as `.sav` files in the `models/` directory. The app dynamically builds the input form based on each model’s required features, shows a corresponding emoji header, and returns a friendly interpretation of the prediction.

## Folder structure
```
ML pro/
├── app.py                 # Streamlit application (single‑file implementation)
├── models/                # Saved pipelines (diabetes, heart disease, Parkinson's)
│   ├── diabetes_pipeline.sav
│   ├── heart_disease_pipeline.sav
│   └── parkinsons_pipeline.sav
└── README.md              # <-- you are reading this file
```

## Prerequisites
- **Windows** (the user environment)
- **Python 3.9+**
- Internet connection (only for the initial `pip install` step)

## Installation
Open a terminal (PowerShell or Command Prompt) and run:
```powershell
cd "C:\Users\omarh\Desktop\ML pro"
# (optional) create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell
# or venv\Scripts\activate.bat for cmd.exe
pip install --upgrade pip
pip install streamlit joblib pandas scikit-learn
```

## Running the app
```powershell
streamlit run app.py
```
A browser window will open at `http://localhost:8501`. You can then:
1. Choose a model from the dropdown.
2. Adjust the numeric inputs (plus/minus arrows step by 1).
3. If the model includes a **Sex** feature, a radio button appears.
4. Click **Predict** to see the result.

## Model feature details
The script that inspects the pipelines discovered the following feature names:
- **Diabetes**: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`
- **Heart disease**: `age, trestbps, chol, thalach, oldpeak, sex, cp, fbs, restecg, exang, slope, ca, thal`
- **Parkinson's**: `MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE`

These are automatically mapped to input fields by the app.

## Customisation
- **Styling** – The app uses a dark‑mode glass‑morphism theme. Feel free to edit the CSS block in `app.py`.
- **Additional models** – Drop a new `.sav` file into `models/` and add its feature list to the `MODEL_FEATURES` dictionary (or let the inspection script generate it).

## License
This project is provided under the MIT License. You are free to use, modify, and distribute it.

## Acknowledgements
- Streamlit – for rapid UI creation.
- scikit‑learn – for model pipelines.
- The original dataset authors for diabetes, heart disease, and Parkinson's disease.
