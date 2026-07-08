import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import os
import pandas as pd


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_FEATURES = {
    "diabetes_pipeline.sav": {
        "feature_names": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
    },
    "heart_disease_pipeline.sav": {
        "feature_names": [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
    },
    "parkinsons_pipeline.sav": {
        "feature_names": [
            "MDVP:Fo(Hz)",
            "MDVP:Fhi(Hz)",
            "MDVP:Flo(Hz)",
            "MDVP:Jitter(%)",
            "MDVP:Jitter(Abs)",
            "MDVP:RAP",
            "MDVP:PPQ",
            "Jitter:DDP",
            "MDVP:Shimmer",
            "MDVP:Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "MDVP:APQ",
            "Shimmer:DDA",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "spread1",
            "spread2",
            "D2",
            "PPE",
        ]
    },
}

FEATURE_RANGES = {
    "Pregnancies": {"min": 0, "max": 17, "default": 0, "step": 1},
    "Glucose": {"min": 0, "max": 250, "default": 100, "step": 1},
    "BloodPressure": {"min": 0, "max": 200, "default": 70, "step": 1},
    "SkinThickness": {"min": 0, "max": 100, "default": 20, "step": 1},
    "Insulin": {"min": 0, "max": 900, "default": 80, "step": 1},
    "BMI": {"min": 0.0, "max": 70.0, "default": 25.0, "step": 0.1},
    "DiabetesPedigreeFunction": {"min": 0.0, "max": 3.0, "default": 0.5, "step": 0.01},
    "Age": {"min": 0, "max": 120, "default": 30, "step": 1},

    "age": {"min": 0, "max": 120, "default": 40, "step": 1},
    "trestbps": {"min": 60, "max": 220, "default": 120, "step": 1},
    "chol": {"min": 100, "max": 600, "default": 200, "step": 1},
    "thalach": {"min": 60, "max": 220, "default": 150, "step": 1},
    "oldpeak": {"min": 0.0, "max": 7.0, "default": 1.0, "step": 0.1},
    "cp": {"min": 0, "max": 3, "default": 0, "step": 1},
    "fbs": {"min": 0, "max": 1, "default": 0, "step": 1},
    "restecg": {"min": 0, "max": 2, "default": 0, "step": 1},
    "exang": {"min": 0, "max": 1, "default": 0, "step": 1},
    "slope": {"min": 0, "max": 2, "default": 1, "step": 1},
    "ca": {"min": 0, "max": 4, "default": 0, "step": 1},
    "thal": {"min": 0, "max": 3, "default": 2, "step": 1},

    "MDVP:Fo(Hz)": {"min": 80.0, "max": 260.0, "default": 150.0, "step": 0.1},
    "MDVP:Fhi(Hz)": {"min": 100.0, "max": 600.0, "default": 200.0, "step": 0.1},
    "MDVP:Flo(Hz)": {"min": 60.0, "max": 240.0, "default": 100.0, "step": 0.1},
    "MDVP:Jitter(%)": {"min": 0.0, "max": 0.1, "default": 0.005, "step": 0.001},
    "MDVP:Jitter(Abs)": {"min": 0.0, "max": 0.001, "default": 0.00005, "step": 0.00001},
    "MDVP:RAP": {"min": 0.0, "max": 0.05, "default": 0.003, "step": 0.0001},
    "MDVP:PPQ": {"min": 0.0, "max": 0.05, "default": 0.003, "step": 0.0001},
    "Jitter:DDP": {"min": 0.0, "max": 0.15, "default": 0.01, "step": 0.0001},
    "MDVP:Shimmer": {"min": 0.0, "max": 0.2, "default": 0.03, "step": 0.001},
    "MDVP:Shimmer(dB)": {"min": 0.0, "max": 2.0, "default": 0.3, "step": 0.01},
    "Shimmer:APQ3": {"min": 0.0, "max": 0.1, "default": 0.015, "step": 0.001},
    "Shimmer:APQ5": {"min": 0.0, "max": 0.1, "default": 0.02, "step": 0.001},
    "MDVP:APQ": {"min": 0.0, "max": 0.15, "default": 0.025, "step": 0.001},
    "Shimmer:DDA": {"min": 0.0, "max": 0.3, "default": 0.045, "step": 0.001},
    "NHR": {"min": 0.0, "max": 0.5, "default": 0.02, "step": 0.001},
    "HNR": {"min": 0.0, "max": 40.0, "default": 20.0, "step": 0.1},
    "RPDE": {"min": 0.0, "max": 1.0, "default": 0.5, "step": 0.001},
    "DFA": {"min": 0.0, "max": 1.0, "default": 0.7, "step": 0.001},
    "spread1": {"min": -10.0, "max": 0.0, "default": -5.0, "step": 0.01},
    "spread2": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.001},
    "D2": {"min": 0.0, "max": 5.0, "default": 2.0, "step": 0.01},
    "PPE": {"min": 0.0, "max": 1.0, "default": 0.2, "step": 0.001},
}


@st.cache_resource
def load_models():
    models = {}
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.sav'):
            path = os.path.join(MODELS_DIR, filename)
            try:
                models[filename] = joblib.load(path)
            except Exception as e:
                st.error(f"Failed to load {filename}: {e}")
    return models


models = load_models()

st.set_page_config(page_title="Multiple Disease Prediction System", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background-color: #0b0b16; color: #eaeaf0;}

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14142b 0%, #0b0b16 100%);
        min-height: 100vh;
        border-right: 1px solid #26264a;
    }
    section[data-testid="stSidebar"] > div {
        height: 100vh;
    }

    h1, h2, h3, h4, .stMarkdown {color: #f2f2f7;}

    .stButton > button {
        background: linear-gradient(135deg, #7c4dff 0%, #536dfe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 14px rgba(124, 77, 255, 0.4);
    }

    div[data-testid="stSlider"] * {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {
        background: #33334d !important;
    }

    div[data-testid="stSlider"] div[role="slider"] {
        background: #ff4b4b !important;
        border: none !important;
    }

    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] p {
        color: #c9c9dd !important;
    }

    div[data-testid="stRadio"] label {
        color: #c9c9dd !important;
    }

    div[role="radiogroup"] label {
        background: transparent !important;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        margin-right: 0.5rem;
        border: 1px solid transparent;
    }

    .stAlert {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    model_option = option_menu(
        menu_title="📋 Multiple Disease\nPrediction System",
        options=["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        icons=["activity", "heart-pulse", "person-wheelchair"],
        menu_icon="clipboard2-pulse",
        default_index=0,
        styles={
            "container": {
                "padding": "20px 15px",
                "background-color": "transparent",
                "height": "100vh",
            },
            "icon": {"color": "#c9c9dd", "font-size": "18px"},
            "menu-title": {
                "color": "#ffffff",
                "font-size": "19px",
                "font-weight": "700",
                "margin-bottom": "10px",
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "8px 0",
                "color": "#c9c9dd",
                "border-radius": "10px",
                "padding": "14px 16px",
                "--hover-color": "#1f1f3d",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #7c4dff 0%, #536dfe 100%)",
                "color": "white",
                "font-weight": "700",
                "box-shadow": "0 4px 14px rgba(124, 77, 255, 0.35)",
            },
        },
    )

menu_to_file = {
    "Diabetes Prediction": "diabetes_pipeline.sav",
    "Heart Disease Prediction": "heart_disease_pipeline.sav",
    "Parkinsons Prediction": "parkinsons_pipeline.sav",
}
model_file = menu_to_file[model_option]
feature_names = MODEL_FEATURES[model_file]["feature_names"]

model_info = {
    "diabetes_pipeline.sav": {"emoji": "🩸", "name": "Diabetes"},
    "heart_disease_pipeline.sav": {"emoji": "❤️", "name": "Heart Disease"},
    "parkinsons_pipeline.sav": {"emoji": "🧠", "name": "Parkinson's"},
}
info = model_info.get(model_file, {"emoji": "❓", "name": "Model"})
st.header(f"{info['emoji']} {info['name']} Prediction")
st.subheader(f"Input features for **{model_file.replace('_pipeline.sav','').replace('_',' ').title()}**")

sex_key = next((f for f in feature_names if f.lower() == "sex"), None)
sex_choice = st.radio("Sex", ["Male", "Female"], horizontal=True)
sex_val = 1 if sex_choice == "Male" else 0

inputs = {}
if sex_key:
    inputs[sex_key] = sex_val

cols = st.columns(2)
remaining_features = [f for f in feature_names if f != sex_key]
for i, feature in enumerate(remaining_features):
    col = cols[i % 2]
    r = FEATURE_RANGES.get(feature)
    with col:
        if r:
            val = st.slider(
                feature,
                min_value=r["min"],
                max_value=r["max"],
                value=r["default"],
                step=r["step"],
            )
        else:
            val = st.number_input(feature, min_value=0, value=0, step=1, format="%d")
    inputs[feature] = val

input_df = pd.DataFrame([inputs])[feature_names]

if st.button("Predict"):
    if model_file not in models:
        st.error("Model not loaded correctly.")
    else:
        try:
            model_obj = models[model_file]
            pred = model_obj.predict(input_df)[0]

            proba = None
            if hasattr(model_obj, "predict_proba"):
                try:
                    proba = model_obj.predict_proba(input_df)[0]
                except Exception:
                    proba = None

            is_risky = False
            if "diabetes" in model_file.lower():
                result = "Positive (Diabetes)" if pred == 1 else "Negative (No Diabetes)"
                is_risky = pred == 1
            elif "heart" in model_file.lower():
                result = "High risk of heart disease" if pred == 1 else "Low risk of heart disease"
                is_risky = pred == 1
            elif "parkinsons" in model_file.lower():
                result = f"Predicted disease progression score: {pred:.2f}"
                is_risky = pred == 1
            else:
                result = f"Prediction: {pred}"

            if is_risky:
                st.error(f"**Result:** {result}")
            else:
                st.success(f"**Result:** {result}")

            if is_risky and proba is not None:
                risk_percent = proba[1] * 100
                st.error(f"⚠️ Risk probability: **{risk_percent:.1f}%**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("Made with ❤️ using Streamlit – a sleek, dark‑mode UI for quick health insights.")