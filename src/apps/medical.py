import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

def resolve_paths():
    root = Path(__file__).resolve().parents[2]
    return {
        "root": root,
        "data_path": root / "resources" / "medical" / "disease_diagnosis.csv",
        "text_path": root / "resources" / "medical" / "symptoms_text_2.txt",
        "combined_models": root / "models" / "medical",
        "hybrid_model": root / "src" / "medical-nn" / "complete_hybrid_model.pth",
    }


@st.cache_data
def load_medical_dataset(data_path):
    df = pd.read_csv(data_path)
    df["Blood_Pressure_mmHg_lower"] = df["Blood_Pressure_mmHg"].apply(
        lambda x: int(x.split("/")[1]) if isinstance(x, str) and "/" in x else np.nan
    )
    df["Blood_Pressure_mmHg_upper"] = df["Blood_Pressure_mmHg"].apply(
        lambda x: int(x.split("/")[0]) if isinstance(x, str) and "/" in x else np.nan
    )
    df["Body_Temperature_C_scaled"] = (
        df["Body_Temperature_C"] - df["Body_Temperature_C"].min()
    ) / (df["Body_Temperature_C"].max() - df["Body_Temperature_C"].min())
    return df


@st.cache_resource
def load_combined_models(models_path):
    xgb_model = joblib.load(models_path / "xgb_model.pkl")
    log_model = joblib.load(models_path / "log_model.pkl")
    meta_model = joblib.load(models_path / "meta_model.pkl")
    return xgb_model, log_model, meta_model


class HybridMedicalModel(nn.Module):
    def __init__(self, text_dim, num_dim, hidden_dims_text, hidden_dims_num, dropout=0.3):
        super(HybridMedicalModel, self).__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dims_text[0]),
            nn.BatchNorm1d(hidden_dims_text[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims_text[0], hidden_dims_text[1]),
            nn.BatchNorm1d(hidden_dims_text[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.num_branch = nn.Sequential(
            nn.Linear(num_dim, hidden_dims_num[0]),
            nn.BatchNorm1d(hidden_dims_num[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims_num[0], hidden_dims_num[1]),
            nn.BatchNorm1d(hidden_dims_num[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims_text[1] + hidden_dims_num[1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, text_features, num_features):
        text_out = self.text_branch(text_features)
        num_out = self.num_branch(num_features)
        combined = torch.cat([text_out, num_out], dim=1)
        return self.classifier(combined)


@st.cache_resource
def load_hybrid_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    hidden_dims = config.get("hidden_dims", [128, 64])
    text_dim = config.get("text_dim")
    num_dim = config.get("num_dim")
    dropout = config.get("dropout", 0.3)

    model = HybridMedicalModel(
        text_dim=text_dim,
        num_dim=num_dim,
        hidden_dims_text=hidden_dims,
        hidden_dims_num=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = checkpoint.get("scaler")
    label_encoder = checkpoint.get("label_encoder")
    text_encoder = checkpoint.get("text_encoder")
    if text_encoder is None:
        text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    return model, scaler, label_encoder, text_encoder


@st.cache_resource
def load_glove_corpus():
    import gensim.downloader as api

    return api.load("glove-wiki-gigaword-100")


def tokenize_text(text):
    return re.findall(r"\b[a-zA-Z0-9']+\b", str(text).lower())


def vectorize_symptoms(text, corpus):
    tokens = tokenize_text(text)
    vectors = [corpus[word] for word in tokens if word in corpus]
    if len(vectors) == 0:
        return np.zeros(corpus.vector_size, dtype=float)
    return np.mean(vectors, axis=0)


@st.cache_data
def fit_label_encoders(df):
    from sklearn.preprocessing import LabelEncoder

    gender_encoder = LabelEncoder()
    gender_encoder.fit(df["Gender"].astype(str))

    treatment_encoder = LabelEncoder()
    treatment_encoder.fit(df["Treatment_Plan"].astype(str))

    return gender_encoder, treatment_encoder


def build_combined_features(user_inputs, gender_encoder, corpus, xgb_model, log_model, meta_model, treatment_encoder):
    text_vector = vectorize_symptoms(user_inputs["symptoms_text"], corpus)

    numeric_values = [
        user_inputs["Oxygen_Saturation_%"],
        user_inputs["Heart_Rate_bpm"],
        user_inputs["Body_Temperature_C_scaled"],
        user_inputs["Blood_Pressure_mmHg_upper"],
        user_inputs["Blood_Pressure_mmHg_lower"],
        user_inputs["Age"],
        gender_encoder.transform([user_inputs["Gender"]])[0],
    ]
    numeric_array = np.array([numeric_values], dtype=float)
    proba_num = xgb_model.predict_proba(numeric_array)
    proba_text = log_model.predict_proba(np.array([text_vector], dtype=float))
    meta_features = np.hstack([proba_num, proba_text])
    final_class = int(meta_model.predict(meta_features)[0])
    class_probs = dict(zip(treatment_encoder.classes_, meta_model.predict_proba(meta_features)[0]))
    return treatment_encoder.inverse_transform([final_class])[0], class_probs, numeric_array, text_vector


@st.cache_resource
def build_shap_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)


def shap_abnormality_report(xgb_model, feature_names, numeric_array, dataset_stats):
    explainer = build_shap_explainer(xgb_model)
    shap_values = explainer.shap_values(numeric_array)
    predicted_class = int(xgb_model.predict(numeric_array)[0])
    if isinstance(shap_values, list):
        shap_values = shap_values[predicted_class]
    shap_values = np.array(shap_values).reshape(-1)

    report = []
    abs_vals = np.abs(shap_values)
    threshold = max(np.percentile(abs_vals, 60), 0.01)
    for name, value, shap_value in zip(feature_names, numeric_array[0], shap_values):
        if name == "Gender_encoded":
            continue
        mean_value = dataset_stats[name]["mean"]
        std_value = dataset_stats[name]["std"]
        abnormal = abs(shap_value) >= threshold
        direction = "increases" if shap_value > 0 else "decreases"
        report.append(
            {
                "field": name,
                "value": float(value),
                "mean": float(mean_value),
                "shap_value": float(shap_value),
                "abnormal": abnormal,
                "direction": direction,
                "difference": float(value - mean_value),
            }
        )
    report.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return report


def load_dataset_stats(df):
    numeric_names = [
        "Oxygen_Saturation_%",
        "Heart_Rate_bpm",
        "Body_Temperature_C_scaled",
        "Blood_Pressure_mmHg_upper",
        "Blood_Pressure_mmHg_lower",
        "Age",
    ]
    stats = {}
    for name in numeric_names:
        stats[name] = {
            "mean": df[name].mean(),
            "std": df[name].std(),
            "min": df[name].min(),
            "max": df[name].max(),
        }
    return stats


def predict_hybrid(user_inputs, hybrid_model, scaler, label_encoder, text_encoder):
    raw_text = user_inputs["symptoms_text"]
    embedding = text_encoder.encode([raw_text], show_progress_bar=False)
    numeric_input = np.array([
        user_inputs["Oxygen_Saturation_%"],
        user_inputs["Heart_Rate_bpm"],
        user_inputs["Body_Temperature_C_scaled"],
        user_inputs["Blood_Pressure_mmHg_upper"],
        user_inputs["Blood_Pressure_mmHg_lower"],
        user_inputs["Age"],
        user_inputs["Gender_encoded"],
    ], dtype=float)
    numeric_scaled = scaler.transform([numeric_input])
    with torch.no_grad():
        text_tensor = torch.from_numpy(np.array(embedding, dtype=np.float32))
        if text_tensor.ndim == 1:
            text_tensor = text_tensor.reshape(1, -1)
        num_tensor = torch.from_numpy(numeric_scaled.astype(np.float32))
        output = hybrid_model(text_tensor, num_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probabilities))

    label = label_encoder.inverse_transform([prediction])[0]
    probs = dict(zip(label_encoder.classes_, probabilities.tolist()))
    return label, float(np.max(probabilities)), probs


def main():
    paths = resolve_paths()
    st.set_page_config(
        page_title="Medical Patient State Prediction",
        layout="wide",
        page_icon="🩺",
    )

    st.title("Medical Patient State Predictor")
    st.markdown(
        "Use the form below to enter patient vital data and symptom descriptions. "
        "Switch between the combined ML model and the hybrid neural network model to compare predictions. "
        "Missing numeric values are replaced with dataset averages."
    )

    df = load_medical_dataset(paths["data_path"])
    dataset_stats = load_dataset_stats(df)
    gender_options = sorted(df["Gender"].dropna().unique().astype(str).tolist())
    default_gender = df["Gender"].mode().iloc[0]
    default_temperature = df["Body_Temperature_C"].mean()
    temp_min = df["Body_Temperature_C"].min()
    temp_max = df["Body_Temperature_C"].max()

    model_choice = st.radio("Choose prediction model", ["Combined ML", "Hybrid NN"])

    with st.sidebar:
        st.header("Patient input")
        st.markdown("Leave any fields unchanged to use the dataset average value.")
        oxygen = st.number_input(
            "Oxygen Saturation (%)",
            min_value=50.0,
            max_value=100.0,
            value=float(df["Oxygen_Saturation_%"].mean()),
            step=0.1,
        )
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=30.0,
            max_value=200.0,
            value=float(df["Heart_Rate_bpm"].mean()),
            step=0.5,
        )
        body_temp = st.number_input(
            "Body Temperature (°C)",
            min_value=float(temp_min),
            max_value=float(temp_max),
            value=float(default_temperature),
            step=0.1,
        )
        bp_upper = st.number_input(
            "Blood Pressure Systolic (upper)",
            min_value=60,
            max_value=220,
            value=int(round(df["Blood_Pressure_mmHg_upper"].mean())),
            step=1,
        )
        bp_lower = st.number_input(
            "Blood Pressure Diastolic (lower)",
            min_value=30,
            max_value=140,
            value=int(round(df["Blood_Pressure_mmHg_lower"].mean())),
            step=1,
        )
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=int(round(df["Age"].mean())),
            step=1,
        )
        gender = st.selectbox("Gender", gender_options, index=gender_options.index(default_gender))
        symptoms_text = st.text_area(
            "Symptom description",
            value="Patient reports shortness of breath, mild cough, and fatigue.",
            height=180,
        )

    temp_scaled = (body_temp - temp_min) / (temp_max - temp_min) if temp_max != temp_min else 0.5
    user_inputs = {
        "Oxygen_Saturation_%": oxygen,
        "Heart_Rate_bpm": heart_rate,
        "Body_Temperature_C_scaled": temp_scaled,
        "Blood_Pressure_mmHg_upper": bp_upper,
        "Blood_Pressure_mmHg_lower": bp_lower,
        "Age": age,
        "Gender": gender,
        "symptoms_text": symptoms_text,
    }

    st.subheader("Current input values")
    col1, col2 = st.columns(2)
    with col1:
        st.write(
            "### Numeric vitals"
        )
        st.metric("Oxygen Saturation", f"{oxygen:.1f}%")
        st.metric("Heart Rate", f"{heart_rate:.0f} bpm")
        st.metric("Body Temperature", f"{body_temp:.1f} °C")
        st.metric("Blood Pressure", f"{bp_upper:.0f}/{bp_lower:.0f} mmHg")
        st.metric("Age", f"{age:.0f}")
        st.metric("Gender", gender)
    with col2:
        st.write("### Symptom description")
        st.write(symptoms_text or "No symptom description provided.")

    st.write("---")

    if model_choice == "Combined ML":
        with st.spinner("Loading combined models and evaluating..."):
            xgb_model, log_model, meta_model = load_combined_models(paths["combined_models"])
            gender_encoder, treatment_encoder = fit_label_encoders(df)
            label, class_probs, numeric_array, text_vector = build_combined_features(
                user_inputs,
                gender_encoder,
                load_glove_corpus(),
                xgb_model,
                log_model,
                meta_model,
                treatment_encoder,
            )

        st.header("Combined ML model prediction")
        st.success(f"Predicted treatment plan: {label}")
        st.write("**Class probabilities:**")
        st.write(class_probs)

        shap_report = shap_abnormality_report(
            xgb_model,
            [
                "Oxygen_Saturation_%",
                "Heart_Rate_bpm",
                "Body_Temperature_C_scaled",
                "Blood_Pressure_mmHg_upper",
                "Blood_Pressure_mmHg_lower",
                "Age",
                "Gender_encoded",
            ],
            numeric_array,
            dataset_stats,
        )

        st.subheader("Abnormal values by SHAP contribution")
        abnormal_df = pd.DataFrame(shap_report)
        abnormal_df["abnormal"] = abnormal_df["abnormal"].map({True: "Yes", False: "No"})
        st.dataframe(
            abnormal_df[["field", "value", "mean", "difference", "shap_value", "direction", "abnormal"]]
        )
        st.markdown(
            "Features marked as `Yes` are the fields that most strongly influence the treatment plan prediction according to the XGBoost model SHAP explanation."
        )
        st.bar_chart(
            pd.DataFrame(
                {
                    "SHAP contribution": [row["shap_value"] for row in shap_report],
                },
                index=[row["field"] for row in shap_report],
            )
        )
    else:
        with st.spinner("Loading hybrid neural network and evaluating..."):
            hybrid_model, scaler, label_encoder, text_encoder = load_hybrid_model(paths["hybrid_model"])
            gender_encoder, treatment_encoder = fit_label_encoders(df)
            user_inputs["Gender_encoded"] = int(gender_encoder.transform([gender])[0])
            label, confidence, probs = predict_hybrid(user_inputs, hybrid_model, scaler, label_encoder, text_encoder)

        st.header("Hybrid NN model prediction")
        st.success(f"Predicted severity: {label}")
        st.write("**Prediction probabilities:**")
        st.write(probs)
        st.info(
            "This hybrid model combines symptom text embeddings with vital signs. "
            "If you want numeric SHAP explanations, switch to the Combined ML model."
        )

    st.write("---")
    st.subheader("Dataset average values used when fields were left default")
    summary = pd.DataFrame(
        {
            "mean": [dataset_stats[k]["mean"] for k in [
                "Oxygen_Saturation_%",
                "Heart_Rate_bpm",
                "Body_Temperature_C_scaled",
                "Blood_Pressure_mmHg_upper",
                "Blood_Pressure_mmHg_lower",
                "Age",
            ]]
        },
        index=[
            "Oxygen Saturation (%)",
            "Heart Rate (bpm)",
            "Body Temperature (scaled)",
            "Systolic BP",
            "Diastolic BP",
            "Age",
        ],
    )
    st.table(summary)


if __name__ == "__main__":
    main()
