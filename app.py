from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
)

# ---------------- CONSTANTS ----------------
DATA_PATH = Path(__file__).resolve().parent / "diabetes_dataset.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

REQUIRED_COLUMNS = FEATURES + ["Outcome"]

# ---------------- CUSTOM STYLE ----------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #2563eb;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🩺 Diabetes Prediction System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Health Risk Screening'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset columns missing: " + ", ".join(missing_columns)
        )

    data = data[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Outcome"])

    data = data[data["Outcome"].isin([0, 1])]
    data["Outcome"] = data["Outcome"].astype(int)

    if data.empty:
        raise ValueError("Dataset lo valid records levu.")

    return data


if not DATA_PATH.is_file():
    st.error(
        f"Dataset file dorakaledu: {DATA_PATH.name}. "
        "app.py pakkana diabetes_dataset.csv file undho check cheyyi."
    )
    st.stop()

try:
    df = load_dataset(DATA_PATH)
except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()

if df["Outcome"].nunique() != 2:
    st.error(
        "Outcome column lo 0 and 1 rendu classes undali. "
        "Dataset ni check cheyyi."
    )
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Predict Diabetes", "Dataset", "Model Performance"],
)

st.sidebar.info(
    "This application provides an educational risk estimate, "
    "not a medical diagnosis."
)

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data["Outcome"]

    def make_model():
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    metrics = None

    # Evaluate on unseen data when dataset size allows it.
    class_counts = y.value_counts()

    if len(data) >= 10 and class_counts.min() >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )

            evaluation_model = make_model()
            evaluation_model.fit(X_train, y_train)
            predictions = evaluation_model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "confusion_matrix": confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                ),
                "report": classification_report(
                    y_test,
                    predictions,
                    labels=[0, 1],
                    target_names=["No diabetes", "Diabetes"],
                    output_dict=True,
                    zero_division=0,
                ),
                "test_size": len(y_test),
            }
        except ValueError:
            metrics = None

    # Train final prediction model on all available rows.
    final_model = make_model()
    final_model.fit(X, y)

    return final_model, metrics


try:
    model, metrics = train_model(df)
except Exception as error:
    st.error(f"Model training error: {error}")
    st.stop()

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.subheader("📊 Project Dashboard")

    total_records = len(df)
    positive = int((df["Outcome"] == 1).sum())
    negative = int((df["Outcome"] == 0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", total_records)
    c2.metric("Outcome = 1", positive)
    c3.metric("Outcome = 0", negative)

    st.markdown("### Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Outcome Distribution")
    outcome_counts = (
        df["Outcome"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )
    st.bar_chart(
        pd.DataFrame(
            {
                "Records": outcome_counts.values
            },
            index=["No diabetes (0)", "Diabetes (1)"],
        )
    )

    if metrics is not None:
        st.metric(
            "Held-out Test Accuracy",
            f"{metrics['accuracy'] * 100:.1f}%",
        )
        st.caption(
            "Accuracy is estimated on a held-out portion of this dataset. "
            "Small datasets can produce unreliable results."
        )
    else:
        st.warning(
            "Dataset is too small for a reliable train/test evaluation. "
            "Add more valid records before interpreting model performance."
        )

# ---------------- PREDICTION PAGE ----------------
elif page == "Predict Diabetes":
    st.subheader("🧑‍⚕️ Enter Patient Details")

    st.write(
        "Enter the values below to generate a model-based risk estimate."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=30,
                value=1,
                step=1,
            )
            glucose = st.number_input(
                "Glucose",
                min_value=0.0,
                max_value=400.0,
                value=120.0,
            )
            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=0.0,
                max_value=250.0,
                value=70.0,
            )
            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=150.0,
                value=20.0,
            )

        with col2:
            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=1000.0,
                value=80.0,
            )
            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
            )
            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
            )
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30,
                step=1,
            )

        submitted = st.form_submit_button(
            "🔍 Predict Risk",
            use_container_width=True,
        )

    if submitted:
        patient = pd.DataFrame(
            [
                [
                    pregnancies,
                    glucose,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    bmi,
                    pedigree,
                    age,
                ]
            ],
            columns=FEATURES,
        )

        prediction = int(model.predict(patient)[0])
        probabilities = model.predict_proba(patient)[0]
        class_positions = list(model.named_steps["classifier"].classes_)
        positive_position = class_positions.index(1)
        risk_score = float(probabilities[positive_position]) * 100

        st.markdown("### Prediction Result")

        if prediction == 1:
            st.warning(
                "The model flags a possible elevated diabetes risk. "
                "This is not a diagnosis."
            )
        else:
            st.success(
                "The model did not flag elevated risk for this input. "
                "This does not rule out diabetes."
            )

        st.metric("Model Estimated Probability for Class 1", f"{risk_score:.1f}%")
        st.progress(int(round(risk_score)))

        st.caption(
            "This percentage is the model's estimated probability, "
            "not a clinically validated personal risk score."
        )

        st.info(
            "Consult a qualified healthcare professional for appropriate "
            "testing and interpretation. Do not make medical decisions "
            "based on this application alone."
        )

# ---------------- DATASET PAGE ----------------
elif page == "Dataset":
    st.subheader("📁 Dataset Explorer")

    st.write("Dataset shape:", df.shape)
    st.write("Column names:", list(df.columns))

    st.markdown("### Complete Dataset")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Missing Values")
    missing_values = df.isna().sum().rename("Missing Values")
    st.dataframe(missing_values, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Cleaned Dataset",
        data=csv_data,
        file_name="cleaned_diabetes_dataset.csv",
        mime="text/csv",
    )

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.subheader("📈 Model Performance")

    if metrics is None:
        st.warning(
            "Not enough examples per class to create a useful test set. "
            "Add more data to evaluate the model."
        )
    else:
        st.metric(
            "Test Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%",
        )

        st.caption(
            f"Evaluation used {metrics['test_size']} held-out records. "
            "Results depend on dataset size and quality."
        )

        st.markdown("### Confusion Matrix")
        cm = metrics["confusion_matrix"]

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            ),
            use_container_width=True,
        )

        st.markdown("### Classification Report")
        report = pd.DataFrame(metrics["report"]).T
        st.dataframe(report.round(3), use_container_width=True)

    st.markdown("### Input Features")
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": FEATURES,
                "Description": [
                    "Number of pregnancies",
                    "Glucose measurement",
                    "Blood pressure measurement",
                    "Skin thickness measurement",
                    "Insulin measurement",
                    "Body mass index",
                    "Diabetes pedigree function",
                    "Age in years",
                ],
            }
        ),
        use_container_width=True,
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Diabetes Prediction System | Educational ML project | "
    "Not a substitute for professional medical advice."
)