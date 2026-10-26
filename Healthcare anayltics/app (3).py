
# ==============================================================
# HEALTHPULSE ANALYTICS
# Healthcare Data Analytics & Cardiovascular Risk Prediction
# Input: healthcare_data.csv (same directory as app.py)
# ==============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- CONFIG -----------------------------

st.set_page_config(
    page_title="HealthPulse | Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "healthcare_data.csv"
TARGET = "synthetic_cardiovascular_risk_label"

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "glucose_mg_dl",
    "cholesterol_mg_dl",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
]

# ----------------------------- DESIGN -----------------------------

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"] {
        background: #f4f7fc !important;
        color: #18243a !important;
    }

    [data-testid="stHeader"] {
        border-bottom: 1px solid #e0e7f1;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background: #ffffff !important;
        border-right: 1px solid #dce5f0;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp li, .stApp label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #18243a;
    }

    .stApp h1 {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.7px;
    }

    .stApp h2, .stApp h3 {
        font-weight: 720;
    }

    .hp-subtitle {
        color: #52627a !important;
        font-size: 1rem;
        margin-top: -8px;
        margin-bottom: 22px;
    }

    .hp-brand {
        font-size: 1.45rem;
        font-weight: 800;
        color: #204fb5 !important;
    }

    .hp-card {
        background: #ffffff;
        border: 1px solid #dce5f0;
        border-radius: 15px;
        padding: 17px 19px;
        margin-bottom: 12px;
        box-shadow: 0 3px 10px rgba(28, 48, 82, 0.04);
    }

    .hp-card-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #61718a !important;
        margin-bottom: 8px;
    }

    .hp-card-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #18243a !important;
    }

    .hp-card-note {
        color: #52627a !important;
        font-size: 0.8rem;
        margin-top: 5px;
    }

    .hp-blue {
        border-top: 4px solid #3867d6;
    }

    .hp-teal {
        border-top: 4px solid #078a83;
    }

    .hp-orange {
        border-top: 4px solid #e59a30;
    }

    .hp-red {
        border-top: 4px solid #d84d61;
    }

    .hp-green {
        border-top: 4px solid #29956f;
    }

    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #dce5f0;
        border-radius: 13px;
        padding: 15px 17px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #53637b !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #18243a !important;
        font-weight: 750 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #087f68 !important;
    }

    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e0e7f0;
        border-radius: 13px;
        padding: 5px;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #ffffff !important;
        border: 1px solid #dce5f0;
        border-radius: 10px;
    }

    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="textarea"] > div {
        background: #ffffff !important;
        border-color: #cbd5e1 !important;
    }

    .stApp input,
    .stApp textarea {
        background: #ffffff !important;
        color: #18243a !important;
        -webkit-text-fill-color: #18243a !important;
    }

    [data-baseweb="popover"],
    [role="listbox"] {
        background: #ffffff !important;
        color: #18243a !important;
    }

    [role="option"] {
        color: #18243a !important;
    }

    .stApp .stButton > button,
    .stApp .stDownloadButton > button,
    .stApp [data-testid="stFormSubmitButton"] > button {
        background: #2457c5 !important;
        color: #ffffff !important;
        border: 1px solid #2457c5 !important;
        border-radius: 9px;
        font-weight: 650;
        min-height: 40px;
    }

    .stApp .stButton > button *,
    .stApp .stDownloadButton > button * {
        color: #ffffff !important;
    }

    .stApp .stButton > button:hover,
    .stApp .stDownloadButton > button:hover {
        background: #1946a6 !important;
    }

    .hp-footer {
        text-align: center;
        color: #66758c !important;
        font-size: 0.82rem;
        padding: 14px;
    }

    hr {
        border-color: #dce5f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- DATA -----------------------------

@st.cache_data(show_spinner="Loading healthcare data...")
def load_dataset(path, modified_time):
    data = pd.read_csv(path)
    data.columns = [str(c).strip() for c in data.columns]

    for col in NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "patient_id" not in data.columns:
        data.insert(0, "patient_id", range(1, len(data) + 1))

    if TARGET in data.columns:
        data[TARGET] = (
            data[TARGET].astype("string").str.strip().str.title()
        )

    return data


if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH}\n\n"
        "Place healthcare_data.csv in the same folder as app.py."
    )
    st.stop()

try:
    df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to read the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The healthcare dataset contains no records.")
    st.stop()


# ----------------------------- HELPERS -----------------------------

def num(data, column):
    if column not in data.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(data[column], errors="coerce").dropna()


def average(data, column):
    values = num(data, column)
    return float(values.mean()) if not values.empty else None


def fmt(value, decimals=1, suffix=""):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def flag_mask(series):
    return series.astype("string").str.strip().str.lower().isin(
        ["yes", "true", "1", "positive", "present"]
    )


def risk_labels(data):
    if TARGET not in data.columns:
        return pd.Series("Unknown", index=data.index)
    return (
        data[TARGET].astype("string").str.strip().str.title()
        .fillna("Unknown")
    )


def plot_style(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", color="#18243a", size=12),
        title_font=dict(size=16, color="#18243a"),
        margin=dict(l=20, r=20, t=55, b=25),
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_color="#18243a",
            bordercolor="#cbd5e1",
        ),
        legend=dict(
            font=dict(color="#42516a"),
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )
    fig.update_xaxes(
        tickfont=dict(color="#42516a"),
        title_font=dict(color="#42516a"),
        gridcolor="#e8edf5",
        zerolinecolor="#dce5f0",
    )
    fig.update_yaxes(
        tickfont=dict(color="#42516a"),
        title_font=dict(color="#42516a"),
        gridcolor="#e8edf5",
        zerolinecolor="#dce5f0",
    )
    return fig


def chart(fig, key, height=360):
    st.plotly_chart(
        plot_style(fig, height),
        use_container_width=True,
        key=key,
    )


def kpi_card(label, value, note="", color="blue"):
    st.markdown(
        f"""
        <div class="hp-card hp-{color}">
            <div class="hp-card-label">{label}</div>
            <div class="hp-card-value">{value}</div>
            <div class="hp-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


# ----------------------------- MODEL -----------------------------

@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model(path, modified_time):
    data = pd.read_csv(path)
    data.columns = [str(c).strip() for c in data.columns]

    missing = [c for c in NUMERIC_FEATURES if c not in data.columns]
    if TARGET not in data.columns:
        return {"error": f"Target column '{TARGET}' is missing."}

    if missing:
        return {"error": "Missing model features: " + ", ".join(missing)}

    model_data = data[NUMERIC_FEATURES + [TARGET]].copy()

    for col in NUMERIC_FEATURES:
        model_data[col] = pd.to_numeric(
            model_data[col], errors="coerce"
        )

    model_data[TARGET] = (
        model_data[TARGET].astype("string").str.strip().str.title()
    )

    model_data = model_data.dropna(subset=[TARGET])
    model_data = model_data[
        model_data[TARGET].isin(["High", "Low"])
    ]

    if len(model_data) < 10:
        return {
            "error": "At least 10 rows with High/Low target labels are needed."
        }

    if model_data[TARGET].nunique() != 2:
        return {"error": "Training data must contain both High and Low labels."}

    counts = model_data[TARGET].value_counts()
    if counts.min() < 2:
        return {
            "error": "Each risk class needs at least two records for evaluation."
        }

    X = model_data[NUMERIC_FEATURES]
    y = model_data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        labels=["Low", "High"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["Low", "High"],
    )

    classifier = model.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "Feature": NUMERIC_FEATURES,
            "Importance": classifier.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label="High", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, pos_label="High", zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, pos_label="High", zero_division=0
        ),
        "report": report,
        "matrix": matrix,
        "importance": importance,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_rows": len(model_data),
        "class_counts": counts.to_dict(),
        "error": None,
    }


model_result = train_model(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime,
)


# ----------------------------- SIDEBAR -----------------------------

with st.sidebar:
    st.markdown(
        '<div class="hp-brand">🏥 HealthPulse</div>',
        unsafe_allow_html=True,
    )
    st.caption("Healthcare Intelligence Platform")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Patient Explorer",
            "Risk Analysis",
            "Clinical Analytics",
            "Predict Cardiovascular Risk",
            "Model Performance",
            "Data Quality & Export",
        ],
    )

    st.divider()
    st.markdown("### Data filters")

    filtered = df.copy()

    if "gender" in filtered.columns:
        gender_values = sorted(
            filtered["gender"].dropna().astype(str).unique().tolist()
        )
        selected_genders = st.multiselect(
            "Gender",
            gender_values,
            default=gender_values,
        )
        filtered = filtered[
            filtered["gender"].astype(str).isin(selected_genders)
        ]

    if "age" in filtered.columns:
        age_values = num(filtered, "age")
        if not age_values.empty and age_values.min() < age_values.max():
            age_range = st.slider(
                "Age range",
                int(age_values.min()),
                int(age_values.max()),
                (int(age_values.min()), int(age_values.max())),
            )
            filtered = filtered[
                pd.to_numeric(filtered["age"], errors="coerce")
                .between(age_range[0], age_range[1])
            ]

    if TARGET in filtered.columns:
        risk_values = sorted(
            filtered[TARGET].dropna().astype(str).unique().tolist()
        )
        selected_risks = st.multiselect(
            "Risk category",
            risk_values,
            default=risk_values,
        )
        filtered = filtered[
            filtered[TARGET].astype(str).isin(selected_risks)
        ]

    st.divider()
    st.caption(f"Filtered records: {len(filtered):,}")
    st.caption(f"Total records: {len(df):,}")


# ----------------------------- SHARED DATA -----------------------------

view = filtered.copy()
view["_risk"] = risk_labels(view)

total = len(view)
avg_age = average(view, "age")
avg_bmi = average(view, "bmi")
avg_glucose = average(view, "glucose_mg_dl")
avg_sbp = average(view, "systolic_bp")
avg_chol = average(view, "cholesterol_mg_dl")

high_count = int((view["_risk"] == "High").sum())
low_count = int((view["_risk"] == "Low").sum())
high_pct = high_count / total * 100 if total else 0

diabetes_count = (
    int(flag_mask(view["diabetes_indicator"]).sum())
    if "diabetes_indicator" in view.columns else 0
)

hypertension_count = (
    int(flag_mask(view["hypertension"]).sum())
    if "hypertension" in view.columns else 0
)


# ==============================================================
# 1. EXECUTIVE OVERVIEW
# ==============================================================

if page == "Executive Overview":
    st.title("Healthcare Intelligence Dashboard")
    st.markdown(
        '<div class="hp-subtitle">'
        "Patient demographics, clinical indicators, cardiovascular risk "
        "patterns and machine-learning insights."
        "</div>",
        unsafe_allow_html=True,
    )

    # Eight dashboard KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Patients", f"{total:,}", "Filtered dataset", "blue")
    with k2:
        kpi_card("Average Age", fmt(avg_age, 1), "Years", "teal")
    with k3:
        kpi_card("Average BMI", fmt(avg_bmi, 1), "Body mass index", "orange")
    with k4:
        kpi_card("High Risk Labels", f"{high_count:,}", f"{high_pct:.1f}% of records", "red")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        kpi_card("Diabetes Indicators", f"{diabetes_count:,}", "Positive records", "orange")
    with k6:
        kpi_card("Hypertension", f"{hypertension_count:,}", "Positive records", "red")
    with k7:
        kpi_card("Average Glucose", fmt(avg_glucose, 1), "mg/dL", "teal")
    with k8:
        kpi_card("Average Systolic BP", fmt(avg_sbp, 1), "mmHg", "blue")

    st.divider()

    left, right = st.columns(2)

    with left:
        section("Patient Age Distribution")
        ages = num(view, "age")
        if not ages.empty:
            chart(
                px.histogram(
                    pd.DataFrame({"Age": ages}),
                    x="Age",
                    nbins=25,
                    color_discrete_sequence=["#3867d6"],
                ),
                "overview_age",
            )
        else:
            st.info("Age data is unavailable.")

    with right:
        section("Cardiovascular Risk Distribution")
        counts = (
            view["_risk"].value_counts()
            .rename_axis("Risk")
            .reset_index(name="Patients")
        )
        if not counts.empty:
            chart(
                px.pie(
                    counts,
                    names="Risk",
                    values="Patients",
                    hole=0.55,
                    color="Risk",
                    color_discrete_map={
                        "High": "#d84d61",
                        "Low": "#168a73",
                        "Medium": "#e59a30",
                        "Unknown": "#8792a2",
                    },
                ),
                "overview_risk",
            )
        else:
            st.info("No risk labels available.")

    left, right = st.columns(2)

    with left:
        section("BMI Category Analysis")
        bmis = num(view, "bmi")
        if not bmis.empty:
            bmi_data = pd.DataFrame({"BMI": bmis})
            bmi_data["Category"] = pd.cut(
                bmi_data["BMI"],
                [-np.inf, 18.5, 25, 30, np.inf],
                labels=["Underweight", "Normal", "Overweight", "Obesity"],
            )
            bmi_counts = (
                bmi_data["Category"].value_counts()
                .rename_axis("Category")
                .reset_index(name="Patients")
            )
            chart(
                px.bar(
                    bmi_counts,
                    x="Category",
                    y="Patients",
                    color="Category",
                    color_discrete_sequence=[
                        "#75a4e8", "#168a73", "#e59a30", "#d84d61"
                    ],
                ),
                "overview_bmi",
            )
        else:
            st.info("BMI data is unavailable.")

    with right:
        section("Clinical Indicator Averages")
        indicators = [
            ("Glucose", "glucose_mg_dl"),
            ("Cholesterol", "cholesterol_mg_dl"),
            ("Systolic BP", "systolic_bp"),
            ("Diastolic BP", "diastolic_bp"),
            ("Heart Rate", "heart_rate"),
        ]
        indicator_data = [
            {"Indicator": label, "Average": average(view, col)}
            for label, col in indicators
            if average(view, col) is not None
        ]
        if indicator_data:
            chart(
                px.bar(
                    pd.DataFrame(indicator_data),
                    x="Indicator",
                    y="Average",
                    color="Indicator",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                ),
                "overview_clinical",
            )
        else:
            st.info("Clinical measurements are unavailable.")

    st.divider()
    left, right = st.columns(2)

    with left:
        section("Glucose vs BMI")
        if {"glucose_mg_dl", "bmi"}.issubset(view.columns):
            scatter = view[["glucose_mg_dl", "bmi", "_risk"]].copy()
            scatter = scatter.dropna(subset=["glucose_mg_dl", "bmi"])
            if not scatter.empty:
                chart(
                    px.scatter(
                        scatter,
                        x="bmi",
                        y="glucose_mg_dl",
                        color="_risk",
                        opacity=0.7,
                        color_discrete_map={
                            "High": "#d84d61",
                            "Low": "#168a73",
                            "Medium": "#e59a30",
                            "Unknown": "#8792a2",
                        },
                        labels={
                            "bmi": "BMI",
                            "glucose_mg_dl": "Glucose (mg/dL)",
                            "_risk": "Risk label",
                        },
                    ),
                    "overview_scatter",
                )
            else:
                st.info("No complete BMI and glucose records.")
        else:
            st.info("BMI and glucose columns are required.")

    with right:
        section("Systolic Blood Pressure")
        bp = num(view, "systolic_bp")
        if not bp.empty:
            chart(
                px.histogram(
                    pd.DataFrame({"Systolic BP": bp}),
                    x="Systolic BP",
                    nbins=25,
                    color_discrete_sequence=["#087f8c"],
                ),
                "overview_bp",
            )
        else:
            st.info("Systolic blood pressure is unavailable.")

    if model_result.get("error"):
        st.caption("Model status: " + model_result["error"])
    else:
        st.caption(
            f"Random Forest trained on {model_result['train_rows']:,} "
            f"records; held-out test accuracy: "
            f"{model_result['accuracy']:.1%}."
        )


# ==============================================================
# 2. PATIENT EXPLORER
# ==============================================================

elif page == "Patient Explorer":
    st.title("Patient Explorer")
    st.markdown(
        '<div class="hp-subtitle">'
        "Search, inspect, filter and export individual patient records."
        "</div>",
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Search patient ID",
        placeholder="Type a patient ID or part of one",
    )

    patient_view = view.copy()

    if search.strip():
        patient_view = patient_view[
            patient_view["patient_id"].astype(str).str.contains(
                search.strip(), case=False, na=False
            )
        ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Matching patients", f"{len(patient_view):,}")
    c2.metric("Columns displayed", f"{len(df.columns):,}")
    c3.metric("High-risk labels", f"{int((patient_view['_risk'] == 'High').sum()):,}")

    display_cols = [c for c in df.columns if c in patient_view.columns]
    st.dataframe(
        patient_view[display_cols],
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    st.download_button(
        "Download filtered patient records",
        patient_view[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="healthcare_filtered_patients.csv",
        mime="text/csv",
    )

    st.divider()
    section("Individual Patient Details")

    if not patient_view.empty:
        ids = patient_view["patient_id"].astype(str).tolist()
        chosen_id = st.selectbox("Select patient", ids)
        record = patient_view[
            patient_view["patient_id"].astype(str) == chosen_id
        ].iloc[0]

        detail_cols = st.columns(3)
        for i, column in enumerate(display_cols):
            with detail_cols[i % 3]:
                value = record[column]
                kpi_card(
                    column.replace("_", " ").title(),
                    str(value) if pd.notna(value) else "N/A",
                    "Patient record",
                    ["blue", "teal", "orange"][i % 3],
                )
    else:
        st.info("No patient records match your filters.")


# ==============================================================
# 3. RISK ANALYSIS
# ==============================================================

elif page == "Risk Analysis":
    st.title("Cardiovascular Risk Analysis")
    st.markdown(
        '<div class="hp-subtitle">'
        "Explore risk labels across age, gender and clinical measurements."
        "</div>",
        unsafe_allow_html=True,
    )

    if TARGET not in df.columns:
        st.error(f"The target column '{TARGET}' is missing from the dataset.")
        st.stop()

    risk_counts = (
        view["_risk"].value_counts()
        .rename_axis("Risk")
        .reset_index(name="Patients")
    )

    left, right = st.columns(2)

    with left:
        section("Risk Category Counts")
        chart(
            px.bar(
                risk_counts,
                x="Risk",
                y="Patients",
                color="Risk",
                color_discrete_map={
                    "High": "#d84d61",
                    "Low": "#168a73",
                    "Medium": "#e59a30",
                    "Unknown": "#8792a2",
                },
            ),
            "risk_counts",
        )

    with right:
        section("Risk Category Percentages")
        if not risk_counts.empty:
            chart(
                px.pie(
                    risk_counts,
                    names="Risk",
                    values="Patients",
                    hole=0.5,
                    color="Risk",
                    color_discrete_map={
                        "High": "#d84d61",
                        "Low": "#168a73",
                        "Medium": "#e59a30",
                        "Unknown": "#8792a2",
                    },
                ),
                "risk_pie",
            )

    left, right = st.columns(2)

    with left:
        section("Risk Labels by Age Group")
        if "age" in view.columns:
            age_data = view[["age", "_risk"]].copy()
            age_data["age"] = pd.to_numeric(age_data["age"], errors="coerce")
            age_data = age_data.dropna(subset=["age"])
            if not age_data.empty:
                age_data["Age Group"] = pd.cut(
                    age_data["age"],
                    [-np.inf, 30, 45, 60, 75, np.inf],
                    labels=["Under 30", "30–44", "45–59", "60–74", "75+"],
                    right=False,
                )
                grouped = (
                    age_data.groupby(
                        ["Age Group", "_risk"], observed=False
                    ).size().reset_index(name="Patients")
                )
                chart(
                    px.bar(
                        grouped,
                        x="Age Group",
                        y="Patients",
                        color="_risk",
                        barmode="group",
                        labels={"_risk": "Risk label"},
                        color_discrete_map={
                            "High": "#d84d61",
                            "Low": "#168a73",
                            "Medium": "#e59a30",
                            "Unknown": "#8792a2",
                        },
                    ),
                    "risk_by_age",
                )

    with right:
        section("Risk Labels by Gender")
        if "gender" in view.columns:
            gender_risk = (
                view.groupby(["gender", "_risk"], dropna=False)
                .size().reset_index(name="Patients")
            )
            chart(
                px.bar(
                    gender_risk,
                    x="gender",
                    y="Patients",
                    color="_risk",
                    barmode="group",
                    labels={"gender": "Gender", "_risk": "Risk label"},
                    color_discrete_map={
                        "High": "#d84d61",
                        "Low": "#168a73",
                        "Medium": "#e59a30",
                        "Unknown": "#8792a2",
                    },
                ),
                "risk_by_gender",
            )
        else:
            st.info("Gender column is unavailable.")

    st.divider()
    section("Clinical Measurements by Risk Label")

    measurements = [
        c for c in NUMERIC_FEATURES if c in view.columns
    ]

    if measurements:
        selected = st.selectbox(
            "Clinical measurement",
            measurements,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        box_data = view[[selected, "_risk"]].copy()
        box_data[selected] = pd.to_numeric(
            box_data[selected], errors="coerce"
        )
        box_data = box_data.dropna(subset=[selected])

        if not box_data.empty:
            chart(
                px.box(
                    box_data,
                    x="_risk",
                    y=selected,
                    color="_risk",
                    points="outliers",
                    labels={
                        "_risk": "Risk label",
                        selected: selected.replace("_", " ").title(),
                    },
                    color_discrete_map={
                        "High": "#d84d61",
                        "Low": "#168a73",
                        "Medium": "#e59a30",
                        "Unknown": "#8792a2",
                    },
                ),
                "risk_box",
                height=420,
            )

    st.caption(
        "Risk-group comparisons describe patterns in the supplied labels."
    )


# ==============================================================
# 4. CLINICAL ANALYTICS
# ==============================================================

elif page == "Clinical Analytics":
    st.title("Clinical Analytics")
    st.markdown(
        '<div class="hp-subtitle">'
        "Inspect distributions, descriptive statistics and relationships "
        "among clinical measurements."
        "</div>",
        unsafe_allow_html=True,
    )

    available = [c for c in NUMERIC_FEATURES if c in view.columns]

    if not available:
        st.error("No supported clinical measurements are available.")
        st.stop()

    selected = st.selectbox(
        "Select indicator",
        available,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    values = num(view, selected)

    if not values.empty:
        a, b, c, d = st.columns(4)
        a.metric("Valid records", f"{len(values):,}")
        b.metric("Mean", fmt(values.mean(), 2))
        c.metric("Median", fmt(values.median(), 2))
        d.metric("Std. deviation", fmt(values.std(), 2))

        left, right = st.columns(2)

        with left:
            section("Distribution")
            chart(
                px.histogram(
                    pd.DataFrame({selected: values}),
                    x=selected,
                    nbins=30,
                    color_discrete_sequence=["#3867d6"],
                ),
                "clinical_distribution",
            )

        with right:
            section("Outlier and Spread Analysis")
            chart(
                px.box(
                    pd.DataFrame({selected: values}),
                    y=selected,
                    points="outliers",
                    color_discrete_sequence=["#078a83"],
                ),
                "clinical_box",
            )

    st.divider()
    section("Compare Two Clinical Indicators")

    if len(available) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox(
                "X-axis",
                available,
                index=0,
                key="clinical_x",
                format_func=lambda x: x.replace("_", " ").title(),
            )
        with c2:
            y_col = st.selectbox(
                "Y-axis",
                available,
                index=1,
                key="clinical_y",
                format_func=lambda x: x.replace("_", " ").title(),
            )

        if x_col != y_col:
            scatter_data = view[[x_col, y_col, "_risk"]].copy()
            scatter_data[x_col] = pd.to_numeric(
                scatter_data[x_col], errors="coerce"
            )
            scatter_data[y_col] = pd.to_numeric(
                scatter_data[y_col], errors="coerce"
            )
            scatter_data = scatter_data.dropna(subset=[x_col, y_col])

            if not scatter_data.empty:
                chart(
                    px.scatter(
                        scatter_data,
                        x=x_col,
                        y=y_col,
                        color="_risk",
                        opacity=0.7,
                        labels={
                            x_col: x_col.replace("_", " ").title(),
                            y_col: y_col.replace("_", " ").title(),
                            "_risk": "Risk label",
                        },
                        color_discrete_map={
                            "High": "#d84d61",
                            "Low": "#168a73",
                            "Medium": "#e59a30",
                            "Unknown": "#8792a2",
                        },
                    ),
                    "clinical_scatter",
                    height=430,
                )

                corr = scatter_data[x_col].corr(scatter_data[y_col])
                st.metric("Pearson Correlation", fmt(corr, 3))
                st.caption(
                    "Correlation measures linear association; it does not "
                    "establish causation."
                )

    st.divider()
    section("Clinical Correlation Matrix")

    matrix_data = view[available].apply(pd.to_numeric, errors="coerce")
    corr = matrix_data.corr()

    if corr.shape[0] >= 2:
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(xaxis_tickangle=-35)
        chart(fig, "clinical_corr", height=530)


# ==============================================================
# 5. MACHINE-LEARNING PREDICTION
# ==============================================================

elif page == "Predict Cardiovascular Risk":
    st.title("Cardiovascular Risk Prediction")
    st.markdown(
        '<div class="hp-subtitle">'
        "Random Forest classification using the clinical features in "
        "your healthcare dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    if model_result.get("error"):
        st.error("Model could not be trained: " + model_result["error"])
        st.stop()

    model = model_result["model"]
    classifier = model.named_steps["classifier"]

    a, b, c = st.columns(3)
    a.metric("Model", "Random Forest")
    b.metric("Test Accuracy", f"{model_result['accuracy']:.1%}")
    c.metric("Training Records", f"{model_result['train_rows']:,}")

    st.divider()

    with st.form("patient_prediction_form"):
        st.subheader("Enter Patient Measurements")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age (years)", 1, 120, 45
            )
            bmi = st.number_input(
                "BMI", 10.0, 80.0, 24.0, step=0.1
            )
            glucose = st.number_input(
                "Glucose (mg/dL)", 30, 700, 95
            )

        with c2:
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)", 50, 700, 180
            )
            systolic = st.number_input(
                "Systolic BP (mmHg)", 50, 280, 120
            )
            diastolic = st.number_input(
                "Diastolic BP (mmHg)", 30, 180, 80
            )

        with c3:
            heart_rate = st.number_input(
                "Heart Rate (bpm)", 25, 250, 72
            )

        predict_clicked = st.form_submit_button(
            "Predict Cardiovascular Risk",
            use_container_width=True,
        )

    if predict_clicked:
        patient = pd.DataFrame(
            [{
                "age": age,
                "bmi": bmi,
                "glucose_mg_dl": glucose,
                "cholesterol_mg_dl": cholesterol,
                "systolic_bp": systolic,
                "diastolic_bp": diastolic,
                "heart_rate": heart_rate,
            }],
            columns=NUMERIC_FEATURES,
        )

        predicted_class = model.predict(patient)[0]
        probabilities = model.predict_proba(patient)[0]
        classes = list(classifier.classes_)
        probability_map = dict(zip(classes, probabilities))
        predicted_probability = float(
            probability_map.get(predicted_class, np.nan)
        )

        st.divider()
        st.subheader("Prediction Result")

        r1, r2 = st.columns(2)

        with r1:
            st.metric("Predicted Risk Category", predicted_class.upper())

        with r2:
            st.metric(
                "Model Confidence",
                f"{predicted_probability:.1%}"
                if not np.isnan(predicted_probability)
                else "N/A",
            )

        probability_df = pd.DataFrame({
            "Risk Category": classes,
            "Predicted Probability": probabilities,
        })

        chart(
            px.bar(
                probability_df,
                x="Risk Category",
                y="Predicted Probability",
                color="Risk Category",
                text=probability_df["Predicted Probability"].map(
                    lambda x: f"{x:.1%}"
                ),
                range_y=[0, 1],
                color_discrete_map={
                    "High": "#d84d61",
                    "Low": "#168a73",
                },
            ),
            "prediction_probabilities",
            height=320,
        )

        prediction_record = patient.copy()
        prediction_record["predicted_risk"] = predicted_class
        prediction_record["model_confidence"] = predicted_probability

        st.download_button(
            "Download Prediction Result",
            prediction_record.to_csv(index=False).encode("utf-8"),
            file_name="cardiovascular_prediction.csv",
            mime="text/csv",
        )

        st.caption(
            "The classifier is trained on the dataset's existing labels. "
            "Its probability is a model estimate, not a calibrated "
            "medical probability or an individual diagnosis."
        )


# ==============================================================
# 6. MODEL PERFORMANCE
# ==============================================================

elif page == "Model Performance":
    st.title("Machine Learning Model Performance")
    st.markdown(
        '<div class="hp-subtitle">'
        "Random Forest evaluation on a stratified hold-out test set."
        "</div>",
        unsafe_allow_html=True,
    )

    if model_result.get("error"):
        st.error("Model evaluation unavailable: " + model_result["error"])
        st.stop()

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{model_result['accuracy']:.2%}")
    b.metric("High-Risk Precision", f"{model_result['precision']:.2%}")
    c.metric("High-Risk Recall", f"{model_result['recall']:.2%}")
    d.metric("High-Risk F1-score", f"{model_result['f1']:.2%}")

    st.caption(
        f"Training records: {model_result['train_rows']:,} | "
        f"Test records: {model_result['test_rows']:,} | "
        "Split: 80% training / 20% testing"
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        section("Confusion Matrix")
        cm = model_result["matrix"]
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Low", "Predicted High"],
                y=["Actual Low", "Actual High"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                showscale=True,
            )
        )
        cm_fig.update_layout(
            xaxis_title="Predicted label",
            yaxis_title="Actual label",
        )
        chart(cm_fig, "confusion_matrix", height=400)

    with right:
        section("Feature Importance")
        importance = model_result["importance"]
        chart(
            px.bar(
                importance.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                labels={
                    "Feature": "Input feature",
                    "Importance": "Importance",
                },
            ),
            "feature_importance",
            height=400,
        )

    st.divider()
    section("Classification Report")

    report = model_result["report"]
    report_rows = []

    for label in ["Low", "High"]:
        metrics = report.get(label, {})
        report_rows.append({
            "Class": label,
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1-score": metrics.get("f1-score", 0),
            "Support": metrics.get("support", 0),
        })

    report_df = pd.DataFrame(report_rows)
    st.dataframe(
        report_df.style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-score": "{:.3f}",
            "Support": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Classification Report",
        report_df.to_csv(index=False).encode("utf-8"),
        file_name="model_classification_report.csv",
        mime="text/csv",
    )

    st.divider()
    section("Test Set Predictions")

    test_predictions = model_result["X_test"].copy()
    test_predictions["Actual Risk"] = model_result["y_test"].values
    test_predictions["Predicted Risk"] = model_result["predictions"]
    test_predictions["Correct"] = (
        test_predictions["Actual Risk"]
        == test_predictions["Predicted Risk"]
    )

    st.dataframe(
        test_predictions.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Test Predictions",
        test_predictions.to_csv(index=False).encode("utf-8"),
        file_name="model_test_predictions.csv",
        mime="text/csv",
    )


# ==============================================================
# 7. DATA QUALITY & EXPORT
# ==============================================================

elif page == "Data Quality & Export":
    st.title("Data Quality & Export")
    st.markdown(
        '<div class="hp-subtitle">'
        "Inspect data types, missing values, duplicates, statistics "
        "and downloadable datasets."
        "</div>",
        unsafe_allow_html=True,
    )

    rows, columns = df.shape
    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    a, b, c, d = st.columns(4)
    a.metric("Records", f"{rows:,}")
    b.metric("Columns", f"{columns:,}")
    c.metric("Missing Cells", f"{missing_total:,}")
    d.metric("Memory Usage", f"{memory_mb:.2f} MB")

    st.metric("Duplicate Rows", f"{duplicate_rows:,}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Preview",
        "Quality Report",
        "Statistics",
        "Export",
    ])

    with tab1:
        section("Dataset Preview")
        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        section("Column-Level Quality")

        quality = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
            "Missing %": [
                round(float(df[c].isna().mean() * 100), 2)
                for c in df.columns
            ],
            "Unique Values": [
                int(df[c].nunique(dropna=True)) for c in df.columns
            ],
        })

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True,
        )

        missing_cols = quality[quality["Missing Values"] > 0]

        if missing_cols.empty:
            st.success("No missing values found.")
        else:
            st.write("Columns with missing values:")
            st.dataframe(
                missing_cols,
                use_container_width=True,
                hide_index=True,
            )

        st.write(f"Duplicate rows: {duplicate_rows:,}")

    with tab3:
        section("Numeric Descriptive Statistics")
        numeric_df = df.select_dtypes(include=np.number)

        if not numeric_df.empty:
            st.dataframe(
                numeric_df.describe().T.round(3),
                use_container_width=True,
            )
        else:
            st.info("No numeric columns found.")

        section("Categorical Descriptive Statistics")
        categorical_df = df.select_dtypes(
            include=["object", "string", "category"]
        )

        if not categorical_df.empty:
            st.dataframe(
                categorical_df.describe().T,
                use_container_width=True,
            )
        else:
            st.info("No categorical columns found.")

    with tab4:
        section("Download Data")

        st.download_button(
            "Download Full Dataset",
            df.to_csv(index=False).encode("utf-8"),
            file_name="healthcare_data_export.csv",
            mime="text/csv",
            key="full_export",
        )

        st.download_button(
            "Download Filtered Dataset",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="healthcare_filtered_export.csv",
            mime="text/csv",
            key="filtered_export",
        )

        st.download_button(
            "Download Quality Report",
            quality.to_csv(index=False).encode("utf-8"),
            file_name="healthcare_data_quality.csv",
            mime="text/csv",
            key="quality_export",
        )

        st.caption(f"Source file: {DATA_PATH.name}")


# ----------------------------- FOOTER -----------------------------

st.divider()
st.markdown(
    '<div class="hp-footer">'
    "HealthPulse Analytics · Healthcare Data Analytics Project"
    "</div>",
    unsafe_allow_html=True,
)
