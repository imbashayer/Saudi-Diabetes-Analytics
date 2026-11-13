"""
Saudi Diabetes Analytics Dashboard
Educational portfolio project using synthetic data.
NOT FOR MEDICAL USE.

Run:
    python -m streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Saudi Diabetes Analytics",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

CSV_NAME = "synthetic_saudi_diabetes_central.csv"

FEATURES = [
    "age",
    "sex_m",
    "bmi",
    "family_history",
    "hypertension",
    "hypercholesterolemia",
]

STATUS_ORDER = [
    "normal",
    "prediabetes",
    "diabetes",
]

AGE_ORDER = [
    "15-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
]


# =========================================================
# 2. PAGE HEADER
# =========================================================

st.title("Saudi Diabetes Analytics")

st.caption(
    "A synthetic-data portfolio project | Saudi population context"
)

st.warning(
    "EDUCATIONAL USE ONLY — This project uses entirely synthetic "
    "records. It contains no real patient data and does not provide "
    "validated clinical risk estimates, diagnosis, or treatment advice."
)


# =========================================================
# 3. FIND DATASET
# =========================================================

def find_dataset():

    candidates = [
        BASE_DIR / "data" / CSV_NAME,
        BASE_DIR / "data" / "output" / CSV_NAME,
        BASE_DIR / "data" / "data" / "synthetic" / CSV_NAME,
        BASE_DIR / CSV_NAME,
    ]

    for path in candidates:
        if path.exists():
            return path

    # Last attempt: search inside the project folder
    matches = list(BASE_DIR.glob(f"**/{CSV_NAME}"))

    if matches:
        return matches[0]

    return None


CSV = find_dataset()


# =========================================================
# 4. DATA CONVERSION
# =========================================================

def convert_binary(series, column):

    """
    Convert binary values to 0/1.
    Missing values are preserved for later cleaning.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype("float")

    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
    }

    text_values = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    result = text_values.map(mapping)

    # Preserve missing values
    result = result.where(
        text_values.notna(),
        np.nan,
    )

    invalid = (
        text_values.notna()
        & result.isna()
    )

    if invalid.any():

        examples = (
            series[invalid]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        raise ValueError(
            f"Unsupported values in '{column}': {examples}. "
            "Expected 0/1, True/False, or Yes/No."
        )

    return result.astype(float)


# =========================================================
# 5. LOAD AND VALIDATE DATA
# =========================================================

@st.cache_data
def load_data(file_path, modified_time):

    df = pd.read_csv(file_path)

    required_columns = {
        "sex",
        "age",
        "age_group",
        "bmi",
        "family_history",
        "diabetes_status",
        "hypertension",
        "hypercholesterolemia",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in CSV: {sorted(missing_columns)}"
        )

    df = df.copy()

    # -----------------------------------------------------
    # Diabetes status: preserve all three categories
    # -----------------------------------------------------

    if df["diabetes_status"].isna().any():
        raise ValueError(
            "diabetes_status contains missing values."
        )

    status = (
        df["diabetes_status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    observed_status = set(status.unique())

    allowed_status = set(STATUS_ORDER)

    if observed_status != allowed_status:

        raise ValueError(
            "Unexpected diabetes_status labels.\n"
            f"Observed: {sorted(observed_status)}\n"
            f"Expected: {sorted(allowed_status)}"
        )

    df["diabetes_status"] = pd.Categorical(
        status,
        categories=STATUS_ORDER,
        ordered=True,
    )

    # -----------------------------------------------------
    # Binary target for machine learning only
    #
    # 1 = Diabetes
    # 0 = Normal or Prediabetes
    # -----------------------------------------------------

    df["y"] = (
        df["diabetes_status"]
        .astype("string")
        .eq("diabetes")
        .astype(int)
    )

    # -----------------------------------------------------
    # Sex
    # -----------------------------------------------------

    sex = (
        df["sex"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not sex.isin(["M", "F"]).all():
        raise ValueError(
            "The sex column must contain only M and F."
        )

    df["sex"] = sex

    df["sex_m"] = (
        df["sex"] == "M"
    ).astype(int)

    # -----------------------------------------------------
    # Numeric variables
    # -----------------------------------------------------

    for col in ["age", "bmi"]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # -----------------------------------------------------
    # Binary variables
    # -----------------------------------------------------

    binary_columns = [
        "family_history",
        "hypertension",
        "hypercholesterolemia",
    ]

    for col in binary_columns:

        df[col] = convert_binary(
            df[col],
            col,
        )

    # -----------------------------------------------------
    # Age group
    # -----------------------------------------------------

    df["age_group"] = (
        df["age_group"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    needed_columns = FEATURES + [
        "y",
        "age_group",
        "sex",
    ]

    original_rows = len(df)

    df = df.dropna(
        subset=needed_columns
    ).copy()

    removed_rows = original_rows - len(df)

    if len(df) < 40:
        raise ValueError(
            f"Only {len(df)} complete records remain. "
            "At least 40 records are required."
        )

    # -----------------------------------------------------
    # Outcome validation
    # -----------------------------------------------------

    if df["y"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both diabetes "
            "and non-diabetes records."
        )

    if df["y"].value_counts().min() < 2:
        raise ValueError(
            "Each binary outcome class needs at least "
            "two records."
        )

    # -----------------------------------------------------
    # Range validation
    # -----------------------------------------------------

    if not df["age"].between(15, 85).all():
        raise ValueError(
            "Age values must be between 15 and 85."
        )

    if not df["bmi"].between(10, 80).all():
        raise ValueError(
            "Some BMI values are outside the expected range 10–80."
        )

    return df, removed_rows


# =========================================================
# 6. TRAIN MODELS
# =========================================================

def train_models(data):

    X = data[FEATURES].copy()
    y = data["y"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    models = {}

    model_settings = [
        ("Unweighted", None),
        ("Balanced", "balanced"),
    ]

    for name, class_weight in model_settings:

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight=class_weight,
                random_state=42,
            ),
        )

        model.fit(
            X_train,
            y_train,
        )

        models[name] = model

    return (
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )


# =========================================================
# 7. LOAD APPLICATION DATA
# =========================================================

if CSV is None:

    st.error(
        f"Dataset not found: {CSV_NAME}"
    )

    st.info(
        "Place the CSV inside the project's data folder."
    )

    st.code(
        "Diabetes_Analysis/\n"
        "    app.py\n"
        "    data/\n"
        "        synthetic_saudi_diabetes_central.csv"
    )

    st.stop()


try:

    df, removed_rows = load_data(
        str(CSV),
        CSV.stat().st_mtime,
    )

    (
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_models(df)

except Exception as error:

    st.error(
        f"Data/model error: {error}"
    )

    st.exception(error)

    st.stop()


# =========================================================
# 8. DATASET STATISTICS
# =========================================================

simulator_model = models["Unweighted"]

prevalence = df["y"].mean()

status_counts = (
    df["diabetes_status"]
    .value_counts()
)

normal_n = int(status_counts.get("normal", 0))

prediabetes_n = int(
    status_counts.get("prediabetes", 0)
)

diabetes_n = int(
    status_counts.get("diabetes", 0)
)

normal_pct = normal_n / len(df) * 100

prediabetes_pct = prediabetes_n / len(df) * 100

diabetes_pct = diabetes_n / len(df) * 100


# =========================================================
# 9. DASHBOARD OVERVIEW
# =========================================================

st.subheader("Dataset Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Synthetic records",
    f"{len(df):,}",
)

m2.metric(
    "Normal",
    f"{normal_n:,}",
    f"{normal_pct:.1f}%",
)

m3.metric(
    "Prediabetes",
    f"{prediabetes_n:,}",
    f"{prediabetes_pct:.1f}%",
)

m4.metric(
    "Diabetes",
    f"{diabetes_n:,}",
    f"{diabetes_pct:.1f}%",
)

st.caption(
    f"Removed incomplete rows: {removed_rows:,} | "
    f"Training records: {len(X_train):,} | "
    f"Test records: {len(X_test):,}"
)

st.info(
    "The three original categories are preserved in descriptive "
    "analysis. For binary machine learning experiments only, "
    "Diabetes is the positive class. Normal and Prediabetes are "
    "combined into the negative class. This does NOT mean "
    "prediabetes is a normal condition."
)


# =========================================================
# 10. HELPER FUNCTIONS
# =========================================================

def risk(
    model,
    age,
    sex,
    bmi,
    fam,
    htn,
    chol,
):

    row = pd.DataFrame(
        [[
            age,
            int(sex == "Male"),
            bmi,
            int(fam),
            int(htn),
            int(chol),
        ]],
        columns=FEATURES,
    )

    return float(
        model.predict_proba(row)[0, 1]
    )


def factor_effects(model, person):

    reference = {
        "age": 40,
        "sex": "Female",
        "bmi": 24.0,
        "fam": False,
        "htn": False,
        "chol": False,
    }

    labels = {
        "age": "Age",
        "sex": "Sex",
        "bmi": "BMI",
        "fam": "Family history",
        "htn": "Hypertension",
        "chol": "High cholesterol",
    }

    current = risk(
        model,
        **person,
    )

    results = {}

    for key, label in labels.items():

        comparison = {
            **person,
            key: reference[key],
        }

        results[label] = (
            current - risk(model, **comparison)
        ) * 100

    return results


# =========================================================
# 11. DASHBOARD TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk Simulator",
    "Threshold Explorer",
    "Risk Pyramid",
    "What-if Analysis",
    "Model Diagnostics",
])


# =========================================================
# TAB 1 — RISK SIMULATOR
# =========================================================

with tab1:

    st.subheader(
        "Synthetic Model Score Simulator"
    )

    left, right = st.columns(2)

    with left:

        age = st.slider(
            "Age",
            15,
            85,
            45,
            key="sim_age",
        )

        sex = st.radio(
            "Sex",
            ["Male", "Female"],
            horizontal=True,
            key="sim_sex",
        )

        bmi = st.slider(
            "BMI",
            15.0,
            55.0,
            28.0,
            0.1,
            key="sim_bmi",
        )

        fam = st.checkbox(
            "Family history of diabetes",
            key="sim_fam",
        )

        htn = st.checkbox(
            "Hypertension",
            key="sim_htn",
        )

        chol = st.checkbox(
            "High cholesterol",
            key="sim_chol",
        )

    person = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "fam": fam,
        "htn": htn,
        "chol": chol,
    }

    score = risk(
        simulator_model,
        **person,
    )

    with right:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number={
                    "suffix": "%",
                    "valueformat": ".1f",
                },
                title={
                    "text": "Model-predicted score",
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                    },
                    "bar": {
                        "color": "#4C78A8",
                    },
                    "steps": [
                        {
                            "range": [0, 15],
                            "color": "#d4edda",
                        },
                        {
                            "range": [15, 35],
                            "color": "#fff3cd",
                        },
                        {
                            "range": [35, 100],
                            "color": "#f8d7da",
                        },
                    ],
                },
            )
        )

        gauge.update_layout(
            height=300
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

        st.metric(
            "Synthetic dataset diabetes prevalence",
            f"{prevalence:.1%}",
        )

    st.caption(
        "This is a model score learned from synthetic data, "
        "not a validated medical risk estimate. "
        "Gauge colors are visual bands only."
    )

    effects = factor_effects(
        simulator_model,
        person,
    )

    effect_df = pd.DataFrame({
        "Factor": list(effects.keys()),
        "Score difference (percentage points)": list(
            effects.values()
        ),
    })

    st.subheader(
        "Comparison with a Reference Profile"
    )

    st.caption(
        "Reference: age 40, female, BMI 24, and no recorded "
        "risk factors. Differences are associations learned "
        "from synthetic data, not causal effects."
    )

    effect_fig = px.bar(
        effect_df,
        x="Score difference (percentage points)",
        y="Factor",
        orientation="h",
        color="Score difference (percentage points)",
        color_continuous_scale="RdYlGn_r",
    )

    st.plotly_chart(
        effect_fig,
        use_container_width=True,
    )


# =========================================================
# TAB 2 — THRESHOLD EXPLORER
# =========================================================

with tab2:

    st.subheader(
        "Classification Threshold Explorer"
    )

    model_name = st.radio(
        "Model",
        ["Balanced", "Unweighted"],
        horizontal=True,
        key="threshold_model",
    )

    model = models[model_name]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    st.caption(
        f"Test ROC-AUC: {test_auc:.3f}"
    )

    default_threshold = (
        0.50 if model_name == "Balanced" else 0.25
    )

    threshold = st.slider(
        "Decision threshold",
        0.05,
        0.95,
        default_threshold,
        0.01,
        key="decision_threshold",
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    recall_value = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    precision_value = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    f1_value = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Recall",
        f"{recall_value:.1%}",
    )

    c2.metric(
        "Precision",
        f"{precision_value:.1%}",
    )

    c3.metric(
        "Specificity",
        f"{specificity:.1%}",
    )

    c4.metric(
        "F1-score",
        f"{f1_value:.3f}",
    )

    st.info(
        f"At threshold {threshold:.2f}, the model identifies "
        f"{tp:,} of {tp + fn:,} positive test cases, flags "
        f"{fp:,} negative cases as positive, and misses "
        f"{fn:,} positive cases."
    )

    cm_fig = go.Figure(
        go.Heatmap(
            z=[
                [tn, fp],
                [fn, tp],
            ],
            x=[
                "Predicted: Non-diabetes*",
                "Predicted: Diabetes",
            ],
            y=[
                "Actual: Non-diabetes*",
                "Actual: Diabetes",
            ],
            text=[
                [tn, fp],
                [fn, tp],
            ],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )

    cm_fig.update_layout(
        title="Confusion Matrix",
        height=350,
        yaxis_autorange="reversed",
    )

    st.plotly_chart(
        cm_fig,
        use_container_width=True,
    )

    st.caption(
        "*Non-diabetes combines normal and prediabetes. "
        "Changing the threshold changes classification decisions; "
        "it does not recalibrate predicted probabilities."
    )


# =========================================================
# TAB 3 — RISK PYRAMID
# =========================================================

with tab3:

    st.subheader(
        "Diabetes Prevalence by Age Group and Sex"
    )

    grouped = (
        df.groupby(
            ["age_group", "sex"],
            observed=True,
        )
        .agg(
            cases=("y", "sum"),
            population=("y", "size"),
            rate=("y", "mean"),
        )
        .reset_index()
    )

    grouped["rate"] *= 100

    grouped["sex_label"] = grouped["sex"].map({
        "M": "Male",
        "F": "Female",
    })

    grouped["signed_population"] = np.where(
        grouped["sex_label"].eq("Male"),
        -grouped["population"],
        grouped["population"],
    )

    observed_groups = (
        grouped["age_group"]
        .unique()
        .tolist()
    )

    age_order = [
        group
        for group in AGE_ORDER
        if group in observed_groups
    ]

    age_order += [
        group
        for group in observed_groups
        if group not in age_order
    ]

    pyramid = px.bar(
        grouped,
        x="signed_population",
        y="age_group",
        color="rate",
        orientation="h",
        barmode="overlay",
        category_orders={
            "age_group": age_order,
        },
        color_continuous_scale="YlOrRd",
        hover_data={
            "sex_label": True,
            "rate": ":.2f",
            "population": True,
            "cases": True,
            "signed_population": False,
        },
        labels={
            "signed_population": (
                "Synthetic population (Male ← → Female)"
            ),
            "age_group": "Age group",
            "rate": "Diabetes prevalence (%)",
            "sex_label": "Sex",
        },
    )

    pyramid.update_layout(
        height=500
    )

    st.plotly_chart(
        pyramid,
        use_container_width=True,
    )

    st.caption(
        "Bar length represents synthetic group size; "
        "color represents the percentage labelled as diabetes. "
        "These are not verified Saudi population estimates."
    )

    table = grouped[
        [
            "age_group",
            "sex_label",
            "cases",
            "population",
            "rate",
        ]
    ].rename(columns={
        "age_group": "Age group",
        "sex_label": "Sex",
        "cases": "Diabetes cases",
        "population": "Group size",
        "rate": "Diabetes (%)",
    })

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TAB 4 — WHAT-IF ANALYSIS
# =========================================================

with tab4:

    st.subheader(
        "BMI Scenario Comparison"
    )

    left, right = st.columns(2)

    with left:

        age_w = st.slider(
            "Age",
            15,
            85,
            50,
            key="whatif_age",
        )

        sex_w = st.radio(
            "Sex",
            ["Male", "Female"],
            horizontal=True,
            key="whatif_sex",
        )

        fam_w = st.checkbox(
            "Family history",
            key="whatif_fam",
        )

        htn_w = st.checkbox(
            "Hypertension",
            key="whatif_htn",
        )

        chol_w = st.checkbox(
            "High cholesterol",
            key="whatif_chol",
        )

        bmi_now = st.slider(
            "Scenario A: BMI",
            15.0,
            55.0,
            32.0,
            0.1,
            key="bmi_now",
        )

        bmi_goal = st.slider(
            "Scenario B: BMI",
            15.0,
            55.0,
            27.0,
            0.1,
            key="bmi_goal",
        )

    fixed = {
        "age": age_w,
        "sex": sex_w,
        "fam": fam_w,
        "htn": htn_w,
        "chol": chol_w,
    }

    score_a = risk(
        simulator_model,
        bmi=bmi_now,
        **fixed,
    )

    score_b = risk(
        simulator_model,
        bmi=bmi_goal,
        **fixed,
    )

    with right:

        c1, c2 = st.columns(2)

        c1.metric(
            "Scenario A score",
            f"{score_a:.1%}",
        )

        c2.metric(
            "Scenario B score",
            f"{score_b:.1%}",
        )

        bmi_grid = np.arange(
            15,
            55.1,
            0.5,
        )

        score_curve = [
            risk(
                simulator_model,
                bmi=float(value),
                **fixed,
            ) * 100
            for value in bmi_grid
        ]

        curve_fig = go.Figure(
            go.Scatter(
                x=bmi_grid,
                y=score_curve,
                mode="lines",
                name="Model score",
            )
        )

        curve_fig.add_vline(
            x=bmi_now,
            line_dash="dot",
        )

        curve_fig.add_vline(
            x=bmi_goal,
            line_dash="dash",
        )

        curve_fig.update_layout(
            height=320,
            xaxis_title="BMI",
            yaxis_title="Model score (%)",
        )

        st.plotly_chart(
            curve_fig,
            use_container_width=True,
        )

    st.warning(
        "This analysis changes BMI while holding other inputs fixed. "
        "It shows a model association, not the causal effect of weight "
        "change or a prediction of future health."
    )


# =========================================================
# TAB 5 — MODEL DIAGNOSTICS
# =========================================================

with tab5:

    st.subheader(
        "Held-out Test-set Evaluation"
    )

    report_rows = []

    for name, model in models.items():

        proba = model.predict_proba(
            X_test
        )[:, 1]

        pred = (
            proba >= 0.5
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            pred,
            labels=[0, 1],
        ).ravel()

        report_rows.append({
            "Model": name,

            "ROC-AUC": roc_auc_score(
                y_test,
                proba,
            ),

            "Accuracy": accuracy_score(
                y_test,
                pred,
            ),

            "Recall": recall_score(
                y_test,
                pred,
                zero_division=0,
            ),

            "Precision": precision_score(
                y_test,
                pred,
                zero_division=0,
            ),

            "Specificity": (
                tn / (tn + fp)
                if tn + fp
                else 0.0
            ),

            "F1-score": f1_score(
                y_test,
                pred,
                zero_division=0,
            ),

            "Brier score": brier_score_loss(
                y_test,
                proba,
            ),
        })

    report = pd.DataFrame(
        report_rows
    )

    st.dataframe(
        report.style.format({
            col: "{:.3f}"
            for col in report.columns
            if col != "Model"
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Higher ROC-AUC generally indicates better ranking. "
        "Lower Brier score indicates better probabilistic accuracy. "
        "These metrics evaluate synthetic test data only."
    )

    # -----------------------------------------------------
    # Calibration curve
    # -----------------------------------------------------

    st.subheader(
        "Probability Calibration"
    )

    calibration_fig = go.Figure()

    for name, model in models.items():

        proba = model.predict_proba(
            X_test
        )[:, 1]

        observed, predicted = calibration_curve(
            y_test,
            proba,
            n_bins=8,
            strategy="quantile",
        )

        calibration_fig.add_trace(
            go.Scatter(
                x=predicted,
                y=observed,
                mode="lines+markers",
                name=name,
            )
        )

    calibration_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={
                "dash": "dash",
            },
            name="Perfect calibration",
        )
    )

    calibration_fig.update_layout(
        title="Calibration Curve — Held-out Test Set",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed positive fraction",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=450,
    )

    st.plotly_chart(
        calibration_fig,
        use_container_width=True,
    )

    st.caption(
        "Calibration is evaluated against synthetic test labels only. "
        "It does not establish real-world medical calibration."
    )

    # -----------------------------------------------------
    # Dataset summary
    # -----------------------------------------------------

    st.subheader(
        "Dataset Summary"
    )

    summary = pd.DataFrame({
        "Metric": [
            "Synthetic records",
            "Synthetic diabetes prevalence",
            "Synthetic prediabetes prevalence",
            "Synthetic normal prevalence",
            "Test records",
            "Test diabetes prevalence",
        ],

        "Value": [
            f"{len(df):,}",
            f"{prevalence:.2%}",
            f"{prediabetes_n / len(df):.2%}",
            f"{normal_n / len(df):.2%}",
            f"{len(y_test):,}",
            f"{y_test.mean():.2%}",
        ],
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # Three-category distribution
    # -----------------------------------------------------

    st.subheader(
        "Synthetic Diabetes Status Distribution"
    )

    status_plot = pd.DataFrame({
        "Status": [
            "Normal",
            "Prediabetes",
            "Diabetes",
        ],

        "Records": [
            normal_n,
            prediabetes_n,
            diabetes_n,
        ],
    })

    status_fig = px.pie(
        status_plot,
        names="Status",
        values="Records",
        hole=0.45,
        title="Distribution of Generated Records",
    )

    st.plotly_chart(
        status_fig,
        use_container_width=True,
    )

    st.caption(
        "All three generated status categories are retained "
        "in descriptive analysis."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption(
    "Saudi Diabetes Analytics | Synthetic educational project | "
    "Not intended for medical diagnosis or clinical decisions."
)