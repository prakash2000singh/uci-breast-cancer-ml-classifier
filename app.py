import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# 1. Page Configuration
st.set_page_config(
    page_title="UCI Breast Cancer Classifier Evaluation",
    page_icon="🩺",
    layout="wide"
)

# 2. Header & UCI Dataset Metadata
st.title("🩺 UCI Breast Cancer Wisconsin (Diagnostic) — ML Evaluation Portal")
st.markdown(
    """
    **M.Tech (AIML) — Machine Learning Assignment 2**  
    * **Dataset Source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
    * **Target Encoding:** `0 = Malignant (M)` | `1 = Benign (B)`  
    * **Workflow:** Upload the held-out `test_data.csv` file in the sidebar to evaluate the 5 pre-trained classifiers without retraining.
    """
)

# 3. Load Saved Preprocessing Artifacts & Pre-Trained Models (No Retraining)
@st.cache_resource
def load_saved_artifacts():
    scaler = joblib.load("model/scaler.joblib")
    feature_cols = joblib.load("model/feature_columns.joblib")
    trained_models = {
        "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
        "Decision Tree": joblib.load("model/decision_tree.joblib"),
        "KNN": joblib.load("model/knn.joblib"),
        "Naive Bayes": joblib.load("model/naive_bayes.joblib"),
        "Support Vector Machine": joblib.load("model/support_vector_machine.joblib"),
    }
    return scaler, feature_cols, trained_models

try:
    scaler, feature_cols, models = load_saved_artifacts()
except Exception as e:
    st.error(
        f"⚠️ Error loading saved model artifacts from the `model/` folder: {e}. "
        "Please verify that `model/*.joblib` files are uploaded to your GitHub repository."
    )
    st.stop()

# 4. Sidebar Controls (Step 6 Features a & b)
st.sidebar.header("⚙️ Evaluation Controls")

uploaded_file = st.sidebar.file_uploader(
    "a. Upload Held-Out Test Data (CSV)",
    type=["csv"],
    help="Upload 'test_data.csv' generated from the UCI Breast Cancer dataset (30 features + 'target' column)."
)

selected_model_name = st.sidebar.selectbox(
    "b. Select Classification Model",
    options=list(models.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Repository & Model Info:**
    * **Total Input Features:** `30` (UCI WDBC)
    * **Models Loaded:** `5` (`.joblib` artifacts)
    * **Inference Mode:** Pre-trained (`StandardScaler` + Classifier)
    """
)

# 5. Helper Function to Compute All 6 Mandatory Evaluation Metrics
def compute_metrics(clf, X_scaled, y_true):
    y_pred = clf.predict(X_scaled)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_scaled)[:, 1]
    else:
        y_prob = y_pred

    metrics_dict = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    return metrics_dict, y_pred

# 6. Main Application View
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    # Optional support if someone uploads a file with raw 'diagnosis' ('M'/'B') instead of 'target'
    if "target" not in test_df.columns and "diagnosis" in test_df.columns:
        test_df["target"] = test_df["diagnosis"].map({"M": 0, "B": 1})

    st.subheader("📄 Uploaded Test Dataset Preview (`test_data.csv`)")
    st.caption(f"Showing first 5 rows | Total Test Instances: **{test_df.shape[0]}** | Columns: **{test_df.shape[1]}**")
    st.dataframe(test_df.head(5), use_container_width=True)

    # Validate that all 30 UCI feature columns and 'target' exist
    missing_features = [col for col in feature_cols if col not in test_df.columns]
    if missing_features:
        st.error(f"❌ Uploaded CSV is missing required UCI feature columns: {missing_features[:5]}...")
        st.stop()
    if "target" not in test_df.columns:
        st.error("❌ Uploaded CSV must include the ground-truth `target` column (`0` or `1`) to calculate evaluation metrics.")
        st.stop()

    # Separate features and target, and apply saved StandardScaler (without refitting)
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate the currently selected model from the dropdown
    selected_clf = models[selected_model_name]
    single_metrics, y_pred = compute_metrics(selected_clf, X_test_scaled, y_test)

    st.markdown("---")

    # Step 6 Feature (c): Display of 6 Evaluation Metrics
    st.subheader(f"📊 c. Evaluation Metrics — **{selected_model_name}**")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{single_metrics['Accuracy']:.4f}")
    m2.metric("AUC Score", f"{single_metrics['AUC']:.4f}")
    m3.metric("Precision", f"{single_metrics['Precision']:.4f}")
    m4.metric("Recall", f"{single_metrics['Recall']:.4f}")
    m5.metric("F1 Score", f"{single_metrics['F1']:.4f}")
    m6.metric("MCC Score", f"{single_metrics['MCC']:.4f}")

    st.markdown("---")

    # Step 6 Feature (d): Confusion Matrix & Classification Report
    st.subheader(f"🔍 d. Confusion Matrix & Classification Report — **{selected_model_name}**")
    left_col, right_col = st.columns([1, 1.25])

    with left_col:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=False,
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax
        )
        ax.set_xlabel("Predicted Diagnosis", fontweight="bold")
        ax.set_ylabel("Actual Diagnosis", fontweight="bold")
        ax.set_title(f"{selected_model_name} — Test Confusion Matrix")
        st.pyplot(fig)

    with right_col:
        st.markdown("**Detailed Classification Report**")
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=["Malignant (Class 0)", "Benign (Class 1)"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")

    # Step 6 Requirement: "The results of different models on your 'test data' should be visible on the streamlit app."
    st.subheader("🏆 All 5 Models — Comparative Benchmark on Uploaded `test_data.csv`")
    benchmark_rows = []
    for model_label, model_obj in models.items():
        m_scores, _ = compute_metrics(model_obj, X_test_scaled, y_test)
        benchmark_rows.append({
            "ML Model Name": model_label,
            "Accuracy": round(m_scores["Accuracy"], 4),
            "AUC": round(m_scores["AUC"], 4),
            "Precision": round(m_scores["Precision"], 4),
            "Recall": round(m_scores["Recall"], 4),
            "F1": round(m_scores["F1"], 4),
            "MCC": round(m_scores["MCC"], 4),
        })

    benchmark_df = pd.DataFrame(benchmark_rows)
    st.dataframe(
        benchmark_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}",
        }).highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            color="#d4edda"
        ),
        use_container_width=True
    )

else:
    st.info(
        "👈 **Action Required:** Please upload **`test_data.csv`** using the sidebar uploader on the left "
        "to view the evaluation metrics, confusion matrix, classification report, and 5-model comparison table."
    )