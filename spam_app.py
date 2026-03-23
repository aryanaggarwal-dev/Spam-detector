import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

st.set_page_config(page_title="Email Spam Detector", layout="wide")


@st.cache_resource
def load_model():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)


stop_words = set(stopwords.words("english"))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)


def predict(email_text, artifacts):
    cleaned = clean_text(email_text)
    vec = artifacts["vectorizer"].transform([cleaned]).toarray()
    vec_sel = vec[:, artifacts["best_features"]]
    pred = artifacts["model"].predict(vec_sel)[0]
    proba = artifacts["model"].predict_proba(vec_sel)[0]
    conf = round(float(max(proba)) * 100, 1)
    label = "Spam" if pred == 1 else "Ham"

    if os.path.exists("spam_classifier.db"):
        conn = sqlite3.connect("spam_classifier.db")
        conn.execute(
            "INSERT INTO predictions (email_text, prediction, confidence) VALUES (?, ?, ?)",
            (email_text[:200], label, conf),
        )
        conn.commit()
        conn.close()

    return label, conf


st.title("Email Spam Detector")
st.caption("Hybrid PSO + Gaussian Mutation | SVM Classifier")
st.markdown("---")

try:
    artifacts = load_model()
    tab1, tab2, tab3 = st.tabs(["Classify Email", "Model Info", "Prediction Log"])

    with tab1:
        st.subheader("Classify an Email")
        st.caption("Quick samples:")
        col1, col2, col3, col4 = st.columns(4)

        spam1 = "Congratulations! You won a FREE lottery prize. Claim now!"
        spam2 = "URGENT: Your bank account suspended. Click to verify now."
        ham1 = "Hi, confirming our meeting tomorrow at 10am. Bring updates."
        ham2 = "Are you free for lunch on Friday? Let me know."

        if col1.button("Spam sample 1"):
            st.session_state["ei"] = spam1
        if col2.button("Spam sample 2"):
            st.session_state["ei"] = spam2
        if col3.button("Ham sample 1"):
            st.session_state["ei"] = ham1
        if col4.button("Ham sample 2"):
            st.session_state["ei"] = ham2

        email_input = st.text_area(
            "Paste or type your email here:",
            value=st.session_state.get("ei", ""),
            height=150,
            placeholder="Type an email message here...",
        )

        if st.button("Classify", type="primary"):
            if email_input.strip():
                label, conf = predict(email_input, artifacts)
                st.markdown("---")
                if label == "Spam":
                    st.error("SPAM detected  |  Confidence: " + str(conf) + "%")
                else:
                    st.success("HAM (Not Spam)  |  Confidence: " + str(conf) + "%")
                st.progress(conf / 100)
            else:
                st.warning("Please enter an email to classify.")

    with tab2:
        st.subheader("Model Performance")
        if os.path.exists("spam_classifier.db"):
            conn = sqlite3.connect("spam_classifier.db")
            results_df = pd.read_sql_query(
                "SELECT * FROM model_results ORDER BY id DESC LIMIT 1", conn
            )
            conn.close()
            if not results_df.empty:
                r = results_df.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  str(round(r["accuracy"] * 100, 1)) + "%")
                c2.metric("Precision", str(round(r["precision_score"] * 100, 1)) + "%")
                c3.metric("Recall",    str(round(r["recall_score"] * 100, 1)) + "%")
                c4.metric("F1-Score",  str(round(r["f1_score"] * 100, 1)) + "%")
                st.markdown("---")
                reduction = round(
                    (1 - r["selected_features"] / r["total_features"]) * 100, 1
                )
                st.write("Total features : " + str(r["total_features"]))
                st.write("PSO selected   : " + str(r["selected_features"]))
                st.write("Reduction      : " + str(reduction) + "%")

        st.markdown("---")
        st.subheader("Architecture")
        st.markdown(
            """
| Component | Detail |
|---|---|
| Feature Extraction | TF-IDF (500 features) |
| Optimization | PSO + Gaussian Mutation |
| Classifier | SVM (linear kernel) |
| Database | SQLite |
"""
        )

    with tab3:
        st.subheader("Prediction Log")
        if os.path.exists("spam_classifier.db"):
            conn = sqlite3.connect("spam_classifier.db")
            log_df = pd.read_sql_query(
                """SELECT id, timestamp, prediction, confidence,
                   SUBSTR(email_text, 1, 60) AS preview
                   FROM predictions ORDER BY id DESC LIMIT 20""",
                conn,
            )
            conn.close()
            if not log_df.empty:
                st.dataframe(log_df, use_container_width=True)
            else:
                st.info("No predictions logged yet. Classify some emails first.")

except FileNotFoundError:
    st.error("model_artifacts.pkl not found. Please run the notebook first.")
    st.code("Run all cells in the notebook, then: streamlit run spam_app.py")
