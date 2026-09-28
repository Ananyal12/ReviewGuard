from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix, hstack

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with TFIDF_PATH.open("rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf


def build_features(review: str, tfidf):
    """Create the exact feature representation used during training."""
    review = str(review)
    exclamation_count = review.count("!")
    uppercase_count = sum(char.isupper() for char in review)
    word_count = len(review.split())

    text_vector = tfidf.transform([review])
    extra_features = csr_matrix(
        [[exclamation_count, uppercase_count, word_count]], dtype=np.float64
    )
    return hstack([text_vector, extra_features], format="csr")


def main():
    st.set_page_config(
        page_title="Fake Review Detector",
        page_icon="🔎",
        layout="centered",
    )

    st.title("🔎 Fake Review Detector")
    st.write(
        "Enter a product or service review and the trained NLP model will estimate "
        "whether it is likely to be fake or genuine."
    )

    with st.sidebar:
        st.header("About the model")
        st.write("• TF-IDF text features (5,000 features)")
        st.write("• 3 writing-style features")
        st.write("• Logistic Regression classifier")
        st.write("• Test accuracy: ~91.0%")
        st.caption("Predictions are model estimates, not proof that a review is fake.")

    review = st.text_area(
        "Enter your review",
        placeholder="Example: The product quality was excellent and delivery was very fast!",
        height=180,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        predict_clicked = st.button("🔍 Check Review", use_container_width=True)
    with col2:
        clear_clicked = st.button("🧹 Clear", use_container_width=True)

    if clear_clicked:
        st.rerun()

    if predict_clicked:
        if not review.strip():
            st.warning("Please enter a review before clicking Check Review.")
            return

        try:
            model, tfidf = load_artifacts()
            features = build_features(review, tfidf)
            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities)) * 100

            st.divider()
            if prediction == 1:
                st.error("🚨 Likely Fake Review")
                st.write(
                    f"The model estimates this review as **fake** with "
                    f"{confidence:.1f}% confidence."
                )
            else:
                st.success("✅ Likely Genuine Review")
                st.write(
                    f"The model estimates this review as **genuine** with "
                    f"{confidence:.1f}% confidence."
                )

            with st.expander("View extracted features"):
                st.write(f"Word count: **{len(review.split())}**")
                st.write(f"Exclamation marks: **{review.count('!')}**")
                st.write(
                    "Uppercase characters: **"
                    f"{sum(char.isupper() for char in review)}**"
                )

        except Exception as exc:
            st.error("The model could not process this review.")
            st.exception(exc)


if __name__ == "__main__":
    main()
