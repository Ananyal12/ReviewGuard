"""Train and save the Fake Review Detector model.

The important detail is that TF-IDF is fitted ONLY on the training reviews.
This prevents information from the test set leaking into model training.
"""
from pathlib import Path
import pickle

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "fake_reviews.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"


def add_features(df):
    df = df.copy()
    df["text_"] = df["text_"].fillna("").astype(str)
    df["exclamation_count"] = df["text_"].str.count("!")
    df["uppercase_count"] = df["text_"].apply(lambda text: sum(c.isupper() for c in text))
    df["word_count"] = df["text_"].str.split().str.len()
    return df


def main():
    df = add_features(pd.read_csv(DATA_PATH))
    y = df["label"].map({"CG": 1, "OR": 0}).astype(int)

    train_text, test_text, y_train, y_test, train_idx, test_idx = train_test_split(
        df["text_"],
        y,
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    train_text_features = vectorizer.fit_transform(train_text)
    test_text_features = vectorizer.transform(test_text)

    extra = df[["exclamation_count", "uppercase_count", "word_count"]].values
    train_extra = csr_matrix(extra[train_idx])
    test_extra = csr_matrix(extra[test_idx])

    X_train = hstack([train_text_features, train_extra], format="csr")
    X_test = hstack([test_text_features, test_extra], format="csr")

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print("Model iterations:", model.n_iter_)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with TFIDF_PATH.open("wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved: {MODEL_PATH.name} and {TFIDF_PATH.name}")


if __name__ == "__main__":
    main()
