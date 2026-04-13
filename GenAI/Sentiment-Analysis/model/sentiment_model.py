# ==============================
# SENTIMENT ANALYSIS PIPELINE
# ==============================

import pandas as pd
import re
import spacy
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

# ==============================
# LOAD SPACY MODEL
# ==============================
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ==============================
# PREPROCESS FUNCTION
# ==============================
def preprocess_text(texts):
    cleaned_texts = []

    for doc in nlp.pipe(texts, batch_size=500):
        tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            tokens.append(token.lemma_.lower())

        cleaned_texts.append(" ".join(tokens))

    print("Text preprocessing done...")

    return cleaned_texts


# ==============================
# LOAD DATASET
# ==============================
def load_data(path="../asset/IMDB Dataset.csv"):
    df = pd.read_csv(path)

    # Convert labels
    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    X = df["review"]
    y = df["sentiment"]

    print("Dataset loaded successfully...")

    return X, y


# ==============================
# BUILD PIPELINE
# ==============================
def build_pipeline():
    pipeline = Pipeline([
        ("preprocess", FunctionTransformer(preprocess_text)),
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9
        )),
        ("model", LinearSVC())
    ])

    print("Pipeline built successfully...")

    return pipeline


# ==============================
# TRAIN MODEL
# ==============================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return pipeline


# ==============================
# SAVE MODEL
# ==============================
def save_model(model, path="sentiment_pipeline.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved at {path}")


# ==============================
# LOAD MODEL
# ==============================
def load_model(path="sentiment_pipeline.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully...")
    return model


# ==============================
# PREDICT FUNCTION
# ==============================
def predict_sentiment(model, texts):
    predictions = model.predict(texts)

    results = []
    for text, pred in zip(texts, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        results.append({
            "review": text,
            "sentiment": sentiment
        })

    print("Predictions made successfully...")

    return results


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    
    # 1. Load data
    X, y = load_data("../asset/IMDB Dataset.csv")

    # 2. Train model
    model = train_model(X, y)

    # 3. Save model
    save_model(model)

    # 4. Test prediction
    test_reviews = [
        "This movie was amazing!",
        "Worst movie I have ever seen"
    ]

    print("\nSample Predictions:")
    results = predict_sentiment(model, test_reviews)

    for r in results:
        print(f"Review: {r['review']}")
        print(f"Sentiment: {r['sentiment']}\n")