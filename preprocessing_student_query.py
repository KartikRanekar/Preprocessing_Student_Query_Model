# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data quietly
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

def main():
    try:
        df = pd.read_csv('uoft_reddit_dataset_20250719_141657.csv')
    except Exception as e:
        print("CRITICAL ERROR: Dataset could not be loaded.", e)
        sys.exit(1)

    possible_text_cols = ['title', 'selftext', 'body', 'query', 'content']
    possible_label_cols = ['link_flair_text', 'flair', 'tag', 'label', 'quality_recommendation']

    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    label_col = next((col for col in possible_label_cols if col in df.columns), None)

    if text_col and label_col:
        print(f"Using Text Column: '{text_col}' and Label Column: '{label_col}'")

        if 'title' in df.columns and 'selftext' in df.columns:
            df['combined_text'] = df['title'].fillna('') + " " + df['selftext'].fillna('')
            text_col = 'combined_text'

        df = df.dropna(subset=[label_col, text_col])
        df = df[df[text_col].astype(str).str.strip() != '']

        flair_counts = df[label_col].value_counts()
        valid_flairs = flair_counts[flair_counts > 50].index
        df = df[df[label_col].isin(valid_flairs)]

        print(f"Filtered Dataset Size: {len(df)}")
        if len(df) == 0:
            print("CRITICAL WARNING: Dataset empty after filtering.")
            sys.exit(1)
        print("Top Flairs:", df[label_col].value_counts().head())
    else:
        print("CRITICAL WARNING: Could not find suitable text or label columns.")
        sys.exit(1)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    spell = SpellChecker()

    custom_stops = {'removed', 'deleted', 'view', 'poll', 'reddit', 'https', 'www', 'com'}
    stop_words.update(custom_stops)

    slang_map = {
        "cs": "computer science",
        "uoft": "university of toronto",
        "prof": "professor",
        "uni": "university",
        "res": "residence",
        "ta": "teaching assistant",
        "gpa": "grade point average",
        "calc": "calculus",
        "stats": "statistics",
        "bird course": "easy course",
        "rn": "right now",
        "dm": "direct message",
        "idk": "i do not know"
    }

    def preprocess_text(text, fix_spelling=False):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'u/[\w-]+', '', text)
        text = re.sub(r'r/[\w-]+', '', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = text.split()
        clean_tokens = []
        for token in tokens:
            token = slang_map.get(token, token)
            if token not in stop_words and len(token) > 1:
                lemma = lemmatizer.lemmatize(token)
                clean_tokens.append(lemma)
        if fix_spelling:
            clean_tokens = [spell.correction(word) if spell.correction(word) else word for word in clean_tokens]
        return " ".join(clean_tokens)

    print("Preprocessing text... this may take a while depending on dataset size.")
    df['processed_text'] = df[text_col].apply(lambda x: preprocess_text(x, fix_spelling=False))
    df = df[df['processed_text'] != '']
    print("Preprocessing complete.")

    X = df['processed_text']
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("Vectorization complete.")

    print("Training LinearSVC Model...")
    model = LinearSVC(random_state=42, dual='auto')
    model.fit(X_train_vec, y_train)
    print("Model trained successfully.")

    print("Evaluating Model...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n[TEST SUCCESSFUL] Execution completed without errors.")

    print("\n--- Model Testing Loop ---")
    print("You can now test the model interactively. Type 'exit' to stop.")
    while True:
        try:
            user_input = input("\nEnter a test query: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if user_input.strip():
                processed_query = preprocess_text(user_input, fix_spelling=True)
                if not processed_query:
                    print("Processed Query: [Could not process text]")
                    print("Predicted Label: Unknown")
                    continue
                vec = vectorizer.transform([processed_query])
                prediction = model.predict(vec)[0]
                print(f"Processed Query: {processed_query}")
                print(f"Predicted Label: {prediction}")
            else:
                print("Please enter a query.")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()