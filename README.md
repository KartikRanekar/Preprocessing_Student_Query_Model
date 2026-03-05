# Student Query Preprocessing and Intent Classification Model

This project establishes a data preprocessing pipeline and a machine learning model for analyzing, cleaning, and classifying student queries. It utilizes natural language processing (NLP) to parse typical university-related queries, clean the text, auto-correct spelling, and predict the intent or "quality recommendation" of a given query based on an underlying dataset.

## 🚀 Features

*   **Robust Text Preprocessing Pipeline:**
    *   Lowercasing all inputs for consistency.
    *   URL & User Removal (designed keeping Reddit datasets in mind, removes `http`, `www`, `u/username`, `r/subreddit`).
    *   Special Character & Punctuation Cleanup.
    *   Tokenization into individual words.
    *   **Custom Slang Translation:** Translates common student abbreviations (e.g., `cs` -> `computer science`, `gpa` -> `grade point average`, `bird course` -> `easy course`).
    *   Stopword Removal (including custom domain-specific exclusions).
    *   Lemmatization using NLTK's `WordNetLemmatizer`.
    *   **Spelling Correction:** (Optional during bulk training for speed, enforced during interactive inference) using `pyspellchecker`.
*   **Machine Learning Model:**
    *   Feature Extraction via `TfidfVectorizer` (up to 5000 unigram and bigram features).
    *   Classification using `LinearSVC` (Support Vector Machine), optimized for text features.
*   **Interactive Testing Loop:** Test out the trained model directly from the command line on your custom queries.

## 📋 Prerequisites

To deploy this project locally, ensure you have Python installed. The required dependencies are:
*   `pandas`
*   `numpy`
*   `nltk`
*   `scikit-learn`
*   `pyspellchecker`
*   `matplotlib` (optional for future visualization)
*   `seaborn` (optional for future visualization)

The NLTK resources (`stopwords`, `wordnet`, `omw-1.4`) will be downloaded automatically when running the script for the first time.

## 🔧 Installation

1.  Clone or download this repository to your local machine.
2.  Install the required Python packages using pip:

```bash
pip install pandas numpy nltk scikit-learn pyspellchecker matplotlib seaborn
```

3.  Ensure the dataset `uoft_reddit_dataset_20250719_141657.csv` is present in the root directory.

## 🏃 Usage

Run the main Python script to start the training pipeline and enter the interactive testing mode:

```bash
python preprocessing_student_query.py
```

### What happens when you run the script?

1.  **Data Loading:** The dataset is read using Pandas.
2.  **Column Detection:** The script detects the appropriate text and label columns (e.g., uses `title` and `quality_recommendation`).
3.  **Preprocessing:** The text is cleaned and transformed. Warning: Depending on dataset size, this may take a minute.
4.  **Vectorization & Training:** The text is converted into numerical features (TF-IDF), and the LinearSVC model is trained.
5.  **Evaluation:** It prints out the accuracy and a robust classification report based on test data.
6.  **Interactive Test:** The script launches a command-line prompt (`Enter a test query:`) where you can type queries. The model will run the full preprocessing (including spell check) and return its prediction. Type `exit` to close it.

## 📂 Project Structure

*   `preprocessing_student_query.py`: The main Python script containing the pipeline, NLP mapping, training, and the interactive chatbot loop.
*   `uoft_reddit_dataset_20250719_141657.csv`: The core dataset containing student queries and discussions used for training the model.

## 💡 Future Improvements

*   Implement Deep Learning techniques (e.g., BERT, LSTMs) to improve intent detection, specifically for ambiguous sentences.
*   Expand the `slang_map` dictionary to cover more university terms and student dialects.
*   Export the trained model using `joblib` or `pickle` for inference in other applications without re-training on startup.
*   Deploy a frontend UI tool (like Streamlit or FastAPI) to make the model visually accessible.








# 🗺️ Roadmap: Student Query Preprocessing and Intent Classification

This roadmap outlines the past development phases, the current state of the project, and a strategic guide for future enhancements to elevate this tool from a local command-line script to a potentially shippable, production-ready product.

---

## ✅ Phase 1: Foundation and Proof of Concept (Completed)
*   [x] Gather and inspect the foundational university/student discussion dataset.
*   [x] Establish an initial data loading and visualization baseline.
*   [x] Identify key features (Text, Intent Labels).
*   [x] Build a functional NLP text cleaning pipeline (regex, lowercasing, stopword removal).
*   [x] Basic intent prediction model using Bag of Words / Basic Vectorization.

## ✅ Phase 2: Refinement and Pipeline Standardization (Completed - Current State)
*   [x] Convert development `.ipynb` Notebooks to a standalone, standardized `.py` script.
*   [x] Refine NLP tasks: Introduce lemmatization and dynamic slang-to-standard-English mapping.
*   [x] Incorporate `pyspellchecker` for auto-correcting user input on inference.
*   [x] Transition to advanced TF-IDF feature extraction (Unigrams + Bigrams).
*   [x] Optimize classifier model using `LinearSVC` for highly dimensional text data points.
*   [x] Create a continuous, integrated Command-Line Interface (CLI) testing loop for immediate demonstrations.
*   [x] Graceful error handling and dataset dependency checks.

---

## 🚀 Phase 3: Model Persistence and Optimization (Short-term Goals)
*   **Model Serialization:** Use `joblib` or `pickle` to save (`model.fit()`) and load both the `TfidfVectorizer` and `LinearSVC` model. This avoids retraining the model every single time the script is executed.
*   **Hyperparameter Tuning:** Implement `GridSearchCV` to optimize the parameters (like `C`, `penalty`) of the Linear SVM.
*   **Expanded Validation:** Introduce K-Fold Cross Validation rather than a strict 80/20 train/test split to guarantee generalization.
*   **Expanded Vocabulary:** Expand the student `slang_map` to include global academic slang instead of focusing only on one specific university domain (e.g., generalize `uoft` to `university`).

## 🌐 Phase 4: UI/UX & Interaction (Medium-term Goals)
*   **Streamlit Dashboard:** Wrap the trained model in a simple web app using Streamlit to offer a GUI for users instead of a terminal interface.
*   **FastAPI / Flask Backend:** Extract the model prediction code into a standalone API endpoint (`/predict`) that takes JSON queries and returns predictions and cleaned texts.
*   **Advanced Chatbot Integration:** Shift from just identifying "Quality/Intent" towards passing the cleaned query downstream into an LLM or a rule-based system to dynamically *answer* the student.

## 🧠 Phase 5: Deep Learning & State of the Art (Long-term Goals)
*   **Word Embeddings:** Transition from TF-IDF sparse vectors to dense semantic embeddings using Word2Vec or GloVe to capture deeper contextual meanings.
*   **Transformer Models:** Fine-tune lightweight masked models (like DistilBERT or RoBERTa) using the HuggingFace `transformers` library for state-of-the-art intent classification that thoroughly beats SVM metrics on subtle or sarcastic queries.
*   **Data Augmentation:** Handle imbalanced "flair" classes through techniques like synthetic minority over-sampling or textual augmentation (synonym replacement).
*   **Real-time Dataset Integration:** Pipeline API integration to scrape and append fresh queries to the dataset weekly to prevent dataset drift.
