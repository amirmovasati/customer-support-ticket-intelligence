# region: Setup and Data Loading
# Loads the cleaned dataset produced during EDA.

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "bitext_customer_support_clean.csv"
OUTPUTS_PATH = Path(r"C:\Projects\SupportTicketOutputs")
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN_DATA_FILE)
# endregion


# region: Train/Test Split
# Stratified split so each intent keeps roughly the same proportion in both sets.

X_train, X_test, y_train, y_test = train_test_split(
    df["instruction"],
    df["intent"],
    test_size=0.2,
    random_state=42,
    stratify=df["intent"],
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
# endregion


# region: TF-IDF Vectorization
# Converts text into numeric vectors based on word importance.

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# endregion


# region: Train Baseline Classifier
# A simple, fast Logistic Regression model as our comparison benchmark.

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_vec, y_train)
# endregion


# region: Evaluation
# Measures how well the baseline performs on unseen data.

y_pred = baseline_model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBaseline accuracy: {accuracy:.4f}\n")
print(classification_report(y_test, y_pred))
# endregion


# region: Save Artifacts
# Persists the model and vectorizer so later phases (and comparisons) don't require retraining.

joblib.dump(baseline_model, OUTPUTS_PATH / "baseline_logreg_model.joblib")
joblib.dump(vectorizer, OUTPUTS_PATH / "baseline_tfidf_vectorizer.joblib")
print(f"Saved baseline artifacts to {OUTPUTS_PATH}")
# endregion