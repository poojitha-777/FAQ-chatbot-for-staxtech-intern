# model_train.py
# Train a TF-IDF vectorizer on FAQ questions and save the vectorizer + matrix + faqs to disk.
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

project_dir = Path(__file__).parent

faqs_path = project_dir / "faqs.json"
with open(faqs_path, "r", encoding="utf-8") as f:
    faqs = json.load(f)

questions = [q["question"] for q in faqs]

# Create vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
X = vectorizer.fit_transform(questions)

# Save vectorizer and matrix and faqs
with open(project_dir / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open(project_dir / "faq_matrix.pkl", "wb") as f:
    pickle.dump(X, f)

with open(project_dir / "faqs_saved.pkl", "wb") as f:
    pickle.dump(faqs, f)

print("Training complete. Saved vectorizer.pkl, faq_matrix.pkl, faqs_saved.pkl")