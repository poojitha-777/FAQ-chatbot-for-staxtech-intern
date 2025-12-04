# app.py
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder="static", template_folder="templates")
project_dir = Path(__file__).parent

# Load model artifacts
try:
    with open(project_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(project_dir / "faq_matrix.pkl", "rb") as f:
        faq_matrix = pickle.load(f)
    with open(project_dir / "faqs_saved.pkl", "rb") as f:
        faqs = pickle.load(f)
except Exception as e:
    vectorizer = None
    faq_matrix = None
    faqs = []
    print("Model artifacts not found. Run 'python model_train.py' first.", e)


def get_best_answer(user_query, top_k=1):
    q_lower = user_query.strip().lower()

    greetings = ["hi", "hello", "hey", "namaste"]
    if q_lower in greetings:
        return "🤖 Hello! How can I assist you today?"

    if vectorizer is None:
        return "⚠️ Model not trained. Please run 'python model_train.py'."

    q_vec = vectorizer.transform([user_query])
    sims = cosine_similarity(q_vec, faq_matrix).flatten()
    best_idx = sims.argmax()
    best_score = float(sims[best_idx])

    if best_score < 0.08:
        return "❓ Sorry, I couldn't find an answer. Please try asking differently."

    # Add emoji to every answer
    return "✨ " + faqs[best_idx]["answer"] + f" (match={best_score:.2f})"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    q = data.get("question", "")
    a = get_best_answer(q)
    return jsonify({"answer": a})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
