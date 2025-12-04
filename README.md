# AI FAQ Chatbot (StaxTech) - Lightweight

This is a simple FAQ chatbot built for the StaxTech internship. It uses a TF-IDF vectorizer
and cosine similarity over a small FAQ dataset to return the best matching answer.

## Files
- `faqs.json` : Sample FAQ data (questions + answers)
- `model_train.py` : Trains TF-IDF and saves artifacts (vectorizer.pkl, faq_matrix.pkl, faqs_saved.pkl)
- `app.py` : Flask application that serves the chat UI and the /ask API
- `templates/index.html` : Simple chat UI
- `static/style.css` : Styling for UI
- `requirements.txt` : Python dependencies

## How to run
1. Create a Python virtual environment and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate     # Linux / macOS
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

2. Train the model artifacts:
   ```
   python model_train.py
   ```

3. Run the Flask app:
   ```
   python app.py
   ```

4. Open `http://localhost:5000` in your browser.

## Customize
- Edit `faqs.json` to add or modify questions and answers.
- Re-run `python model_train.py` whenever you change FAQs.

## Notes
- This is a simple, small-scale chatbot suitable for demos and internship projects.
- For production, consider using more advanced NLP models or services (Rasa, Dialogflow, OpenAI).