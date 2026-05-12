# Spam Email Detector

A machine learning web application that classifies emails as **Spam** or **Not Spam** using Naive Bayes.

Built with Python (Flask) for the backend and plain HTML/CSS/JS for the frontend.

---

## Tech Stack
- **ML Model** — Naive Bayes (scikit-learn)
- **Backend** — Python, Flask
- **Frontend** — HTML, CSS, JavaScript

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the server**
```bash
python app.py
```

**3. Open the app**

Double-click `index.html` in your file explorer.

---

## How It Works

1. User types an email/message in the text box
2. Frontend sends it to the Flask API
3. The Naive Bayes model predicts Spam or Not Spam
4. Result is displayed with a confidence percentage
