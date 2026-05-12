from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# ─────────────────────────────────────────────
#  Training Data  (spam=1 / ham=0)
# ─────────────────────────────────────────────
EMAILS = [
    # SPAM
    ("Congratulations! You've won a $1,000 gift card. Click here to claim now!", 1),
    ("URGENT: Your account has been compromised. Verify now to avoid suspension.", 1),
    ("Make money fast! Work from home and earn $5000 per week guaranteed!", 1),
    ("Free iPhone giveaway! You are selected. Act now before offer expires!", 1),
    ("Click here to win a luxury vacation. Limited time offer!", 1),
    ("Buy cheap Viagra online! No prescription needed. Best prices!", 1),
    ("You have been pre-approved for a loan of $50,000. No credit check!", 1),
    ("Double your income working from home. Join millions who already do!", 1),
    ("WINNER! Your email has been chosen for our lottery. Claim your prize!", 1),
    ("Get rich quick! Investment opportunity with 500% returns guaranteed!", 1),
    ("Lose 30 pounds in 30 days with this miracle weight loss pill!", 1),
    ("Hot singles in your area want to meet you tonight!", 1),
    ("Nigerian prince needs your help transferring $10 million. Will reward!", 1),
    ("Your PayPal account is suspended. Click link to restore access now.", 1),
    ("Exclusive deal: Buy now and get 90% discount. Today only!", 1),
    ("You're a lucky winner of our weekly prize draw. Claim your reward!", 1),
    ("Enlarge your presence with our special offer. Discreet shipping.", 1),
    ("FINAL WARNING: Your computer has a virus. Call this number immediately!", 1),
    ("Earn passive income online. No experience needed. Start today!", 1),
    ("Free credit score check! Get yours now with no obligation.", 1),

    # HAM
    ("Hey, are we still on for lunch tomorrow at the usual place?", 0),
    ("Please find attached the report for this quarter's sales figures.", 0),
    ("Can you review my pull request when you get a chance? Thanks!", 0),
    ("The meeting has been rescheduled to 3pm on Friday.", 0),
    ("Happy birthday! Hope you have a wonderful day.", 0),
    ("I've uploaded the slides for tomorrow's presentation to the shared drive.", 0),
    ("Your package has been shipped and will arrive by Thursday.", 0),
    ("Thanks for the feedback on my essay. I'll revise it over the weekend.", 0),
    ("Just a reminder that rent is due on the 1st of the month.", 0),
    ("We're having a team dinner on Friday. Can you make it?", 0),
    ("Your appointment is confirmed for Monday at 10am.", 0),
    ("I found a great recipe for pasta. Want me to send it over?", 0),
    ("The library book you reserved is now available for pickup.", 0),
    ("Can you pick up some groceries on your way home?", 0),
    ("Just checking in — how did your interview go?", 0),
    ("The project deadline has been extended by one week.", 0),
    ("Your subscription renewal is coming up on the 15th.", 0),
    ("I'll be a bit late to the office today — stuck in traffic.", 0),
    ("Great work on the presentation! The client loved it.", 0),
    ("Please complete the onboarding form before your start date.", 0),
]

texts  = [e[0] for e in EMAILS]
labels = [e[1] for e in EMAILS]

# ─────────────────────────────────────────────
#  Train the model
# ─────────────────────────────────────────────
model = Pipeline([
    ("vectorizer", CountVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("classifier", MultinomialNB()),
])
model.fit(texts, labels)

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    proba      = model.predict_proba([text])[0]   # [ham_prob, spam_prob]
    is_spam    = bool(proba[1] > 0.5)
    confidence = round(float(max(proba)) * 100, 1)

    return jsonify({
        "is_spam":    is_spam,
        "label":      "Spam" if is_spam else "Not Spam",
        "confidence": confidence,
        "spam_prob":  round(float(proba[1]) * 100, 1),
        "ham_prob":   round(float(proba[0]) * 100, 1),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
