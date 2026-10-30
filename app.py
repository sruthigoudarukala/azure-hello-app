import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "spam_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE, "tfidf_vectorizer.pkl"))

ENGLISH_STOP_WORDS = frozenset([
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","as","at","be","because","been","before","being","below","between",
    "both","but","by","could","did","do","does","doing","down","during","each",
    "few","for","from","further","had","has","have","having","he","her","here",
    "hers","herself","him","himself","his","how","i","if","in","into","is","it",
    "its","itself","just","me","more","most","my","myself","no","nor","not",
    "now","of","off","on","once","only","or","other","our","ours","ourselves",
    "out","over","own","same","she","should","so","some","such","than","that",
    "the","their","theirs","them","themselves","then","there","these","they",
    "this","those","through","to","too","under","until","up","very","was",
    "we","were","what","when","where","which","while","who","whom","why",
    "will","with","you","your","yours","yourself","yourselves"
])


def clean_text(text):
    text = text.lower()
    words = text.split()
    clean_words = [w for w in words if w.isalpha()]
    return " ".join(clean_words)


def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def predict_message(message):
    cleaned = clean_text(message)
    cleaned = remove_stopwords(cleaned)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Spam" if prediction == 1 else "Ham"


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMS Spam Checker</title>
<style>
body{font-family:Arial,sans-serif;max-width:560px;margin:40px auto;padding:0 16px;color:#222}
textarea{width:100%;height:110px;padding:10px;font-size:16px;box-sizing:border-box}
button{margin-top:10px;padding:10px 20px;font-size:16px;cursor:pointer}
#result{margin-top:20px;font-size:22px;font-weight:bold}
</style>
</head>
<body>
<h2>SMS Spam Checker</h2>
<p>Type an SMS message and check if it is Spam or Ham.</p>
<textarea id="msg" placeholder="Example: You won a free prize, call now"></textarea>
<br>
<button onclick="check()">Check</button>
<div id="result"></div>
<script>
async function check(){
  var text = document.getElementById('msg').value;
  var out = document.getElementById('result');
  out.textContent = 'Checking...';
  try {
    var r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    });
    var d = await r.json();
    out.textContent = 'Result: ' + d.prediction;
  } catch (e) {
    out.textContent = 'Something went wrong. Try again.';
  }
}
</script>
</body>
</html>"""


@app.route("/")
def home():
    return PAGE


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    return jsonify({"message": text, "prediction": predict_message(text)})
