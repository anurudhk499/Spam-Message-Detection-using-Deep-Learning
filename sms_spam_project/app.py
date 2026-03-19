from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ---------------------------
# Load Model and Tokenizer
# ---------------------------
model = tf.keras.models.load_model("models/spam_model.h5")
tokenizer = pickle.load(open("models/tokenizer.pkl", "rb"))

MAX_LEN = 150


# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = " ".join(text.split())

    return text


# ---------------------------
# Prediction Function
# ---------------------------
def predict_spam(text):

    text = clean_text(text)  # Only clean once
    # Remove the second clean_text call

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    prob = float(model.predict(padded, verbose=0)[0][0])

    return prob

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({
                "error": "No message provided"
            }), 400

        message = data["message"].strip()

        if message == "":
            return jsonify({
                "error": "Empty message"
            }), 400

        prob = predict_spam(message)

        threshold = 0.4

        verdict = "SPAM" if prob >= threshold else "HAM"

        response = {
            "message": message,
            "probability": prob,
            "verdict": verdict,
            "threshold": threshold
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)