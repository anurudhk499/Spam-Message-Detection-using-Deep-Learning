# 📩 Spam Message Detection using Deep Learning

A simple and effective spam detection system built using **TensorFlow/Keras** and deployed with a **Flask web application**.

---

##  Features

* Detects whether a message is **Spam or Ham**
* Uses **Neural Network with Embedding + Dropout**
* Achieves **~98–99% accuracy**
* REST API built using **Flask**
* Interactive frontend using HTML + JavaScript

---

## 🧠 Model Details

* **Dataset**: SMS Spam Collection Dataset

* **Preprocessing**:

  * Lowercasing
  * Removing special characters
  * Tokenization
  * Padding sequences

* **Architecture**:

  * Embedding Layer
  * Dense Layers
  * Dropout (for overfitting control)

* **Techniques Used**:

  * Early Stopping
  * Class Weights (for imbalance handling)

---

## 📊 Performance

* **Accuracy**: ~98–99%
* **Low False Positives & False Negatives**

Example Confusion Matrix:

```
[[964   2]
 [  8 141]]
```

---

## 🗂️ Project Structure

```
project/
│
├── app.py
├── models/
│   ├── spam_model.h5
│   ├── tokenizer.pkl
│
├── templates/
│   └── index.html
│
├── notebook/
│   └── spam_detection.ipynb
│
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

## 📡 API Endpoint

### POST `/predict`

**Request:**

```json
{
  "message": "You won a free prize!"
}
```

**Response:**

```json
{
  "message": "You won a free prize!",
  "probability": 0.99,
  "verdict": "SPAM",
  "threshold": 0.5
}
```

---

## Example

| Message                    | Prediction |
| -------------------------- | ---------- |
| "Win cash now!!!"          | SPAM       |
| "Are you coming to class?" | HAM        |

---

## 📌 Notes

* Ensure `tokenizer.pkl` and `spam_model.h5` are from the **same training session**
* Keep `maxlen` consistent between training and inference

---

##  Future Improvements

* Improve UI with better design
* Deploy using **Render / Vercel / AWS**
* Add real-time message filtering

---
