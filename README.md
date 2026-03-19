# 📩 Spam Message Detection using CNN + BiLSTM

A deep learning-based spam detection system that classifies SMS messages as **Spam or Ham** using a hybrid **CNN + Bidirectional LSTM (BiLSTM)** model. The project is deployed using a **Flask web application** with a simple frontend.

---

## Features

* Detects spam messages with high accuracy (~98–99%)
* Uses **CNN + BiLSTM hybrid architecture**
* Handles class imbalance using **class weights**
* Prevents overfitting using **Dropout & Early Stopping**
* REST API built with **Flask**
* Simple frontend using **HTML + JavaScript**

---

## 🧠 Model Architecture (CNN + BiLSTM)

The model combines **CNN** for feature extraction and **BiLSTM** for understanding context.

### 🔹 Architecture Flow

```
Input Text
   ↓
Text Cleaning (lowercase, remove symbols)
   ↓
Tokenization + Padding
   ↓
Embedding Layer
   ↓
Conv1D (CNN)
   ↓
MaxPooling1D
   ↓
Bidirectional LSTM
   ↓
Dense Layer
   ↓
Dropout
   ↓
Output Layer (Sigmoid)
```

---

### 🔹 Layer Explanation

* **Embedding Layer**
  Converts words into dense vectors capturing semantic meaning

* **Conv1D (CNN)**
  Extracts local features like keywords ("free", "win", "offer")

* **MaxPooling**
  Reduces dimensionality and keeps important features

* **Bidirectional LSTM**
  Captures context from both directions (past & future words)

* **Dense Layer**
  Learns higher-level patterns

* **Dropout**
  Prevents overfitting

* **Output Layer (Sigmoid)**
  Outputs probability of spam (0 to 1)

---

### 🔹 Why CNN + BiLSTM?

* CNN captures **important phrases**
* BiLSTM captures **sequence and context**
* Combination improves **accuracy and generalization**

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
Spam-Message-Detection-using-Deep-Learning
/
│
├── app.py
├── README.md
├── requirements.txt
├── models/
│   ├── spam_model.h5
│   ├── tokenizer.pkl
├── templates/
│   └── index.html
├──  dataset
│   └── spam.csv
├──  spam_detection.ipynb
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

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 📡 API Endpoint

### POST `/predict`

#### Request:

```json
{
  "message": "You won a free prize!"
}
```

#### Response:

```json
{
  "message": "You won a free prize!",
  "probability": 0.99,
  "verdict": "SPAM",
  "threshold": 0.4
}
```

---

## Example Predictions

| Message                    | Output |
| -------------------------- | ------ |
| "Win cash now!!!"          | SPAM   |
| "Are you coming to class?" | HAM    |

---

## 📌 Notes

* `tokenizer.pkl` and `spam_model.h5` must be from the **same training session**
* Keep `maxlen` consistent during training and inference
* Custom threshold (`0.4`) used for better spam sensitivity

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Flask
* NumPy, Pandas
* Scikit-learn

---
