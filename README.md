# 📰 Optimized Fake News Detection using PSO and XGBoost

## 📌 Overview

This project is an intelligent **Fake News Detection System** that combines **Particle Swarm Optimization (PSO)** and **XGBoost** to improve the accuracy and efficiency of fake news classification.

The system preprocesses news articles using **Natural Language Processing (NLP)** techniques, converts the text into numerical features using **TF-IDF**, and applies **Particle Swarm Optimization (PSO)** to select the most relevant features. The optimized features are then classified using the **XGBoost** algorithm.

A user-friendly **Flask-based web application** built with **HTML, CSS, and JavaScript** allows users to upload the dataset, perform feature selection, and predict whether a news article is **Real** or **Fake** in real time.

---

## 🎯 Features

- 📰 Fake News Detection
- ⚡ TF-IDF Feature Extraction
- 🧠 PSO-based Feature Selection
- 🚀 XGBoost Classifier
- 📊 Model Accuracy Display
- 🔍 Highlighted Important Features
- 🌐 Interactive Flask Web Interface
- 📂 Dataset Preview
- 📈 Real-time Prediction

---

## 🏗️ System Workflow

```
Dataset
   │
   ▼
Data Preprocessing
(Tokenization, Stopword Removal,
Punctuation Removal, Lemmatization)
   │
   ▼
TF-IDF Feature Extraction
   │
   ▼
Particle Swarm Optimization (PSO)
(Feature Selection)
   │
   ▼
XGBoost Classifier
   │
   ▼
Prediction
(Real / Fake)
   │
   ▼
Flask Web Application
```

---

## 🛠️ Technologies Used

### Programming Languages
- Python
- HTML
- CSS
- JavaScript

### Framework
- Flask

### Machine Learning
- XGBoost
- Scikit-learn
- Particle Swarm Optimization (PSO)

### NLP
- NLTK
- TF-IDF Vectorizer

### Libraries
- Pandas
- NumPy
- Joblib
- Pickle

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── api.py
├── train_model.py
├── templates/
│   └── index.html
├── static/
│
├── models/
│   ├── xgboost_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── selected_features.pkl
│
├── dataset/
│   └── cleaned_combined_news_dataset.csv
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
```

```bash
cd Fake-News-Detection
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python api.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## 📖 How It Works

1. Load the news dataset.
2. The text is preprocessed using NLP techniques.
3. TF-IDF converts the text into numerical vectors.
4. PSO selects the most informative features.
5. XGBoost predicts whether the news is **Real** or **Fake**.
6. The result is displayed through an interactive Flask web interface.

---

## 📊 Model Performance

| Metric | Value |
|---------|-------|
| Accuracy | **99.71%** |
| Feature Selection | Particle Swarm Optimization |
| Classifier | XGBoost |

---

## 📷 Application Features

- Load Dataset
- PSO Feature Selection
- Real-Time Fake News Prediction
- Feature Highlighting
- Accuracy Display
- Modern Interactive GUI

---

## 🚀 Future Improvements

- Support for multilingual fake news detection.
- Integrate Explainable AI (SHAP/LIME) for prediction explanation.
- Detect fake news from images and videos.
- Develop a browser extension for real-time fact checking.
- Deploy the application on cloud platforms.
- Build Android and iOS mobile applications.

---

## 👨‍💻 Developed By

**Muiz Musharrif Takey**
**Owais Arif Batte**
**Rahil Mazgaonkar**
Mini Project in sem 5 of T.E. Computer Science & Engineering (Artificial Intelligence & Machine Learning)

---

## 📜 License

This project is developed for educational and research purposes.
