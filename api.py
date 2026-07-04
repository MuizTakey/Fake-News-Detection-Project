from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import logging
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r"C:\Users\Muiz\OneDrive\Desktop\Mini Project Sem 5\Fake_News_Detection\api.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Load Model and Vectorizer
model, tfidf = None, None
dataset_path = r"C:\Users\Muiz\OneDrive\Desktop\Mini Project Sem 5\Fake_News_Detection\cleaned_combined_news_dataset.csv"
try:
    model_path = r"C:\Users\Muiz\OneDrive\Desktop\Mini Project Sem 5\Fake_News_Detection\xgboost_model.pkl"
    tfidf_path = r"C:\Users\Muiz\OneDrive\Desktop\Mini Project Sem 5\Fake_News_Detection\tfidf_vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        logging.info("Model and TF-IDF vectorizer loaded successfully.")
    else:
        logging.warning("Model or vectorizer file not found.")
except Exception as e:
    logging.error(f"Error loading model/vectorizer: {e}")
    model, tfidf = None, None

# Text Preprocessing
def preprocess_text(text):
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        if len(tokens) == 0:
            return text
        return ' '.join(tokens)
    except:
        return text

# Home
@app.route('/')
def home():
    return render_template("index.html")

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        if not news_text.strip():
            return jsonify({'error': 'Text empty. Please provide proper text.'}), 400
        processed_text = preprocess_text(news_text)
        X_new = tfidf.transform([processed_text]).toarray()
        prediction = model.predict(X_new)[0]
        result = 'real' if prediction == 0 else 'fake'
        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Load Dataset
@app.route('/load_dataset', methods=['GET'])
def load_dataset():
    try:
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found.'}), 404
        df = pd.read_csv(dataset_path)
        preview = df.head(10).to_dict(orient='records')
        return jsonify({
            'status': 'success',
            'message': 'Dataset loaded successfully!',
            'columns': list(df.columns),
            'preview': preview
        })
    except Exception as e:
        logging.error(f"Dataset load error: {e}")
        return jsonify({'error': str(e)}), 500

# Selected Features (Highlight PSO-selected features)
@app.route('/selected_features', methods=['GET'])
def selected_features():
    try:
        features = tfidf.get_feature_names_out()
        importances = model.feature_importances_
        feature_dict = {f: float(importances[i]) if i < len(importances) else 0 for i,f in enumerate(features)}
        top_features = dict(sorted(feature_dict.items(), key=lambda x:x[1], reverse=True)[:50])
        return jsonify({'status':'success','features': top_features})
    except Exception as e:
        logging.error(f"Feature load error: {e}")
        return jsonify({'error': str(e)}), 500

# Model Accuracy
@app.route('/model_accuracy', methods=['GET'])
def model_accuracy():
    try:
        df = pd.read_csv(dataset_path)
        if 'label' not in df.columns:
            return jsonify({'error': 'Dataset missing "label" column'}), 400
        y_true = df['label'].apply(lambda x: 0 if x=='real' else 1).values
        texts = df['text'].apply(preprocess_text).tolist()
        X = tfidf.transform(texts).toarray()
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()
        return jsonify({'status':'success','accuracy': float(acc),'confusion_matrix': cm})
    except Exception as e:
        logging.error(f"Accuracy calculation error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    app.run(debug=True, port=5000)
