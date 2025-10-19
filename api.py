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

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'D:\sem_5\fake_news_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# -------------------- Load Model and Vectorizer --------------------
model, tfidf = None, None
try:
    model_path = r'D:\sem_5\xgboost_model.pkl'
    tfidf_path = r'D:\sem_5\tfidf_vectorizer.pkl'

    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        logging.info("Model and TF-IDF vectorizer loaded successfully.")
    else:
        logging.warning("Model or vectorizer file not found. Flask will run, but predictions won't work.")
except Exception as e:
    logging.error(f"Error loading model/vectorizer: {e}")
    model, tfidf = None, None


# -------------------- Text Preprocessing --------------------
def preprocess_text(text):
    """Clean and preprocess input text."""
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return ''


# -------------------- Home Route --------------------
@app.route('/')
def home():
    return render_template("index.html")


# -------------------- Prediction Route --------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Predict if the given news article is fake or real."""
    try:
        if model is None or tfidf is None:
            return jsonify({'error': 'Model or vectorizer not loaded.'}), 500

        data = request.get_json()
        news_text = data.get('text', '')
        if not news_text.strip():
            return jsonify({'error': 'No text provided.'}), 400

        processed_text = preprocess_text(news_text)
        if not processed_text.strip():
            return jsonify({'error': 'Text empty after preprocessing.'}), 400

        X_new = tfidf.transform([processed_text]).toarray()
        prediction = model.predict(X_new)[0]
        result = 'real' if prediction == 0 else 'fake'

        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# -------------------- Dataset Preview Route --------------------
dataset_path = r'D:\sem_5\cleaned_combined_news_dataset.csv'

@app.route('/load_dataset', methods=['GET'])
def load_dataset():
    """Return the first 10 rows of the dataset."""
    try:
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset file not found.'}), 404

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


# -------------------- Selected Features Route --------------------
@app.route('/selected_features', methods=['GET'])
def selected_features():
    """Return the top features used in classification."""
    try:
        if tfidf is None:
            return jsonify({'error': 'Vectorizer not loaded.'}), 500

        features = tfidf.get_feature_names_out()
        limited_features = features[:50].tolist()  # top 50 for readability
        return jsonify({
            'status': 'success',
            'features': limited_features,
            'count': len(features)
        })
    except Exception as e:
        logging.error(f"Feature load error: {e}")
        return jsonify({'error': str(e)}), 500


# -------------------- Main --------------------
if __name__ == '__main__':
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        logging.warning(f"Some NLTK resources could not be downloaded: {e}")

    app.run(debug=True, port=5000)
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

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'D:\sem_5\fake_news_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# -------------------- Load Model and Vectorizer --------------------
model, tfidf = None, None
try:
    model_path = r'D:\sem_5\xgboost_model.pkl'
    tfidf_path = r'D:\sem_5\tfidf_vectorizer.pkl'

    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        logging.info("Model and TF-IDF vectorizer loaded successfully.")
    else:
        logging.warning("Model or vectorizer file not found. Flask will run, but predictions won't work.")
except Exception as e:
    logging.error(f"Error loading model/vectorizer: {e}")
    model, tfidf = None, None


# -------------------- Text Preprocessing --------------------
def preprocess_text(text):
    """Clean and preprocess input text."""
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return ''


# -------------------- Home Route --------------------
@app.route('/')
def home():
    return render_template("index.html")


# -------------------- Prediction Route --------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Predict if the given news article is fake or real."""
    try:
        if model is None or tfidf is None:
            return jsonify({'error': 'Model or vectorizer not loaded.'}), 500

        data = request.get_json()
        news_text = data.get('text', '')
        if not news_text.strip():
            return jsonify({'error': 'No text provided.'}), 400

        processed_text = preprocess_text(news_text)
        if not processed_text.strip():
            return jsonify({'error': 'Text empty after preprocessing.'}), 400

        X_new = tfidf.transform([processed_text]).toarray()
        prediction = model.predict(X_new)[0]
        result = 'real' if prediction == 0 else 'fake'

        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# -------------------- Dataset Preview Route --------------------
dataset_path = r'D:\sem_5\cleaned_combined_news_dataset.csv'

@app.route('/load_dataset', methods=['GET'])
def load_dataset():
    """Return the first 10 rows of the dataset."""
    try:
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset file not found.'}), 404

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


# -------------------- Selected Features Route --------------------
@app.route('/selected_features', methods=['GET'])
def selected_features():
    """Return the top features used in classification."""
    try:
        if tfidf is None:
            return jsonify({'error': 'Vectorizer not loaded.'}), 500

        features = tfidf.get_feature_names_out()
        limited_features = features[:50].tolist()  # top 50 for readability
        return jsonify({
            'status': 'success',
            'features': limited_features,
            'count': len(features)
        })
    except Exception as e:
        logging.error(f"Feature load error: {e}")
        return jsonify({'error': str(e)}), 500


# -------------------- Main --------------------
if __name__ == '__main__':
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        logging.warning(f"Some NLTK resources could not be downloaded: {e}")

    app.run(debug=True, port=5000)
