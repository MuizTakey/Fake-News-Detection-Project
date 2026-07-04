import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paths
model_path = r"C:\Users\Muiz\OneDrive\Desktop\Mini Project Sem 5\Fake_News_Detection\xgboost_model.pkl"
tfidf_path = r"C:\Users\Muiz\OneDrive\Desktop\Mini Project Sem 5\Fake_News_Detection\tfidf_vectorizer.pkl"
dataset_path = r"C:\Users\Muiz\OneDrive\Desktop\Mini Project Sem 5\Fake_News_Detection\cleaned_combined_news_dataset.csv"

# Load model & vectorizer
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)

# Load dataset
df = pd.read_csv(dataset_path)

# Assuming your dataset has 'text' and 'label' columns
X = tfidf.transform(df['text'].astype(str))
y_true = df['label'].values  # 0 for real, 1 for fake

# Predict
y_pred = model.predict(X)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%\n")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
