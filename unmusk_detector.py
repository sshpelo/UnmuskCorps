import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
nltk.download('stopwords')

# 1. Data Loading
def load_data():
    try:
        # Replace with your dataset path (e.g., Kaggle fake review dataset)
        df = pd.read_csv('amazon_reviews.csv')
        df = df[['text', 'label']]  # Assume columns: text, label (0=genuine, 1=fake)
        return df
    except FileNotFoundError:
        print("Dataset not found. Please download from:")
        print("https://www.kaggle.com/datasets/saurabhshahane/fake-amazon-reviews")
        exit()

# 2. Text Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# 3. Feature Engineering
def extract_features(df):
    # Basic text features
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    tfidf_features = tfidf.fit_transform(df['clean_text'])
    
    return tfidf_features, df[['char_count', 'word_count', 'exclamation_count']]

# 4. Fake Review Detection Model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# 5. Visualization
def plot_results(y_test, y_pred, feature_names, model):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

    # Feature Importance (Top 20)
    if hasattr(model, 'coef_'):
        coefficients = pd.Series(model.coef_[0], index=feature_names)
        plt.figure(figsize=(10,6))
        coefficients.sort_values(ascending=False)[:20].plot(kind='bar')
        plt.title('Top 20 Important Features')
        plt.show()

def main():
    # Load and prepare data
    df = load_data()
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Feature extraction
    tfidf_features, manual_features = extract_features(df)
    X = np.hstack([tfidf_features.toarray(), manual_features])
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize
    feature_names = list(vectorizer.get_feature_names_out()) + list(manual_features.columns)
    plot_results(y_test, y_pred, feature_names, model)

if __name__ == "__main__":
    main()
