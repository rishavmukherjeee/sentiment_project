import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import joblib

# Sample data creation
def create_sample_data():
    texts = [
        "I love this product",
        "Great service",
        "Terrible experience",
        "Not worth the money",
        "Amazing quality",
        # Add more examples as needed
    ]
    labels = [1, 1, 0, 0, 1]  # 1 for positive, 0 for negative
    return pd.DataFrame({'text': texts, 'sentiment': labels})

# Model class
class SentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()
    
    def train(self, X, y):
        # Transform text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X)
        # Train the model
        self.model.fit(X_tfidf, y)
    
    def predict(self, text):
        # Transform text and predict
        X_tfidf = self.vectorizer.transform([text])
        return self.model.predict(X_tfidf)[0]
    
    def save(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        # Save both model and vectorizer
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    @classmethod
    def load(cls, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.vectorizer = joblib.load(vectorizer_path)
        return instance

# Training script
def train_and_save_model():
    # Create sample data
    df = create_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    classifier = SentimentClassifier()
    classifier.train(X_train, y_train)
    
    # Save model
    classifier.save()
    
    return classifier

if __name__ == "__main__":
    classifier = train_and_save_model()
    
    # Test prediction
    test_text = "This is a great product"
    prediction = classifier.predict(test_text)
    print(f"Prediction for '{test_text}': {'Positive' if prediction == 1 else 'Negative'}")