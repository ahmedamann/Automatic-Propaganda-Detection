import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import load_config
from src.data.data_loader import load_data
from src.features.text_processor import prepare_texts_and_labels

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

def train_and_evaluate():
    """Train and evaluate the Naive Bayes model."""
    # Load configuration
    config = load_config()
    
    # Load and preprocess data
    train_df, val_df = load_data(
        config['data']['train_path'],
        config['data']['val_path'],
        config['preprocessing']['remove_not_propaganda']
    )
    
    # Prepare texts and labels
    X_train, y_train = prepare_texts_and_labels(
        train_df,
        config['preprocessing']['bos_token'],
        config['preprocessing']['eos_token']
    )
    X_val, y_val = prepare_texts_and_labels(
        val_df,
        config['preprocessing']['bos_token'],
        config['preprocessing']['eos_token']
    )
    
    # Initialize and fit vectorizer
    vectorizer = TfidfVectorizer(
        max_features=config['nb']['vectorizer']['max_features'],
        ngram_range=tuple(config['nb']['vectorizer']['ngram_range'])
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Train model
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_val_vec)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save model and vectorizer
    model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(clf, os.path.join(model_dir, 'nb_model.joblib'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
    print(f"\nModel and vectorizer saved to {model_dir}")

if __name__ == "__main__":
    train_and_evaluate() 