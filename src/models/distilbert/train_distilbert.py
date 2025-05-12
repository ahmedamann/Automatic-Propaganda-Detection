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

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).mean()}

def train_and_evaluate():
    """Train and evaluate the DistilBERT model."""
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
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(config['distilbert']['model_name'])
    
    def tokenize(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=config['distilbert']['max_length']
        )
    
    # Prepare datasets
    train_dataset = Dataset.from_dict({
        'text': X_train,
        'label': y_train_enc
    }).map(tokenize, batched=True)
    
    val_dataset = Dataset.from_dict({
        'text': X_val,
        'label': y_val_enc
    }).map(tokenize, batched=True)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        config['distilbert']['model_name'],
        num_labels=len(label_encoder.classes_)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=config['distilbert']['epochs'],
        per_device_train_batch_size=config['distilbert']['batch_size'],
        per_device_eval_batch_size=config['distilbert']['batch_size'],
        learning_rate=config['distilbert']['learning_rate'],
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        compute_metrics=compute_metrics
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    # Train model
    trainer.train()
    
    # Save model and tokenizer
    model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\nModel and tokenizer saved to {model_dir}")
    
    # Evaluate
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    print(eval_results)

if __name__ == "__main__":
    train_and_evaluate() 