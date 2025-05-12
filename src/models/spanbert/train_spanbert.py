import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import load_config
from src.data.data_loader import load_data
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import f1_score

def convert_to_bio_for_span_detection(sentence, label):
    """
    Convert sentence into tokens and BIO labels for classification
    """
    # remove BOS and EOS tags and tokenize
    tokens = sentence.replace('<BOS>', ' <BOS> ').replace('<EOS>', ' <EOS> ').split()

    span_start = tokens.index('<BOS>')
    span_end = tokens.index('<EOS>')

    # Remove the markers
    tokens = [t for t in tokens if t not in ('<BOS>', '<EOS>')]
    bio_labels = ['O'] * len(tokens)
    bio_labels[span_start] = 'B-PROP'
    
    for i in range(span_start + 1, span_end - 1):
        bio_labels[i] = 'I-PROP'

    return tokens, bio_labels

def process_data_for_bio(dataframe):
    """Process data into BIO format for token classification."""
    all_tokens = []
    all_labels = []

    for label, sentence in dataframe.values:
        tokens, bio_tags = convert_to_bio_for_span_detection(sentence, label)
        all_tokens.append(tokens)
        all_labels.append(bio_tags)

    return all_tokens, all_labels

def compute_metrics(p):
    """Compute sequence-level F1 score for token classification."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_labels = []
        true_labels_seq = []
        for p_i, l_i in zip(pred_seq, label_seq):
            if l_i != -100:
                pred_labels.append(id2label[p_i])
                true_labels_seq.append(id2label[l_i])
        true_preds.append(pred_labels)
        true_labels.append(true_labels_seq)

    # Sequence-level (seqeval)
    seq_f1 = f1_score(true_labels, true_preds)

    return {
        "seq_f1": seq_f1,
    }

def tokenize_and_align_labels(examples):
    """Tokenize inputs and align labels for token classification."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        max_length=config['spanbert']['max_length'],
        truncation=True,
    )

    labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_seq[word_idx])
            else:
                # for the subword, convert B to I or repeat label
                label_name = id2label[label_seq[word_idx]]
                if label_name.startswith("B-"):
                    label_name = "I-" + label_name[2:]
                label_ids.append(label2id[label_name])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train_and_evaluate():
    """Train and evaluate the SpanBERT model."""
    global tokenizer, config, id2label, label2id
    
    # Load configuration
    config = load_config()
    
    # Load and preprocess data
    train_df, val_df = load_data(
        config['data']['train_path'],
        config['data']['val_path'],
        config['preprocessing']['remove_not_propaganda']
    )
    
    # Process data into BIO format
    tokens_train, tags_train = process_data_for_bio(train_df)
    tokens_val, tags_val = process_data_for_bio(val_df)
    
    # Create label mappings
    all_labels_set = sorted(set(tag for seq in tags_train for tag in seq))
    label2id = {label: i for i, label in enumerate(all_labels_set)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['spanbert']['model_name'])
    
    # Prepare datasets
    train_dataset = Dataset.from_dict({
        "tokens": tokens_train,
        "labels": [[label2id[t] for t in seq] for seq in tags_train]
    })
    
    val_dataset = Dataset.from_dict({
        "tokens": tokens_val,
        "labels": [[label2id[t] for t in seq] for seq in tags_val]
    })
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)
    
    # Initialize model
    model = AutoModelForTokenClassification.from_pretrained(
        config['spanbert']['model_name'],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Initialize data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=config['spanbert']['epochs'],
        per_device_train_batch_size=config['spanbert']['batch_size'],
        per_device_eval_batch_size=config['spanbert']['batch_size'],
        learning_rate=config['spanbert']['learning_rate'],
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='seq_f1',
        greater_is_better=True,
        weight_decay=config['spanbert']['weight_decay'],
        logging_steps=100
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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