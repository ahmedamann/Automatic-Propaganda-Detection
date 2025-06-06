# Automatic Propaganda Detection

This project implements three different approaches for detecting propaganda in text:
1. TF-IDF + Multinomial Naive Bayes classifier
2. DistilBERT transformer model
3. SpanBERT model for propaganda span detection

## Features

- **Text Preprocessing**:
  - Span extraction using <BOS> and <EOS> markers
  - Context-aware text processing
  - Removal of non-propaganda examples (configurable)

- **Naive Bayes Model**:
  - TF-IDF vectorization with configurable parameters
  - Multinomial Naive Bayes classification
  - Model and vectorizer persistence

- **DistilBERT Model**:
  - Fine-tuned DistilBERT transformer
  - Configurable training parameters
  - Automatic model checkpointing
  - Evaluation metrics tracking

- **SpanBERT Model**:
  - Token-level propaganda span detection
  - BIO tagging scheme for span identification
  - Sequence-level F1 score evaluation
  - Early stopping with patience
  - Automatic model checkpointing

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the models:
   - Edit `config/config.yaml` to adjust model parameters
   - Key parameters include:
     - Data paths
     - Preprocessing options
     - Model-specific hyperparameters

2. Train the Naive Bayes model:
```bash
python src/models/nb/train_nb.py
```

3. Train the DistilBERT model:
```bash
python src/models/distilbert/train_distilbert.py
```

4. Train the SpanBERT model:
```bash
python src/models/spanbert/train_spanbert.py
```

## Model Outputs

- **Naive Bayes**:
  - Saves model and vectorizer to `src/models/nb/saved_models/`
  - Outputs classification report with precision, recall, and F1-score

- **DistilBERT**:
  - Saves model checkpoints to `src/models/distilbert/saved_models/`
  - Logs training metrics to `src/models/distilbert/logs/`
  - Outputs evaluation metrics including accuracy

- **SpanBERT**:
  - Saves model checkpoints to `src/models/spanbert/saved_models/`
  - Logs training metrics to `src/models/spanbert/logs/`
  - Outputs sequence-level F1 score for span detection

## Configuration

The `config.yaml` file allows you to configure:
- Data paths and preprocessing options
- TF-IDF vectorizer parameters
- DistilBERT training parameters
- SpanBERT training parameters
- Model saving and logging options

## Development

- Original notebooks are preserved in the `notebooks/` directory for reference
- All paths are configurable through `config.yaml`
- Shared preprocessing code in `src/features/text_processor.py` 