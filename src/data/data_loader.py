import pandas as pd
import os
from src.utils.config import get_project_root

def load_data(train_path, val_path, remove_not_propaganda=True):
    """
    Load and preprocess the propaganda dataset.
    
    Args:
        train_path (str): Path to training data
        val_path (str): Path to validation data
        remove_not_propaganda (bool): Whether to remove non-propaganda examples
    
    Returns:
        tuple: (train_df, val_df) pandas DataFrames
    """
    # Convert relative paths to absolute paths
    root_dir = get_project_root()
    train_path = os.path.join(root_dir, train_path)
    val_path = os.path.join(root_dir, val_path)
    
    # Load data
    train_df = pd.read_csv(train_path, sep="\t")
    val_df = pd.read_csv(val_path, sep="\t")
    
    # Rename columns for consistency
    train_df = train_df.rename(columns={'tagged_in_context': 'text'})
    val_df = val_df.rename(columns={'tagged_in_context': 'text'})
    
    # Filter out non-propaganda examples if requested
    if remove_not_propaganda:
        train_df = train_df[train_df['label'] != 'not_propaganda'].reset_index(drop=True)
        val_df = val_df[val_df['label'] != 'not_propaganda'].reset_index(drop=True)
    
    return train_df, val_df 