def extract_span_and_context(text, bos_token="<BOS>", eos_token="<EOS>"):
    """
    Extract propaganda span and its context from text.
    
    Args:
        text (str): Input text containing propaganda span
        bos_token (str): Beginning of span token
        eos_token (str): End of span token
    
    Returns:
        tuple: (context, span) where context is the surrounding text and span is the propaganda text
    """
    bos_idx = text.find(bos_token)
    eos_idx = text.find(eos_token)
    
    if bos_idx == -1 or eos_idx == -1:
        return text, ""
    
    span = text[bos_idx + len(bos_token):eos_idx].strip()
    context = text.replace(bos_token, "").replace(eos_token, "").strip()
    
    return context, span

def prepare_texts_and_labels(df, bos_token="<BOS>", eos_token="<EOS>"):
    """
    Prepare texts and labels for model training.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'text' and 'label' columns
        bos_token (str): Beginning of span token
        eos_token (str): End of span token
    
    Returns:
        tuple: (texts, labels) where texts is a list of processed texts and labels is a list of labels
    """
    texts, labels = [], []
    
    for _, row in df.iterrows():
        context, span = extract_span_and_context(row['text'], bos_token, eos_token)
        # Combine context and span for better context awareness
        processed_text = f"{context} {span}".strip()
        texts.append(processed_text)
        labels.append(row['label'])
    
    return texts, labels 