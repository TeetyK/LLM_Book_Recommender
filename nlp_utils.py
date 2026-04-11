import re

def preprocess_text(text: str) -> str:
    """
    Applies basic cleanup techniques to a user query.
    Modern embedding models (like HuggingFace) perform better with natural sentences,
    so heavy NLP (tokenization, lemmatization) is avoided.
    """
    if not text:
        return ""

    # 1. Strip extra whitespaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Return the clean natural text suitable for embedding models
    return text
