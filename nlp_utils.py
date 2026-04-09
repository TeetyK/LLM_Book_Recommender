import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Ensure NLTK datasets are available (downloading might be required during first run)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Applies NLP techniques (tokenization, stop words removal, lemmatization)
    to reduce the token count of a user query while retaining the core meaning.
    Supports English texts primarily, useful for RAG queries.
    """
    if not text:
        return ""

    # 1. Lowercase and remove punctuation/special characters
    text = re.sub(r'[^\w\s]', '', text.lower())

    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. Stop words removal
    stop_words = set(stopwords.words('english'))
    # Also add some common Thai stop words if needed, though NLTK focuses on English
    # For a simple approach, we'll just filter English stopwords as LangChain RAG often works well with keywords
    tokens = [word for word in tokens if word not in stop_words]

    # 4. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Return as a string suitable for searching
    return " ".join(lemmatized_tokens)
