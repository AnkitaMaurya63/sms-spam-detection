"""Text cleaning used by both training and prediction."""
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()
_URL = re.compile(r"https?://\S+|www\.\S+")
_NUMBER = re.compile(r"\d+")
_NON_ALPHA = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase -> replace urls/numbers with tokens -> strip punctuation
    -> remove stopwords -> stem."""
    text = str(text).lower()
    text = _URL.sub(" urltoken ", text)
    text = _NUMBER.sub(" numtoken ", text)
    text = _NON_ALPHA.sub(" ", text)
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(_stemmer.stem(t) for t in tokens)
