import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import bisect


nltk.download('stopwords')


def preprocess_text(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    remove_extra_symb: list[str] = re.sub(r'[^\w^\s]+', '', str(text)).lower().split()
    return ' '.join([stemmer.stem(w) for w in remove_extra_symb if w not in stop_words])


def classify_revenue(revenue):
    return bisect.bisect_left([10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000] , revenue)
