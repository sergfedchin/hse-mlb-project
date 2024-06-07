import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import bisect
import numpy as np


nltk.download('stopwords')
CLASS_BORDERS = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000]
CLASS_BORDERS_STR = ["10.000", "100.000", "1.000.000", "10.000.000", "50.000.000", "100.000.000", "500.000.000", "1.000.000.000", "inf"]


def preprocess_text(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    remove_extra_symb: list[str] = re.sub(r'[^\w^\s]+', '', str(text)).lower().split()
    return ' '.join([stemmer.stem(w) for w in remove_extra_symb if w not in stop_words])


def classify_revenue(revenue: int) -> int:
    return bisect.bisect_left(CLASS_BORDERS, revenue)


def declassify_revenue(i: int) -> str:
    return f"${CLASS_BORDERS_STR[i - 1] if i > 0 else 0} - ${CLASS_BORDERS_STR[i]}"


if __name__ == '__main__':
    print(declassify_revenue(classify_revenue(100)), declassify_revenue(classify_revenue(10_000)), declassify_revenue(classify_revenue(10_001)), declassify_revenue(classify_revenue(1_000_000_001)))
