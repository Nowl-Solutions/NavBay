"""One identical Thai preprocessing function for train and test."""
import re
import unicodedata
import os
from pathlib import Path

os.environ.setdefault("PYTHAINLP_DATA", str(Path(__file__).resolve().parents[1] / ".cache" / "pythainlp"))

from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize

# Negation and sentiment-bearing modifiers must survive stopword removal.
KEEP_WORDS = {"ไม่", "ไม่ได้", "ไม่มี", "ไม่ใช่", "มิ", "ห้าม", "อย่า", "ดี", "มาก", "น้อย"}
STOPWORDS = set(thai_stopwords()) - KEEP_WORDS


def tokenize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"https?://\S+", " ", text)
    return [
        token.strip()
        for token in word_tokenize(text, engine="newmm", keep_whitespace=False)
        if token.strip() not in STOPWORDS
        and any(char.isalpha() for char in token)
    ]
