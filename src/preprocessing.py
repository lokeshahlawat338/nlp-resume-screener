"""
Dual-mode text preprocessing.

- clean_only: light cleaning for SBERT (preserves sentence structure)
- full: lemmatization + stopword removal for TF-IDF (bag-of-words)
"""

import re
import spacy


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def clean_only(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x20-\x7E\n]", "", text)
        return text.strip()

    def full(self, text: str) -> str:
        text = self.clean_only(text)
        doc = self.nlp(text.lower())

        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.lemma_) > 1
        ]
        return " ".join(tokens)