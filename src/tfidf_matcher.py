"""
TF-IDF baseline matcher.

A simpler "bag-of-words" comparison used as a baseline to evaluate
how much SBERT's semantic understanding actually helps.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFMatcher:
    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Return cosine similarity (0 to 1) between two texts based on
        TF-IDF vectors (pure word-overlap weighting, no semantic meaning).
        """
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        except ValueError:
            # Happens if both texts are empty / all-stopwords after cleaning
            return 0.0

        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(score)