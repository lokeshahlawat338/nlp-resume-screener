"""
Semantic matching using Sentence-BERT (SBERT).

Converts text into dense vector embeddings that capture meaning,
not just exact words, then compares them via cosine similarity.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # all-MiniLM-L6-v2: a small, fast SBERT model — 384-dim vectors,
        # good tradeoff between speed and accuracy for this use case.
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        """Convert a piece of text into its embedding vector."""
        return self.model.encode(text)

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Return cosine similarity (0 to 1) between two texts'
        embeddings. Higher = more semantically similar.
        """
        emb_a = self.embed(text_a).reshape(1, -1)
        emb_b = self.embed(text_b).reshape(1, -1)
        score = cosine_similarity(emb_a, emb_b)[0][0]
        return float(score)