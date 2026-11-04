"""
Baseline Model: TF-IDF Feature Extractor + Logistic Regression Salience Classifier.
Implements unigram/bigram tokenization, IDF weighting, and logistic classification with L2 regularization.
"""

import math
import re
import numpy as np
from typing import List, Dict, Tuple

class SimpleTfidfVectorizer:
    def __init__(self, min_df: int = 1, max_features: int = 500, ngram_range: Tuple[int, int] = (1, 2)):
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_ = None

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return tokens

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        all_grams = []
        n_min, n_max = self.ngram_range
        for n in range(n_min, n_max + 1):
            for i in range(len(tokens) - n + 1):
                all_grams.append(" ".join(tokens[i:i+n]))
        return all_grams

    def fit(self, texts: List[str]):
        doc_counts = {}
        n_docs = len(texts)
        for t in texts:
            tokens = self._tokenize(t)
            ngrams = set(self._extract_ngrams(tokens))
            for gram in ngrams:
                doc_counts[gram] = doc_counts.get(gram, 0) + 1

        # Filter by min_df
        filtered = {k: v for k, v in doc_counts.items() if v >= self.min_df}
        # Sort by frequency and truncate to max_features
        sorted_terms = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(sorted_terms)}

        # Compute smoothed IDF: log((1 + n_docs) / (1 + doc_count)) + 1
        n_terms = len(self.vocabulary_)
        self.idf_ = np.zeros(n_terms)
        for term, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log((1 + n_docs) / (1 + doc_counts[term])) + 1.0

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        n_docs = len(texts)
        n_terms = len(self.vocabulary_)
        matrix = np.zeros((n_docs, n_terms))

        for row_idx, t in enumerate(texts):
            tokens = self._tokenize(t)
            ngrams = self._extract_ngrams(tokens)
            for gram in ngrams:
                if gram in self.vocabulary_:
                    matrix[row_idx, self.vocabulary_[gram]] += 1.0

        # Multiply by IDF and L2 normalize
        matrix = matrix * self.idf_
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        return matrix

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


class LogisticRegressionBaseline:
    def __init__(self, lr: float = 0.5, l2_reg: float = 0.01, epochs: int = 150):
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Class weighting to handle imbalance
        pos_weight = (n_samples - np.sum(y)) / max(np.sum(y), 1.0)

        for _ in range(self.epochs):
            scores = np.dot(X, self.weights) + self.bias
            preds = self._sigmoid(scores)
            
            # Weighted loss gradient
            sample_weights = np.where(y == 1, pos_weight, 1.0)
            errors = (preds - y) * sample_weights

            dw = (np.dot(X.T, errors) / n_samples) + (self.l2_reg * self.weights)
            db = np.sum(errors) / n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.dot(X, self.weights) + self.bias
        return self._sigmoid(scores)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

