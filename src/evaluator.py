"""
Evaluation metrics for reading salience and highlight detection
Implements:
1. Token-level F1 overlap for passage/quote boundary alignment
2. NDCG@K for passage ranking
3. Averaged classification metrics (precision, recall, F1, accuracy)
Handles edge cases: empty strings, zero true positives, ties in ranking, and zero-relevance queries.
"""

import math
import re
from typing import List, Dict

def tokenize_words(text: str) -> List[str]:
    #tokenize into lowercase word tokens
    if not text:
        return []
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

def compute_token_f1(predicted_text: str, reference_text: str) -> Dict[str, float]:
    """
    compute token-level precision, recall, and F1 overlap between two spans.
    handles edge cases: empty prediction, empty reference, identical spans.
    """
    pred_tokens = tokenize_words(predicted_text)
    ref_tokens = tokenize_words(reference_text)

    #edge cases
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    #count matching tokens with multiset intersection
    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    overlap_count = 0
    ref_counts = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    for tok, count in pred_counts.items():
        if tok in ref_counts:
            overlap_count += min(count, ref_counts[tok])

    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }

def compute_dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """compute discounted cumulative gain at rank K."""
    dcg = 0.0
    for idx, rel in enumerate(relevance_scores[:k]):
        rank = idx + 1
        dcg += rel / math.log2(rank + 1)
    return dcg

def compute_ndcg_at_k(ranked_labels: List[int], k: int = 5) -> float:
    #compute normalized discounted cumulative gain at rank K.
    if not ranked_labels or sum(ranked_labels) == 0:
        return 0.0

    actual_dcg = compute_dcg_at_k(ranked_labels, k)
    ideal_labels = sorted(ranked_labels, reverse=True)
    ideal_dcg = compute_dcg_at_k(ideal_labels, k)

    if ideal_dcg == 0.0:
        return 0.0

    return round(actual_dcg / ideal_dcg, 4)

def evaluate_classification(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    #compute standard classification metrics: precision, recall, F1, accuracy
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

if __name__ == "__main__":
    #sanity checks
    s1 = "Whatever our souls are made of, his and mine are the same."
    s2 = "Whatever our souls are made of, his and mine are identical."
    res = compute_token_f1(s1, s2)
    print("Token F1 test:", res)
    ndcg = compute_ndcg_at_k([1, 0, 1, 0, 0], k=3)
    print("NDCG@3 test:", ndcg)
