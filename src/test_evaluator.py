"""
Evaluator unit test.
Verifies token-level F1 and NDCG@K scoring procedures against exact analytical calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
from evaluator import compute_token_f1, compute_ndcg_at_k

def run_hand_checked_evaluator_test():
    log_lines = []
    log_lines.append("=== EVALUATOR VERIFICATION TEST ===")
    log_lines.append("Team: Alley Cats")
    log_lines.append("Testing Module: src/evaluator.py\n")

    #test case 1: token-level F1 overlap
    ref_text = "It is a truth universally acknowledged"
    pred_text = "a truth universally acknowledged that a man"

    expected_prec = 4.0 / 7.0
    expected_rec = 4.0 / 6.0
    expected_f1 = (2.0 * expected_prec * expected_rec) / (expected_prec + expected_rec)

    computed = compute_token_f1(pred_text, ref_text)

    log_lines.append("Test Case 1: Token-Level F1 Overlap")
    log_lines.append(f"  Reference text: '{ref_text}'")
    log_lines.append(f"  Predicted text: '{pred_text}'")
    log_lines.append(f"  Expected Precision: {expected_prec:.4f} | Computed: {computed['precision']:.4f}")
    log_lines.append(f"  Expected Recall:    {expected_rec:.4f} | Computed: {computed['recall']:.4f}")
    log_lines.append(f"  Expected F1:        {expected_f1:.4f} | Computed: {computed['f1']:.4f}")

    assert abs(computed['precision'] - round(expected_prec, 4)) < 1e-4, "Precision mismatch"
    assert abs(computed['recall'] - round(expected_rec, 4)) < 1e-4, "Recall mismatch"
    assert abs(computed['f1'] - round(expected_f1, 4)) < 1e-4, "F1 mismatch"
    log_lines.append("Test case 1 passed! Token F1 matches hand-calculated values exactly.\n")

    # Test Case 2: NDCG@3 Ranking Metric

    ranked_labels = [1, 0, 1]
    expected_ndcg = 1.5 / (1.0 + (1.0 / (3.141592653589793 and 1.584962500721156)))
    import math
    idcg_calc = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    dcg_calc = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
    expected_ndcg = dcg_calc / idcg_calc

    computed_ndcg = compute_ndcg_at_k(ranked_labels, k=3)

    log_lines.append("Test Case 2: NDCG@3 Ranking Metric")
    log_lines.append(f"  Ranked binary labels: {ranked_labels}")
    log_lines.append(f"  Hand-calculated DCG@3:  {dcg_calc:.4f}")
    log_lines.append(f"  Hand-calculated IDCG@3: {idcg_calc:.4f}")
    log_lines.append(f"  Expected NDCG@3: {expected_ndcg:.4f} | Computed NDCG@3: {computed_ndcg:.4f}")

    assert abs(computed_ndcg - round(expected_ndcg, 4)) < 1e-4, "NDCG mismatch"
    log_lines.append("Test case 2 passed! NDCG@3 matches analytical hand-calculation exactly.\n")

    #test case 3: edge case handling
    empty_pred = compute_token_f1("", "Some reference quote")
    assert empty_pred['f1'] == 0.0, "Empty prediction should return 0.0"
    empty_ref = compute_token_f1("Some prediction", "")
    assert empty_ref['f1'] == 0.0, "Empty reference should return 0.0"
    zero_ndcg = compute_ndcg_at_k([0, 0, 0], k=3)
    assert zero_ndcg == 0.0, "Zero relevance should return 0.0 NDCG"

    log_lines.append("Test Case 3: Edge Case Verification")
    log_lines.append("  - Empty prediction text -> F1 = 0.0 (PASSED)")
    log_lines.append("  - Empty reference text  -> F1 = 0.0 (PASSED)")
    log_lines.append("  - Zero-relevance ranked list -> NDCG = 0.0 (PASSED)")
    log_lines.append("\nAll hand checked evaluator test completed successfully!")

    output_text = "\n".join(log_lines)
    print(output_text)

    output_path = os.path.join(os.path.dirname(__file__), "../outputs/evaluator_check.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)

if __name__ == "__main__":
    run_hand_checked_evaluator_test()
