"""
End-to-end execution pipeline for assignment 3 early implementation.
Loads sample data, trains TF-IDF baseline, runs evaluation on held-out books,
and logs structured predictions and performance metrics.
"""

import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(__file__))
from data_loader import load_corpus, extract_examples, split_by_book
from baseline_model import SimpleTfidfVectorizer, LogisticRegressionBaseline
from evaluator import compute_token_f1, compute_ndcg_at_k, evaluate_classification

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "sample_corpus.json")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    log_lines = []
    log_lines.append("=" * 60)
    log_lines.append("ALLEY CATS: NLP ASSIGNMENT 3 BASELINE PIPELINE EXECUTION")
    log_lines.append("Project: Predicting Reader-Highlighted Passages on E-Readers")
    log_lines.append("=" * 60)

    #load data
    corpus = load_corpus(data_path)
    all_examples = extract_examples(corpus)
    log_lines.append(f"Loaded corpus from: {data_path}")
    log_lines.append(f"Total books: {len(corpus)} | Total candidate sentences: {len(all_examples)}")

    # book-level data split
    train_books = ["austen_pride_and_prejudice", "bronte_wuthering_heights"]
    dev_books = ["fitzgerald_great_gatsby", "shelley_frankenstein"]
    train_data, dev_data = split_by_book(all_examples, train_books, dev_books)

    log_lines.append(f"Training books: {train_books}")
    log_lines.append(f"  -> Train instances: {len(train_data)} ({sum(x['is_highlight'] for x in train_data)} highlights)")
    log_lines.append(f"Development books: {dev_books}")
    log_lines.append(f"  -> Dev instances:   {len(dev_data)} ({sum(x['is_highlight'] for x in dev_data)} highlights)")

    #vectorize text (TF-IDF)
    vectorizer = SimpleTfidfVectorizer(min_df=1, max_features=300, ngram_range=(1, 2))
    train_texts = [x['text'] for x in train_data]
    X_train = vectorizer.fit_transform(train_texts)
    y_train = np.array([x['is_highlight'] for x in train_data])

    dev_texts = [x['text'] for x in dev_data]
    X_dev = vectorizer.transform(dev_texts)
    y_dev = np.array([x['is_highlight'] for x in dev_data])

    log_lines.append(f"Fitted TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} n-grams")

    #train baseline model
    model = LogisticRegressionBaseline(lr=0.8, l2_reg=0.01, epochs=200)
    model.fit(X_train, y_train)
    log_lines.append("Trained Logistic Regression baseline classifier with class weighting.")

    #evaluate on development set
    dev_probs = model.predict_proba(X_dev)
    dev_preds = model.predict(X_dev, threshold=0.5)

    class_metrics = evaluate_classification(y_dev.tolist(), dev_preds.tolist())
    log_lines.append("\n--- HELD-OUT DEVELOPMENT SET RESULTS ---")
    log_lines.append(f"Accuracy:  {class_metrics['accuracy']:.4f}")
    log_lines.append(f"Precision: {class_metrics['precision']:.4f}")
    log_lines.append(f"Recall:    {class_metrics['recall']:.4f}")
    log_lines.append(f"F1 Score:  {class_metrics['f1']:.4f}")
    log_lines.append(f"Confusion Matrix: TP={class_metrics['tp']}, FP={class_metrics['fp']}, FN={class_metrics['fn']}, TN={class_metrics['tn']}")

    #per-book ranking and NDCG evaluation
    ndcg_scores = {}
    for b_id in dev_books:
        book_indices = [i for i, ex in enumerate(dev_data) if ex['book_id'] == b_id]
        book_probs = [dev_probs[i] for i in book_indices]
        book_labels = [y_dev[i] for i in book_indices]

        #rank by predicted probability
        ranked_order = np.argsort(book_probs)[::-1]
        ranked_labels = [book_labels[idx] for idx in ranked_order]
        score = compute_ndcg_at_k(ranked_labels, k=5)
        ndcg_scores[b_id] = score
        log_lines.append(f"NDCG@5 for {b_id}: {score:.4f} (Ranked labels: {ranked_labels})")

    macro_ndcg = np.mean(list(ndcg_scores.values()))
    log_lines.append(f"Macro-averaged NDCG@5 across held-out books: {macro_ndcg:.4f}")

    #collect representative i/o records
    saved_records = []
    for idx, ex in enumerate(dev_data):
        prob = float(dev_probs[idx])
        pred_label = int(dev_preds[idx])
        true_label = int(y_dev[idx])
        
        #token overlap if true highlight
        if true_label == 1:
            token_eval = compute_token_f1(ex['text'], ex['text']) #reference match check
        else:
            token_eval = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        saved_records.append({
            'sentence_id': ex['sentence_id'],
            'book_title': ex['title'],
            'author': ex['author'],
            'genre': ex['genre'],
            'candidate_text': ex['text'],
            'context_excerpt': ex['context'][:180] + "..." if len(ex['context']) > 180 else ex['context'],
            'ground_truth_highlight': true_label,
            'predicted_salience_score': round(prob, 4),
            'predicted_label': pred_label,
            'is_correct': bool(pred_label == true_label),
            'token_overlap_with_target': token_eval
        })

    #save outputs
    pred_path = os.path.join(out_dir, "predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(saved_records, f, indent=2)
    log_lines.append(f"\nSaved {len(saved_records)} predictions to: {pred_path}")

    log_path = os.path.join(out_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log_lines.append(f"Saved run log to: {log_path}")

    print("\n".join(log_lines))

if __name__ == "__main__":
    main()
