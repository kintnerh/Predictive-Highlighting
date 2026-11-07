"""
Data loading and preprocessing pipeline for quote and highlight salience.
Implements book-level splitting to prevent lexical and stylistic leakage.
"""

import json
import re
from typing import List, Dict, Tuple

def clean_text(text: str) -> str:
    #normalize whitespace and quotes
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_corpus(filepath: str) -> List[Dict]:
    #load JSON corpus of books and sentences
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_examples(corpus: List[Dict]) -> List[Dict]:
    #extract records with sentence, context, metadata, and label
    examples = []
    for book in corpus:
        b_id = book['book_id']
        title = book['title']
        author = book['author']
        genre = book['genre']
        for sent in book['sentences']:
            clean_sent = clean_text(sent['text'])
            c_before = clean_text(sent.get('context_before', ''))
            c_after = clean_text(sent.get('context_after', ''))
            full_context = f"{c_before} {clean_sent} {c_after}".strip()
            
            examples.append({
                'sentence_id': sent['sentence_id'],
                'book_id': b_id,
                'title': title,
                'author': author,
                'genre': genre,
                'text': clean_sent,
                'context': full_context,
                'is_highlight': int(sent['is_highlight']),
                'likes': int(sent.get('likes', 0))
            })
    return examples

def split_by_book(examples: List[Dict], train_books: List[str], dev_books: List[str]) -> Tuple[List[Dict], List[Dict]]:
    #split records by book_id to prevent inter-book leakage
    train_set = [ex for ex in examples if ex['book_id'] in train_books]
    dev_set = [ex for ex in examples if ex['book_id'] in dev_books]
    return train_set, dev_set

if __name__ == "__main__":
    import os
    sample_file = os.path.join(os.path.dirname(__file__), "../data/sample_corpus.json")
    corpus = load_corpus(sample_file)
    exs = extract_examples(corpus)
    print(f"Loaded {len(corpus)} books, {len(exs)} total sentences.")
    train_b = ["austen_pride_and_prejudice", "bronte_wuthering_heights"]
    dev_b = ["fitzgerald_great_gatsby", "shelley_frankenstein"]
    tr, dv = split_by_book(exs, train_b, dev_b)
    print(f"Train sentences: {len(tr)} ({sum(x['is_highlight'] for x in tr)} highlights)")
    print(f"Dev sentences: {len(dv)} ({sum(x['is_highlight'] for x in dv)} highlights)")
