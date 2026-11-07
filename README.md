Predictive Highlighting: A Study of Reading Salience

Course: Natural Language Processing
Team Name: Alley Cats  
Team Members:
- Hayley Kintner
- Abbey Reith 
- Anthony Dreier

1. Project Overview & Research Question
During digital and e-reader reading, individuals  mark specific passages that evoke emotional resonance, insight, or stylistic beauty. This project investigates automated reading salience detection: predicting highlight-worthy literary sentences within their surrounding narrative context.

Research Question: To what extent do sparse lexical n-gram representations (TF-IDF) predict highlight-worthy literary passages compared to dense pretrained semantic representations (Sentence-BERT), and does narrative context ($\pm 2$ sentences) improve passage discrimination over isolated sentences?
Scope: Binary classification and candidate passage ranking across held-out Project Gutenberg classic novels aligned with community highlights from Goodreads.

2. Repository Structure
```
Predictive-Highlighting/
├── data/                                         #datasets and documentation
│   ├── README.md                                 #dataset schema and collection notes
│   ├── quotes.csv                                #scraped Goodreads quotes (likes >= 5)
│   ├── gutenberg_goodreads_quotes_with_context.csv #quotes aligned with Gutenberg context
│   ├── gutenberg_goodreads_quotes_unmatched.csv  #unmatched quotes with reason tags
│   └── sample_corpus.json                        #structured pilot corpus for baseline execution
│
├── dataCollection/                               #data acquisition pipeline
│   ├── README.md                                 #scraper documentation
│   ├── gutenberg_goodreads_scraper.py            #Goodreads quote scraper for top Gutenberg books
│   └── gutenberg_quote_context.py                #text search & context extraction module
│
├── src/                                          #modeling & evaluation pipeline (Abbey Reith)
│   ├── data_loader.py                            #context extraction & book-level train/dev splits
│   ├── baseline_model.py                         #TF-IDF vectorizer + Logistic Regression classifier
│   ├── evaluator.py                              #token-level F1 and NDCG@K evaluation engine
│   ├── test_evaluator.py                         #hand-checked evaluator verification unit test
│   └── run_pipeline.py                           #end-to-end baseline training and testing runner
│
├── outputs/                                      #execution evidence & saved predictions
│   ├── predictions.json                          #structured prediction records on held-out books
│   ├── run_log.txt                               #step-by-step pipeline execution trace
│   └── evaluator_check.txt                       #output log of hand-checked evaluator test
│
├── requirements.txt                              #environment dependencies
└── README.md                                     #project documentation
```

3. Environment Setup & Installation

Prerequisites
* Python 3.8+
* `pip` package manager

Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/kintnerh/Predictive-Highlighting.git
cd Predictive-Highlighting
pip install -r requirements.txt
```

4. Execution Instructions

Step 1: Verify the Evaluator (Hand-Checked Sanity Test)
Run the unit test asserting that Token F1 and NDCG@K match step-by-step manual calculations (required by Section 5D of the assignment rubric):
```bash
python3 src/test_evaluator.py
```
Expected Output:
  * Token F1: Computed score (`0.6154`) matches exact analytical value ($32/52 \approx 0.6154$).
  * NDCG@3: Computed ranking score (`0.9197`) matches exact analytical value ($1.5 / 1.6309 \approx 0.9197$).
  * Verification trace is saved to `outputs/evaluator_check.txt`.

Step 2: Run the Baseline Training and Evaluation
Train the TF-IDF Logistic Regression baseline and evaluate on held-out development novels (*The Great Gatsby* and *Frankenstein*):
```bash
python3 src/run_pipeline.py
```
Process:
  1. Loads candidate sentences with $\pm 2$ surrounding sentence context windows.
  2. Partitions data strictly by book to eliminate leakage (Training on *Pride and Prejudice* and *Wuthering Heights*; Development testing on *The Great Gatsby* and *Frankenstein*).
  3. Fits TF-IDF unigram and bigram features strictly on training text.
  4. Trains a regularized Logistic Regression classifier with class-frequency weighting.
  5. Evaluates classification metrics and passage ranking NDCG@5 on held-out development books.
  6. Saves structured predictions to `outputs/predictions.json` and updates `outputs/run_log.txt`.

5. Preliminary Baseline Results

Evaluated on held-out development novels (*The Great Gatsby* and *Frankenstein*):

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | 70.00% | 7 of 10 candidate sentences correctly classified |
| **Precision** | 0.7500 | 3 of 4 predicted highlights are genuine quotes |
| **Recall** | 0.6000 | Captures 3 of 5 highlights on previously unseen authors |
| **Macro F1** | 0.6667 | Balanced under 1:10 positive-to-negative weighting |
| **NDCG@5 (*The Great Gatsby*)** | 1.0000 | Perfect ranking of all highlights at top positions |
| **NDCG@5 (*Frankenstein*)** | 0.6934 | Captures reflective lines; misses novel vocabulary |
| **Macro NDCG@5** | 0.8467 | Strong reference baseline for passage prioritization |

6. Implementation Status & Roadmap

What Runs Now:
* Automated scraping of Goodreads quotes for Project Gutenberg titles with $\ge 5$ likes.
* Sentence context extraction and book-level train/dev splitting.
* Working TF-IDF + Logistic Regression classification baseline.
* Operational evaluation engine (Token F1 and NDCG@5) verified with hand-checked unit tests.
* Saved execution evidence and structured predictions.

What Remains Planned:
* Dense semantic bi-encoder integration (`sentence-transformers/all-MiniLM-L6-v2`) and 2-layer MLP head.
* Controlled context-window studies comparing $\pm 1$, $\pm 2$, and $\pm 4$ sentence windows.
* Qualitative error audit and bootstrap confidence interval reporting.
* Final paper compilation and GitHub cleanup.
