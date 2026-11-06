#!/usr/bin/env python3
"""
Inspect the quote corpus

Inputs (produced by gutenberg_goodreads_scraper.py and gutenberg_quote_context.py):
  quotes.csv                                    QUOTE, AUTHOR, TITLE, LIKES, TAGS
  gutenberg_goodreads_quotes_with_context.csv   BEFORE QUOTE, QUOTE, AFTER QUOTE, AUTHOR, TITLE, LIKES, TAGS
  gutenberg_goodreads_quotes_unmatched.csv      QUOTE, AUTHOR, TITLE, LIKES, TAGS, REASON

Usage:
  python inspect_and_split.py --collected quotes.csv \
      --matched gutenberg_goodreads_quotes_with_context.csv \
      --unmatched gutenberg_goodreads_quotes_unmatched.csv
"""
import argparse, json, os, random, re, sys, unicodedata
import pandas as pd

# Editions that are the same work (listed twice by Gutenberg) or nested inside another.
ALIASES = {"don quijote": "don quixote",
           "romeo and juliet": "the complete works of william shakespeare"}
EXCLUDE_NON_ENGLISH = {"don quijote"}  # Spanish-language edition


def get_splitter():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from gutenberg_quote_context import split_sentences  # same rules as the pipeline
        return split_sentences
    except Exception:
        rx = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=[A-Z"\'(\[])')
        return lambda t: [s for s in rx.split(str(t)) if s.strip()]


def nk(t):
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(t).lower()))


def work_key(title):
    short = re.split(r"[;:]|\bor,", str(title), maxsplit=1, flags=re.I)[0].strip().lower()
    return re.sub(r"\s+", " ", short)


def pct(a, b):
    return round(100.0 * a / b, 1) if b else 0.0


def desc(s):
    s = pd.Series(s).astype(float)
    return {"min": float(s.min()), "median": float(s.median()), "p90": float(s.quantile(.9)), "max": float(s.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", required=True)
    ap.add_argument("--matched", required=True)
    ap.add_argument("--unmatched", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    split_sentences = get_splitter()

    col = pd.read_csv(a.collected, encoding="utf-8-sig")
    mat = pd.read_csv(a.matched, encoding="utf-8-sig")
    unm = pd.read_csv(a.unmatched, encoding="utf-8-sig")
    N = {}

    # ---- 2A: where did the collected quotations go? ----
    col["_dk"] = list(zip(col.TITLE.map(nk), col.QUOTE.map(nk)))
    dup_rows = int(col["_dk"].duplicated().sum())
    uniq = col[~col["_dk"].duplicated()]
    gone = set(col.TITLE) - set(mat.TITLE) - set(unm.TITLE)
    never = int(uniq.TITLE.isin(gone).sum())
    N["collected"] = {"rows": len(col), "titles": int(col.TITLE.nunique()), "authors": int(col.AUTHOR.nunique()),
                      "likes_min": int(col.LIKES.min()), "likes_median": float(col.LIKES.median()),
                      "share_le10_likes_pct": pct(int((col.LIKES <= 10).sum()), len(col))}
    N["accounting"] = {"collected": len(col), "located": len(mat), "not_located": len(unm),
                       "duplicate_rows_dropped_by_pipeline": dup_rows,
                       "never_processed": never, "never_processed_titles": sorted(gone),
                       "sum_check": len(mat) + len(unm) + dup_rows + never,
                       "located_pct_of_unique_attempted": pct(len(mat), len(mat) + len(unm))}
    rate = (uniq.groupby("TITLE").size().rename("n").to_frame()
            .join(mat.groupby("TITLE").size().rename("m")).fillna(0))
    rate["rate"] = rate.m / rate.n
    N["lowest_match_rate_titles"] = {t: [int(r.n), int(r.m), round(float(r.rate), 2)]
                                     for t, r in rate[rate.n >= 10].sort_values("rate").head(9).iterrows()}

    # analysis set: merge duplicate listings, drop Spanish edition, dedupe quote text 
    mat["WORK"] = mat.TITLE.map(work_key)
    n0 = len(mat)
    non_en = int(mat.WORK.isin(EXCLUDE_NON_ENGLISH).sum())
    mat = mat[~mat.WORK.isin(EXCLUDE_NON_ENGLISH)].copy()
    mat["WORK"] = mat.WORK.replace(ALIASES)
    mat["_n"] = mat.QUOTE.map(nk)
    cross = int(mat._n.duplicated().sum())
    ana = mat[~mat._n.duplicated()].reset_index(drop=True)
    N["analysis_set"] = {"located_rows": n0, "non_english_removed": non_en,
                         "cross_listing_duplicates_removed": cross, "rows": len(ana),
                         "works": int(ana.WORK.nunique()), "authors": int(ana.AUTHOR.nunique()),
                         "authors_with_multiple_works": int((ana.groupby("AUTHOR").WORK.nunique() > 1).sum())}

    # inspection
    pw = ana.groupby("WORK").size().sort_values(ascending=False)
    N["per_work"] = {**desc(pw), "top5_share_pct": pct(int(pw.head(5).sum()), len(ana)),
                     "top5": {k: int(v) for k, v in pw.head(5).items()},
                     "works_lt5": int((pw < 5).sum()), "works_lt10": int((pw < 10).sum())}
    toks = ana.QUOTE.astype(str).str.split().str.len()
    nsent = ana.QUOTE.astype(str).map(lambda x: len(split_sentences(x)))
    N["quote_len"] = {"tokens": desc(toks), "sentences": desc(nsent),
                      "multi_sentence_pct": pct(int((nsent > 1).sum()), len(ana)),
                      "gt100_tokens_pct": pct(int((toks > 100).sum()), len(ana))}
    N["likes_matched"] = {"median": float(ana.LIKES.median()),
                          "le10_pct": pct(int((ana.LIKES <= 10).sum()), len(ana))}
    N["tags_missing_pct"] = pct(int(ana.TAGS.isna().sum()), len(ana))
    N["after_context_empty_rows"] = int(ana["AFTER QUOTE"].isna().sum() + (ana["AFTER QUOTE"].astype(str).str.strip() == "").sum())
    ana["BEFORE QUOTE"] = ana["BEFORE QUOTE"].fillna("").astype(str)
    ana["AFTER QUOTE"] = ana["AFTER QUOTE"].fillna("").astype(str)
    by_work = {k: [nk(x) for x in g.QUOTE] for k, g in ana.groupby("WORK")}
    tot = hit = 0
    for _, r in ana.iterrows():
        own = nk(r.QUOTE)
        others = [o for o in by_work[r.WORK] if o != own]
        for s in split_sentences(r["BEFORE QUOTE"]) + split_sentences(r["AFTER QUOTE"]):
            ns = nk(s)
            if len(s.split()) < 4:
                continue
            tot += 1
            if any(ns in o or (len(o) >= 20 and o in ns) for o in others):
                hit += 1
    N["context_contamination"] = {"sentences_checked": tot, "inside_another_quote": hit, "pct": pct(hit, tot)}

    # work level split
    works = sorted(ana.WORK.unique()); random.Random(a.seed).shuffle(works)
    n = len(works); ntr, ndv = round(.7 * n), round(.1 * n)
    split = {w: ("train" if i < ntr else "dev" if i < ntr + ndv else "test") for i, w in enumerate(works)}
    ana["SPLIT"] = ana.WORK.map(split)
    man = ana.groupby(["WORK", "SPLIT"]).agg(quotes=("QUOTE", "size"), author=("AUTHOR", "first")).reset_index()
    man.to_csv(os.path.join(a.out, "split_manifest.csv"), index=False)
    N["split"] = {"seed": a.seed,
                  "works": ana.groupby("SPLIT").WORK.nunique().to_dict(),
                  "quotes": ana.groupby("SPLIT").size().to_dict(),
                  "authors_in_more_than_one_split": int((ana.groupby("AUTHOR").SPLIT.nunique() > 1).sum())}
    ana.sample(n=min(a.sample, len(ana)), random_state=a.seed).drop(columns=["_n", "WORK"]).to_csv(
        os.path.join(a.out, "sample_records.csv"), index=False)
    json.dump(N, open(os.path.join(a.out, "section2_numbers.json"), "w"), indent=2, default=str)
    print(json.dumps(N, indent=2, default=str))


if __name__ == "__main__":
    main()
