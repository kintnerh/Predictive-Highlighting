"""
Add surrounding context to the Goodreads quotes.

Reads the CSV made by gutenberg_goodreads_scraper.py, finds each quote inside the
full Project Gutenberg text of its book, and writes a NEW csv with the N sentences
before and after the quote.

Output columns:
    BEFORE QUOTE | QUOTE | AFTER QUOTE | AUTHOR | TITLE | LIKES | TAGS

Quotes that can't be located in the Gutenberg text are written to a separate
"unmatched" CSV (with a REASON column) instead of being guessed at.

Install:  pip install requests beautifulsoup4
Run:      python gutenberg_quote_context.py
          python gutenberg_quote_context.py --input quotes.csv --sentences 10
"""

import argparse
import bisect
import csv
import difflib
import os
import random
import re
import sys
import time
import unicodedata
from array import array
from collections import OrderedDict
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

GUTENBERG_TOP_URL = "https://www.gutenberg.org/browse/scores/top"
GUTENBERG = "https://www.gutenberg.org"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
session = requests.Session()
session.headers.update(HEADERS)


# Networking helpers
def polite_sleep(lo=2.0, hi=4.0):
    time.sleep(random.uniform(lo, hi))


def get_bytes(url, retries=3):
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code in (429, 503):
                time.sleep(10 * attempt)
                continue
            return None
        except requests.RequestException as e:
            print(f"  ! request error ({e}); retry {attempt}/{retries}", file=sys.stderr)
            time.sleep(3 * attempt)
    return None


def get_html(url):
    data = get_bytes(url)
    return data.decode("utf-8", errors="replace") if data else None


# Map each book (title + author) to its Project Gutenberg ebook ID
def norm_key(t):
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", (t or "").lower()))


def search_title(title):
    short = re.split(r"[;:]|\bor,", title, maxsplit=1, flags=re.I)[0].strip()
    return short or title


def load_top_list():
    html = get_html(GUTENBERG_TOP_URL)
    if not html:
        print("! Could not load the Gutenberg top list; will use Gutenberg search instead.", file=sys.stderr)
        return {}, {}
    soup = BeautifulSoup(html, "html.parser")
    header = soup.find(id="books-last30")
    by_key, by_title = {}, {}
    if header is None:
        return by_key, by_title
    for li in header.find_next("ol").find_all("li"):
        a = li.find("a", href=re.compile(r"/ebooks/\d+"))
        if not a:
            continue
        book_id = re.search(r"/ebooks/(\d+)", a["href"]).group(1)
        text = li.get_text(" ", strip=True)
        text = re.sub(r"\s*\(\d[\d,]*\)\s*$", "", text)
        title, author = text.rsplit(" by ", 1) if " by " in text else (text, "")
        by_key.setdefault((norm_key(title), norm_key(author)), book_id)  # first = highest ranked
        by_title.setdefault(norm_key(title), book_id)
    return by_key, by_title


def search_gutenberg_id(title, author):
    q = quote_plus(f"{search_title(title)} {author}".strip())
    html = get_html(f"{GUTENBERG}/ebooks/search/?query={q}")
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    a = soup.select_one("li.booklink a[href^='/ebooks/']")
    m = re.search(r"/ebooks/(\d+)", a["href"]) if a else None
    return m.group(1) if m else None


def find_gutenberg_id(title, author, by_key, by_title):
    book_id = by_key.get((norm_key(title), norm_key(author))) or by_title.get(norm_key(title))
    return book_id or search_gutenberg_id(title, author)


def download_book(book_id, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{book_id}.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    urls = [
        f"{GUTENBERG}/cache/epub/{book_id}/pg{book_id}.txt",
        f"{GUTENBERG}/ebooks/{book_id}.txt.utf-8",
        f"{GUTENBERG}/files/{book_id}/{book_id}-0.txt",
    ]
    for url in urls:
        data = get_bytes(url)
        if data:
            text = data.decode("utf-8-sig", errors="replace")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            polite_sleep()
            return text
    return None



# Clean the book and split it into sentences
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.I | re.S)

ABBREVS = {
    "mr", "mrs", "ms", "dr", "st", "sr", "jr", "prof", "rev", "hon", "capt", "col", "gen",
    "lt", "sgt", "gov", "messrs", "mme", "mlle", "vs", "mt", "ft", "fig", "vol", "chap",
    "viz", "esq", "bros", "mons", "sig", "hr",
}
BOUNDARY_RE = re.compile(r"([.!?]+[\"\u201d\u2019')\]]*)\s+")
OPENERS = "\"\u201c\u2018'(["
HEADING_RE = re.compile(
    r"^(chapter|volume|vol\.|book|part|letter|act|scene|section|preface|contents|appendix|epilogue|prologue)\b[^.!?]{0,60}$",
    re.I,
)


def clean_gutenberg_text(raw):
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    m = START_RE.search(raw)
    if m:
        raw = raw[m.end():]
    m = END_RE.search(raw)
    if m:
        raw = raw[:m.start()]
    raw = re.sub(r"\[Illustration[^\]]*\]", " ", raw, flags=re.I)
    return raw.replace("_", "")  # Gutenberg marks italics with _underscores_


def skip_paragraph(p):
    if len(p) <= 80 and HEADING_RE.match(p):
        return True
    words = p.split()
    return len(words) <= 8 and p.upper() == p and any(c.isalpha() for c in p)


def split_sentences(para):
    out, start = [], 0
    for m in BOUNDARY_RE.finditer(para):
        nxt = para[m.end():m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in OPENERS):
            continue
        punct = m.group(1)
        if punct.startswith(".") and not punct.startswith(".."):
            wm = re.search(r"([^\W\d_]+)$", para[start:m.start(1)])
            if wm:
                w = wm.group(1)
                if w.lower() in ABBREVS or (len(w) == 1 and w.isupper()):
                    continue  # "Mr. Darcy", "J. Smith" are not sentence ends
        out.append(para[start:m.end(1)].strip())
        start = m.end()
    last = para[start:].strip()
    if last:
        out.append(last)
    return out



# Word index so quotes can be located (tolerates punctuation differences)
WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)


def norm_word(w):
    w = w.lower().replace("'", "").replace("\u2019", "")
    if not w.isascii():
        w = "".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c))
    return sys.intern(w)


def tokenize(text):
    return [norm_word(m.group()) for m in WORD_RE.finditer(text)]


class BookIndex:
    def __init__(self, raw_text):
        self.sentences = []
        for para in re.split(r"\n\s*\n", clean_gutenberg_text(raw_text)):
            para = " ".join(para.split())
            if para and not skip_paragraph(para):
                self.sentences.extend(split_sentences(para))

        self.words = []
        self.w_sent = array("I")
        self.w_start = array("I")
        self.w_end = array("I")
        for si, sent in enumerate(self.sentences):
            for m in WORD_RE.finditer(sent):
                self.words.append(norm_word(m.group()))
                self.w_sent.append(si)
                self.w_start.append(m.start())
                self.w_end.append(m.end())

        self.hay = " ".join(self.words)
        self.hay_p = " " + self.hay + " "
        offs, pos = array("Q"), 0
        for w in self.words:
            offs.append(pos)
            pos += len(w) + 1
        self.hay_off = offs

    def _find_all(self, needle_words, limit):
        needle = " " + " ".join(needle_words) + " "
        hits, pos = [], self.hay_p.find(needle)
        while pos != -1 and len(hits) < limit:
            hits.append(bisect.bisect_right(self.hay_off, pos) - 1)
            pos = self.hay_p.find(needle, pos + 1)
        return hits

    def locate(self, quote, min_coverage=0.7):
        """Return (first_word, last_word, method) or None."""
        qw = tokenize(quote)
        if len(qw) < 2:
            return None

        hits = self._find_all(qw, 1)
        if hits:
            return hits[0], hits[0] + len(qw) - 1, "exact"

        if len(qw) < 4:
            return None

        # Fuzzy: vote with short word chunks, then confirm with a sequence match
        k = 4 if len(qw) < 12 else 6
        votes = {}
        for i in range(0, len(qw) - k + 1, max(1, k // 2)):
            for wi in self._find_all(qw[i:i + k], 25):
                votes[wi - i] = votes.get(wi - i, 0) + 1
        best = None
        for approx, _ in sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            lo = max(0, approx - 5)
            hi = min(len(self.words), approx + len(qw) + 5)
            sm = difflib.SequenceMatcher(None, qw, self.words[lo:hi], autojunk=False)
            blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
            if not blocks:
                continue
            cov = sum(b.size for b in blocks) / len(qw)
            if cov >= min_coverage and (best is None or cov > best[0]):
                best = (cov, lo + blocks[0].b, lo + blocks[-1].b + blocks[-1].size - 1)
        if best:
            return best[1], best[2], f"fuzzy {best[0]:.0%}"
        return None

    # context 
    def context(self, first, last, n):
        s0, s1 = self.w_sent[first], self.w_sent[last]
        lead = self.sentences[s0][:self.w_start[first]]
        trail = self.sentences[s1][self.w_end[last]:]
        lead = lead.strip() if any(c.isalnum() for c in lead) else ""
        trail = trail.strip() if any(c.isalnum() for c in trail) else ""

        before = " ".join(self.sentences[max(0, s0 - n):s0])
        after = " ".join(self.sentences[s1 + 1:s1 + 1 + n])
        # If the quote sits mid-sentence, keep the rest of that sentence as context too
        if lead:
            before = f"{before} {lead}".strip()
        if trail:
            after = f"{trail} {after}".strip()
        return before, after




OUT_COLS = ["BEFORE QUOTE", "QUOTE", "AFTER QUOTE", "AUTHOR", "TITLE", "LIKES", "TAGS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="gutenberg_goodreads_quotes.csv")
    ap.add_argument("--output", default="gutenberg_goodreads_quotes_with_context.csv")
    ap.add_argument("--unmatched", default="gutenberg_goodreads_quotes_unmatched.csv")
    ap.add_argument("--sentences", type=int, default=10, help="sentences to keep before and after each quote")
    ap.add_argument("--min-coverage", type=float, default=0.7,
                    help="how much of a quote must match the book text when wording differs (0-1)")
    ap.add_argument("--cache-dir", default="gutenberg_texts", help="where downloaded books are stored")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing output files, skip books already finished, and append new rows")
    ap.add_argument("--start-at", type=int, default=1,
                    help="start at this book number (1-based, in the order books appear in the input CSV); keeps existing data")
    ap.add_argument("--overwrite", action="store_true", help="start over and replace existing output files")
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows found in {args.input}")

    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["TITLE"], r["AUTHOR"]), []).append(r)
    print(f"{len(rows)} quotes across {len(groups)} books.\n")

    exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    resuming = args.resume or args.start_at > 1
    if exists and not resuming and not args.overwrite:
        sys.exit(f"{args.output} already exists, so nothing was changed.\n"
                 "  --resume        continue where it left off (keeps existing rows)\n"
                 "  --start-at N    continue from book number N (keeps existing rows)\n"
                 "  --overwrite     start over and replace the files")

    progress_path = args.output + ".progress"   # one finished book per line
    append = exists and resuming
    done_titles = set()
    seen = set()  # (title, quote) pairs already written, to drop duplicate rows
    if append:
        if os.path.exists(progress_path):
            with open(progress_path, encoding="utf-8") as pf:
                done_titles = {line.rstrip("\n") for line in pf if line.strip()}
        with open(args.output, newline="", encoding="utf-8-sig") as ef:
            for r in csv.DictReader(ef):
                seen.add((norm_key(r["TITLE"]), norm_key(r["QUOTE"])))
                if not os.path.exists(progress_path):
                    done_titles.add(f"{r['TITLE']}\t{r['AUTHOR']}")
        print(f"Resuming: {len(done_titles)} books already finished, existing rows will be kept.\n")
    elif os.path.exists(progress_path):
        os.remove(progress_path)

    by_key, by_title = load_top_list()

    done_books = {}      # gutenberg id -> BookIndex (kept only for the current book)
    total_ok = total_bad = 0
    mode, enc = ("a", "utf-8") if append else ("w", "utf-8-sig")

    with open(args.output, mode, newline="", encoding=enc) as fo, \
         open(args.unmatched, mode, newline="", encoding=enc) as fu:
        out = csv.DictWriter(fo, fieldnames=OUT_COLS)
        bad = csv.DictWriter(fu, fieldnames=["QUOTE", "AUTHOR", "TITLE", "LIKES", "TAGS", "REASON"])
        if not append:
            out.writeheader()
            bad.writeheader()

        def mark_done(key):
            with open(progress_path, "a", encoding="utf-8") as pf:
                pf.write(key + "\n")

        for gi, ((title, author), grp) in enumerate(groups.items(), 1):
            key = f"{title}\t{author}"
            if gi < args.start_at or key in done_titles:
                print(f"[{gi}/{len(groups)}] {title} - skipped (already done)")
                continue
            print(f"[{gi}/{len(groups)}] {title} - {author}  ({len(grp)} quotes)")

            def reject(r, reason):
                bad.writerow({"QUOTE": r["QUOTE"], "AUTHOR": r["AUTHOR"], "TITLE": r["TITLE"],
                              "LIKES": r["LIKES"], "TAGS": r["TAGS"], "REASON": reason})

            try:
                book_id = find_gutenberg_id(title, author, by_key, by_title)
                polite_sleep(1, 2)
                if not book_id:
                    print("  ! book not found on Gutenberg")
                    for r in grp:
                        reject(r, "book not found on Gutenberg")
                    fu.flush()
                    mark_done(key)
                    total_bad += len(grp)
                    continue
                text = download_book(book_id, args.cache_dir)
                if not text:
                    # Not marked done and nothing logged, so --resume will retry it
                    print(f"  ! could not download ebook {book_id}; will be retried on --resume")
                    continue

                if book_id not in done_books:
                    done_books.clear()
                    print(f"  indexing Gutenberg ebook #{book_id} ...")
                    done_books[book_id] = BookIndex(text)
                idx = done_books[book_id]

                ok = 0
                for r in grp:
                    dup_key = (norm_key(title), norm_key(r["QUOTE"]))
                    if dup_key in seen:
                        continue
                    seen.add(dup_key)

                    hit = idx.locate(r["QUOTE"], args.min_coverage)
                    if not hit:
                        reject(r, "quote not found in Gutenberg text")
                        total_bad += 1
                        continue
                    first, last, method = hit
                    before, after = idx.context(first, last, args.sentences)
                    out.writerow({
                        "BEFORE QUOTE": before,
                        "QUOTE": r["QUOTE"],
                        "AFTER QUOTE": after,
                        "AUTHOR": r["AUTHOR"],
                        "TITLE": r["TITLE"],
                        "LIKES": r["LIKES"],
                        "TAGS": r["TAGS"],
                    })
                    ok += 1
                fo.flush()
                fu.flush()
                mark_done(key)  # only after this book's rows are safely written
                total_ok += ok
                print(f"  + {ok} located, {len(grp) - ok} not located/duplicate")
            except Exception as e:  # keep going if one book fails; it will be retried on --resume
                print(f"  ! error: {e}", file=sys.stderr)

    print(f"\nDone. {total_ok} quotes with context -> {args.output}")
    print(f"      {total_bad} quotes not located -> {args.unmatched}")


if __name__ == "__main__":
    main()
