"""
Scrape popular Goodreads quotes for the Project Gutenberg "Top 100 (last 30 days)" books.

For each book it uses Goodreads' quote search:
  https://www.goodreads.com/quotes/search?utf8=%E2%9C%93&q=<book title + author>&commit=Search

Output CSV columns: QUOTE, AUTHOR, TITLE, LIKES, TAGS

Install:  pip install requests beautifulsoup4
Run:      python gutenberg_goodreads_scraper.py
          python gutenberg_goodreads_scraper.py --min-likes 10 --output quotes.csv
"""

import argparse
import itertools
import csv
import random
import re
import sys
import time
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup, Tag

GUTENBERG_TOP_URL = "https://www.gutenberg.org/browse/scores/top"
GOODREADS = "https://www.goodreads.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)

# Optional: export GOODREADS_COOKIE="..." (copied from your browser) if Goodreads blocks anonymous requests
import os
if os.environ.get("GOODREADS_COOKIE"):
    session.headers["Cookie"] = os.environ["GOODREADS_COOKIE"]

DEBUG = False


def save_debug(name, html):
    """Save raw HTML so you can see what Goodreads actually returned."""
    if not DEBUG or not html:
        return
    os.makedirs("debug", exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:80]
    with open(os.path.join("debug", safe + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def page_hint(html):
    """Tell the user if Goodreads returned a login / captcha page."""
    low = html.lower()
    title = BeautifulSoup(html, "html.parser").title
    title = title.get_text(strip=True) if title else "(no title)"
    flag = ""
    if "captcha" in low or "unusual traffic" in low:
        flag = " [looks like a CAPTCHA/bot check]"
    elif "sign in" in title.lower() or "log in" in title.lower():
        flag = " [looks like a login page]"
    return f"page title: {title!r}{flag}"


def polite_sleep(lo=1.5, hi=3.0):
    time.sleep(random.uniform(lo, hi))


def get(url, retries=3):
    """GET with simple retry/backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 503):
                wait = 10 * attempt
                print(f"  ! HTTP {resp.status_code} - waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  ! HTTP {resp.status_code} for {url}", file=sys.stderr)
            return None
        except requests.RequestException as e:
            print(f"  ! Request error ({e}); retry {attempt}/{retries}", file=sys.stderr)
            time.sleep(3 * attempt)
    return None


# --------------------------------------------------------------------------
# Step 1: Project Gutenberg top 100 (last 30 days)
# --------------------------------------------------------------------------
def get_gutenberg_top_books(limit=100):
    html = get(GUTENBERG_TOP_URL)
    if not html:
        sys.exit("Could not load the Project Gutenberg top list.")
    soup = BeautifulSoup(html, "html.parser")

    header = soup.find(id="books-last30")
    if header is None:
        sys.exit("Couldn't find the 'books-last30' section; Gutenberg may have changed its layout.")
    ol = header.find_next("ol")

    books = []
    for li in ol.find_all("li")[:limit]:
        text = li.get_text(" ", strip=True)
        text = re.sub(r"\s*\(\d[\d,]*\)\s*$", "", text)  # drop trailing download count
        if " by " in text:
            title, author = text.rsplit(" by ", 1)
        else:
            title, author = text, ""
        books.append({"title": title.strip(), "author": author.strip()})
    return books


# --------------------------------------------------------------------------
# Step 2: Goodreads quote search  (quotes/search?q=<title author>)
# --------------------------------------------------------------------------
LIKES_RE = re.compile(r"([\d,]+)\s+likes?", re.I)


def search_title(title):
    """Shorten titles like 'Frankenstein; Or, The Modern Prometheus' -> 'Frankenstein'."""
    short = re.split(r"[;:]|\bor,", title, maxsplit=1, flags=re.I)[0].strip()
    return short or title


def last_name(author):
    parts = author.replace(",", " ").split()
    return parts[-1].lower() if parts else ""


def norm(t):
    return re.sub(r"[^a-z0-9 ]+", "", t.lower().replace("\u2019", "'")).strip()


def search_url(query, page):
    return (f"{GOODREADS}/quotes/search?utf8=%E2%9C%93&q={quote_plus(query)}"
            f"&commit=Search&page={page}")


def parse_quote_block(block):
    qt = block.select_one(".quoteText")
    if qt is None:
        return None

    # Attribution: <span class="authorOrTitle">Jane Austen,</span> <a class="authorOrTitle">Book</a>
    gr_author = ""
    gr_book = ""
    span = qt.select_one("span.authorOrTitle")
    if span:
        gr_author = span.get_text(strip=True).rstrip(",").strip()
    book_a = qt.select_one("a.authorOrTitle")
    if book_a:
        gr_book = book_a.get_text(strip=True)

    # Turn <br> into newlines, keep everything before the author line ("― Author")
    for br in qt.find_all("br"):
        br.replace_with("\n")
    raw = qt.get_text()
    attribution = ""
    if "\u2015" in raw:
        raw, attribution = raw.rsplit("\u2015", 1)
    if not gr_author and attribution:
        gr_author, _, rest = attribution.strip().partition(",")
        gr_author = gr_author.strip()
        gr_book = gr_book or rest.strip()

    text = " ".join(raw.split()).strip().strip("\u201c\u201d\"").strip()

    likes = 0
    qid = ""
    for a in block.select(".quoteFooter a"):
        m = LIKES_RE.search(a.get_text(" ", strip=True))
        if m:
            likes = int(m.group(1).replace(",", ""))
            idm = re.search(r"/quotes/(\d+)", a.get("href", ""))
            qid = idm.group(1) if idm else ""
            break
    if likes == 0:
        m = LIKES_RE.search(block.get_text(" ", strip=True))
        if m:
            likes = int(m.group(1).replace(",", ""))

    tags = [a.get_text(strip=True) for a in block.select(".quoteFooter .left a")]

    if not text:
        return None
    return {"id": qid, "likes": likes, "quote": text, "tags": ", ".join(tags),
            "gr_author": gr_author, "gr_book": gr_book}


def is_match(q, want_title, want_last):
    """Keep only quotes whose author AND book title match the Gutenberg book."""
    if not q["gr_book"] or not q["gr_author"]:
        return False
    if want_last not in norm(q["gr_author"]):
        return False
    gt, wt = norm(q["gr_book"]), norm(want_title)
    return wt in gt or (len(gt) > 3 and gt in wt)


def scrape_book(title, author, min_likes, max_pages):
    """Run the quote search for 'title author' and collect matching quotes with > min_likes."""
    short = search_title(title)
    want_last = last_name(author)
    queries = [f"{short} {author}".strip(), short]  # second query is a fallback

    for qi, query in enumerate(queries):
        found = {}
        for page in itertools.count(1):
            if max_pages and page > max_pages:
                break
            html = get(search_url(query, page))
            if not html:
                break
            save_debug(f"search_{short}_q{qi}_p{page}", html)
            blocks = BeautifulSoup(html, "html.parser").select("div.quote")
            if not blocks:
                if page == 1:
                    print(f"  ! 0 quote blocks for query {query!r} - " + page_hint(html), file=sys.stderr)
                break

            parsed = [q for q in map(parse_quote_block, blocks) if q]
            kept = 0
            for q in parsed:
                if q["likes"] > min_likes and is_match(q, short, want_last):
                    key = q["id"] or q["quote"]
                    if key not in found or q["likes"] > found[key]["likes"]:
                        found[key] = q
                    kept += 1
            print(f"    [{query}] page {page}: {len(blocks)} results, {kept} matched")

            # Results are ordered by likes: stop once nothing on the page beats the threshold
            if not any(q["likes"] > min_likes for q in parsed):
                break
            polite_sleep()

        if found:
            return list(found.values())
        print(f"  - no matches for query {query!r}")
    return []


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-likes", type=int, default=10, help="keep quotes with MORE than this many likes")
    ap.add_argument("--max-pages", type=int, default=0, help="optional cap on result pages per book (0 = no limit; goes until Goodreads runs out of pages)")
    ap.add_argument("--limit", type=int, default=100, help="number of Gutenberg books to process")
    ap.add_argument("--output", default="gutenberg_goodreads_quotes.csv")
    ap.add_argument("--debug", action="store_true", help="save raw HTML pages into ./debug")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing CSV, skip books already finished, and append new rows")
    ap.add_argument("--start-at", type=int, default=1,
                    help="start at this book number in the Gutenberg list (1-based); keeps existing CSV data")
    ap.add_argument("--overwrite", action="store_true", help="start over and replace an existing CSV")
    args = ap.parse_args()
    global DEBUG
    DEBUG = args.debug

    books = get_gutenberg_top_books(args.limit)
    print(f"Found {len(books)} books on Project Gutenberg's top list.\n")

    exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    resuming = args.resume or args.start_at > 1
    if exists and not resuming and not args.overwrite:
        sys.exit(f"{args.output} already exists, so nothing was changed.\n"
                 "  --resume        continue where it left off (keeps existing rows)\n"
                 "  --start-at N    continue from book number N (keeps existing rows)\n"
                 "  --overwrite     start over and replace the file")

    progress_path = args.output + ".progress"   # one finished book per line
    done = set()
    if exists and resuming:
        if os.path.exists(progress_path):
            with open(progress_path, encoding="utf-8") as pf:
                done = {line.rstrip("\n") for line in pf if line.strip()}
        else:  # no progress file yet: infer finished books from the CSV itself
            with open(args.output, newline="", encoding="utf-8-sig") as cf:
                done = {f"{r['TITLE']}\t{r['AUTHOR']}" for r in csv.DictReader(cf)}
        print(f"Resuming: {len(done)} books already finished, existing rows will be kept.\n")

    append = exists and resuming
    if not append and os.path.exists(progress_path):
        os.remove(progress_path)

    with open(args.output, "a" if append else "w", newline="",
              encoding="utf-8" if append else "utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["QUOTE", "AUTHOR", "TITLE", "LIKES", "TAGS"])
        if not append:
            writer.writeheader()

        total = 0
        for i, book in enumerate(books, 1):
            key = f"{book['title']}\t{book['author']}"
            if i < args.start_at or key in done:
                print(f"[{i}/{len(books)}] {book['title']} - skipped (already done)")
                continue
            print(f"[{i}/{len(books)}] {book['title']} - {book['author']}")
            try:
                quotes = scrape_book(book["title"], book["author"], args.min_likes, args.max_pages)
                quotes.sort(key=lambda q: q["likes"], reverse=True)
                for q in quotes:
                    writer.writerow({
                        "QUOTE": q["quote"],
                        "AUTHOR": book["author"],
                        "TITLE": book["title"],
                        "LIKES": q["likes"],
                        "TAGS": q["tags"],
                    })
                f.flush()
                with open(progress_path, "a", encoding="utf-8") as pf:  # mark finished only after rows are saved
                    pf.write(key + "\n")
                total += len(quotes)
                print(f"  + {len(quotes)} quotes (>{args.min_likes} likes)")
            except Exception as e:  # keep going if one book fails; it will be retried on --resume
                print(f"  ! error: {e}", file=sys.stderr)

    print(f"\nDone. {total} quotes saved to {args.output}")


if __name__ == "__main__":
    main()
