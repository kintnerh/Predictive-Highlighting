The quotes.csv file contains all quotes collected for each book using Goodreads' quote search: 

https://www.goodreads.com/quotes/search?utf8=%E2%9C%93&q=<book title + author>&c

The books were selected from Project Gutenburg's top 100 downloaded books in the past 30 days as of **10/29/2026**. The quotes also had to have at least 5 likes to be added to the CSV file. 
Output CSV columns: QUOTE, AUTHOR, TITLE, LIKES, TAGS



The gutenburg_goodreads_quotes_with_context.csv file was created using the CSV made by gutenberg_goodreads_scraper.py, finds each quote inside thefull Project Gutenberg text of its book, and writes a NEW csv with the N sentences before and after the quote.

Output columns:
    BEFORE QUOTE | QUOTE | AFTER QUOTE | AUTHOR | TITLE | LIKES | TAGS

Quotes that could not be located in the Gutenberg text are written to a separate
"unmatched" CSV (with a REASON column).

