This folder contains the python scripts that were written to scrape the data collected in the data folder. 
The gutenberg_goodreads_scraper.py script is responsible for collecting the quotes off of the goodreads site from the top 100 downloaded 
books in the past 30 days. This script creates the quotes.csv file as well as a CSV files to track progress. 
The gutenberg_quote_context.py collects the surrounding context 
from the given quote set from Project gutenburg and creates a CSV file with the quotes with context and 
another with the quotes where context could not be found in the version of the book freely available. 
