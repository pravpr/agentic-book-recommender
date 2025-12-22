import pandas as pd

# ---------- CONFIG ----------
BOOKS_CSV = "books.csv"
RATINGS_CSV = "ratings.csv"

MAX_BOOKS = 2000          # subset size to keep it light
MAX_RATINGS = 50000       # subset ratings for performance
OUTPUT_BOOKS = "books_subset.csv"
OUTPUT_RATINGS = "ratings_subset.csv"
# ----------------------------

print("Loading original CSV files...")
books = pd.read_csv(BOOKS_CSV)
ratings = pd.read_csv(RATINGS_CSV)

print("Original shapes:")
print("books:", books.shape)
print("ratings:", ratings.shape)

# Basic sanity check on columns
print("books columns:", books.columns.tolist())
print("ratings columns:", ratings.columns.tolist())

# Keep only the first MAX_BOOKS book_ids
# book_id in ratings refers to the work_id range 1..10000[web:90]
unique_book_ids = sorted(books['book_id'].unique())[:MAX_BOOKS]
books_subset = books[books['book_id'].isin(unique_book_ids)].copy()

# Filter ratings to those books only
ratings_subset = ratings[ratings['book_id'].isin(unique_book_ids)].copy()

# Optionally down-sample ratings if still large
if len(ratings_subset) > MAX_RATINGS:
    ratings_subset = ratings_subset.sample(n=MAX_RATINGS, random_state=42)

print("Subset shapes:")
print("books_subset:", books_subset.shape)
print("ratings_subset:", ratings_subset.shape)

# Save subsets
books_subset.to_csv(OUTPUT_BOOKS, index=False)
ratings_subset.to_csv(OUTPUT_RATINGS, index=False)

print(f"Saved {OUTPUT_BOOKS} and {OUTPUT_RATINGS}")
