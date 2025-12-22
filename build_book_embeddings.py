import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BOOKS_SUBSET = "books_subset.csv"
EMBEDDINGS_NPY = "book_embeddings.npy"

print("Loading books subset...")
books = pd.read_csv(BOOKS_SUBSET)

# Inspect some columns; common ones include:
# ['book_id','goodreads_book_id','best_book_id','work_id',
#  'books_count','isbn','authors','original_publication_year',
#  'original_title','title','language_code','average_rating',...][web:90]
print("Columns:", books.columns.tolist())

# Build a text field for each book (you can tweak this)
def build_text(row):
    title = str(row.get("title", ""))
    authors = str(row.get("authors", ""))
    year = str(row.get("original_publication_year", ""))
    avg_rating = str(row.get("average_rating", ""))
    return f"{title} by {authors}, published {year}, rating {avg_rating}"

books["text"] = books.apply(build_text, axis=1)

print("Example text row:")
print(books["text"].iloc[0])

print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding book texts...")
embeddings = model.encode(books["text"].tolist(), show_progress_bar=True)

np.save(EMBEDDINGS_NPY, embeddings)

print("Done.")
print("Embeddings shape:", embeddings.shape)
