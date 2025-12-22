# Agentic Book Recommendation System (Capstone Project)

This project is an **agentic AI system** that recommends books based on a user's liked title using a **RAG (Retrieval-Augmented Generation) style pipeline** and simple reasoning rules.

The agent:
- Prepares and contextualizes book data from the **Goodbooks-10k** dataset.
- Uses **embeddings + vector similarity** for retrieval.
- Applies **reasoning and self-reflection** (rating filtering, author diversity).
- Uses **tool-calling** for retrieval and feedback logging.
- Computes **evaluation metrics** from logged user feedback.

---

## 1. Tech Stack

- Python 3.10+
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `sentence-transformers`
- Data:
  - Goodbooks-10k dataset (`books.csv` and `ratings.csv`)

---

## 2. Setup Instructions

### 2.1. Clone repo and create venv

# Agentic Book Recommendation System (Capstone Project)

This project is an **agentic AI system** that recommends books based on a user's liked title using a **RAG (Retrieval-Augmented Generation) style pipeline** and simple reasoning rules.

The agent:
- Prepares and contextualizes book data from the **Goodbooks-10k** dataset.
- Uses **embeddings + vector similarity** for retrieval.
- Applies **reasoning and self-reflection** (rating filtering, author diversity).
- Uses **tool-calling** for retrieval and feedback logging.
- Computes **evaluation metrics** from logged user feedback.

---

## 1. Tech Stack

- Python 3.10+
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `sentence-transformers`
- Data:
  - Goodbooks-10k dataset (`books.csv` and `ratings.csv`)

---

## 2. Setup Instructions

### 2.1. Clone repo and create venv

python -m venv venv

Windows:
venv\Scripts\activate

macOS / Linux:
source venv/bin/activate


### 2.2. Install dependencies

pip install -r requirements.txt


---

## 3. Data Preparation & Embeddings

### 3.1. Download Goodbooks-10k

Download `books.csv` and `ratings.csv` from the Goodbooks-10k dataset (e.g., Kaggle or GitHub) and place them in the project root.

Expected files:
books.csv
ratings.csv

### 3.2. Create subset (lightweight for laptop)

python prepare_goodbooks_subset.py

This script:

- Loads `books.csv` and `ratings.csv`.
- Keeps a subset of ~2,000 books and a limited number of ratings.
- Saves `books_subset.csv` and `ratings_subset.csv`.

### 3.3. Build embeddings
python build_book_embeddings.py


This script:

- Loads `books_subset.csv`.
- Builds a text field per book (title + authors + year + rating).
- Uses `SentenceTransformer('all-MiniLM-L6-v2')` to create embeddings.
- Saves them into `book_embeddings.npy`.

After this, you should have:
books_subset.csv
book_embeddings.npy


---

## 4. Running the Agent

### 4.1. Start the interactive agent

venv\Scripts\activate # or source venv/bin/activate
python recommendation_agent.py

## 5. Evaluation

After some interactive sessions, run:
python evaluate_agent.py

## 6. Files Overview

- `prepare_goodbooks_subset.py`  
  Creates `books_subset.csv` and `ratings_subset.csv` from the original Goodbooks-10k data.

- `build_book_embeddings.py`  
  Builds text fields and embeddings using `SentenceTransformer('all-MiniLM-L6-v2')`, saving them to `book_embeddings.npy`.

- `recommendation_agent.py`  
  Main agent loop:
  - Uses **retrieval tool** (`tool_recommend_books`) with cosine similarity.
  - Applies **reasoning and reflection** (`reflect_and_adjust`).
  - Interacts with the user and calls **feedback logging tool** (`log_feedback`).

- `feedback_tools.py`  
  Initializes and appends to `feedback.csv`:
  - `timestamp, user_id, query_title, recommended_title, liked`.

- `evaluate_agent.py`  
  Loads `feedback.csv` and prints simple evaluation metrics.