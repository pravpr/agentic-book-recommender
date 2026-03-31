# Agentic Book Recommender

## Technology Stack

LLM: Ollama llama3.2:1b (local)

Embeddings: SentenceTransformer all-MiniLM-L6-v2

Vector Store: NumPy cosine similarity (local embeddings)

Data Source: Goodbooks-10k subset (books_subset.csv)

Retrieval: FAISS-style nearest neighbor search

Agent Framework: Custom LangChain-inspired tools

Persistence: NumPy .npy files (book_embeddings.npy)

Feedback Storage: Pandas CSV (feedback.csv)

Evaluation: User acceptance rate metrics

UI: Python CLI interactive loop

Environment: Python venv (Windows local)

Documentation: Markdown (README.md, architecture.md)


## 🚀 Quick Start (Mentor Test - 3 minutes)

```bash
# 1. Clone + install
git clone https://github.com/YOUR_USERNAME/agentic-book-recommender
cd agentic-book-recommender
pip install -r requirements.txt

# 2. Ollama (one-time setup)
ollama pull llama3.2:1b

# 3. Run agent
python recommendation_agent.py
```
