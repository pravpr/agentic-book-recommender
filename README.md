# Agentic Book Recommender (Capstone v2)

**Agentic AI system** 
- ✅ **LLM**: Ollama llama3.2:1b (local, no API key)
- ✅ **RAG**: SentenceTransformer embeddings + cosine similarity
- ✅ **Agent Reasoning**: LLM analysis + reflection
- ✅ **Tools**: Retrieval + Feedback logging
- ✅ **Evaluation**: feedback.csv metrics

## Quick Demo (5 minutes)
```bash
git clone https://github.com/YOUR_USERNAME/agentic-book-recommender
cd agentic-book-recommender
pip install -r requirements.txt

# Install Ollama: ollama.com/download
ollama pull llama3.2:1b

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


## Architecture

subgraph DataPrep["Data Preparation & Embeddings"]
    A["Goodbooks-10k<br/>books.csv + ratings.csv"]
    B["prepare_goodbooks_subset.py<br/>→ books_subset.csv"]
    C["build_book_embeddings.py<br/>SentenceTransformer<br/>→ book_embeddings.npy"]
    A --> B --> C
end

subgraph AgentRuntime["Agent Runtime"]
    U["User Input<br/>Book you like"]
    F["find_book_index_by_title"]
    T1["tool_recommend_books<br/>cosine similarity"]
    R["reflect_and_adjust<br/>rating filter + diversity"]
    OUT["Final Recommendations"]
    FBQ["User Feedback y/n"]
    T2["log_feedback<br/>→ feedback.csv"]
end

subgraph Evaluation["Evaluation"]
    E["evaluate_agent.py<br/>Reads feedback.csv<br/>→ metrics"]
end

C --> T1
U --> F --> T1 --> R --> OUT --> FBQ --> T2
T2 --> E
