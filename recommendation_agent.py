import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
from feedback_tools import log_feedback

# Load your existing data
print("🤖 Loading Goodbooks-10k subset + SentenceTransformer embeddings...")
books = pd.read_csv("books_subset.csv")
embeddings = np.load("book_embeddings.npy")
similarity_matrix = cosine_similarity(embeddings)

from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3.2:1b")

def retrieve_similar_books(query_book_title: str, top_k=5) -> str:
    """TOOL 1: RAG Retrieval using vector similarity"""
    matches = books[books["title"].str.contains(query_book_title, case=False, na=False)]
    if matches.empty:
        return "❌ No matching book found"
    
    idx = matches.index[0]
    sim_scores = sorted(enumerate(similarity_matrix[idx]), key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx][:top_k]
    
    results = []
    for i, (book_idx, score) in enumerate(sim_scores):
        book = books.iloc[book_idx]
        results.append(f"{i+1}. {book['title']} by {book['authors']} (⭐{book['average_rating']:.1f})")
    return "\n".join(results)

def agent_reasoning(query_book_title: str) -> str:
    """TOOL 2: LLM Reasoning + Reflection"""
    raw_recs = retrieve_similar_books(query_book_title)
    
    prompt = f"""You are an intelligent book recommendation agent.

QUERY BOOK: {query_book_title}
RAW RECOMMENDATIONS:
{raw_recs}

REASONING TASK:
1. Analyze the recommendations (ratings, authors, relevance)
2. Reflect on quality: "These work because..." 
3. Select TOP 3-5 recommendations
4. Explain why they match the query book

RESPONSE FORMAT:
"ANALYSIS: [your reasoning]
REFLECTION: [why these are good matches]  
FINAL RECOMMENDATIONS: [numbered list]"
"""
    
    response = llm.invoke(prompt)
    return response

print("\n🎯 CAPSTONE REQUIREMENTS ✓")
print("✅ Data Prep: Goodbooks-10k + embeddings")
print("✅ RAG: Vector similarity tool") 
print("✅ Reasoning: Ollama LLM analysis")
print("✅ Tools: retrieve_similar_books() + log_feedback()")
print("✅ Reflection: LLM self-analysis")
print("✅ Evaluation: feedback.csv ✓")
print("\n📖 Try: 'Hunger Games', 'Harry Potter', '1984'\n")

if __name__ == "__main__":
    user_id = "capstone_user"
    
    while True:
        query = input("📖 Book you like (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
        
        print(f"\n🔍 [TOOL 1] RAG Retrieval...")
        raw_results = retrieve_similar_books(query)
        print(raw_results)
        
        print("\n🧠 [TOOL 2] LLM Reasoning + Reflection...")
        reasoning = agent_reasoning(query)
        print(reasoning)
        
        # Interactive feedback (TOOL 3)
        print("\n👍 [TOOL 3] Evaluation Feedback:")
        for i, line in enumerate(raw_results.split('\n')):
            if i < 5 and line.strip():
                rec_title = line.split(' by ')[0].replace(f"{i+1}. ", "")
                while True:
                    feedback = input(f"Like '{rec_title[:40]}...' (y/n): ").strip().lower()
                    if feedback in ['y', 'n']:
                        liked = feedback == 'y'
                        log_feedback(user_id, query, rec_title, liked)
                        print(f"   → {'👍' if liked else '👎'} Logged to feedback.csv")
                        break
        
        print("\n" + "="*80)