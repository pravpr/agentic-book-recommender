import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feedback_tools import log_feedback

BOOKS_SUBSET = "books_subset.csv"
EMBEDDINGS_NPY = "book_embeddings.npy"

print("Loading books and embeddings...")
books = pd.read_csv(BOOKS_SUBSET)
embeddings = np.load(EMBEDDINGS_NPY)

similarity_matrix = cosine_similarity(embeddings)

# ---- TOOL 1: retrieval ----
def tool_recommend_books(idx, top_n=10):
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx]
    top_idx = [x[0] for x in sim_scores[:top_n]]
    return books.iloc[top_idx].copy()

def find_book_index_by_title(query_title: str):
    matches = books[books["title"].str.contains(query_title, case=False, na=False)]
    if matches.empty:
        return None, None
    idx = matches.index[0]
    return idx, matches.iloc[0]

# ---- Reasoning / reflection ----
def reflect_and_adjust(candidates: pd.DataFrame, max_final: int = 5):
    # Rule 1: filter out very low rated books (< 3.5)
    filtered = candidates[candidates["average_rating"] >= 3.5]
    reflection_notes = []

    if filtered.empty:
        reflection_notes.append("All candidates had low ratings; using original list.")
        filtered = candidates.copy()
    else:
        reflection_notes.append("Filtered out low-rating books (< 3.5).")

    # Rule 2: author diversity – avoid more than 3 from same author in final list
    final = []
    author_counts = {}

    for _, row in filtered.iterrows():
        author = row["authors"]
        count = author_counts.get(author, 0)
        if count >= 3:
            continue
        final.append(row)
        author_counts[author] = count + 1
        if len(final) >= max_final:
            break

    if len(final) < max_final:
        reflection_notes.append("Could not fill full list with diverse authors; some authors repeated.")

    final_df = pd.DataFrame(final)
    return final_df, reflection_notes

if __name__ == "__main__":
    print("🤖 Book Recommendation Agent - Capstone Demo")
    print("=" * 60)
    print("Data: Goodbooks-10k subset (2K books) + SentenceTransformer embeddings")
    print("Try: 'Hunger Games', 'Harry Potter', '1984'\n")
    
    user_id = "demo_user"
    
    while True:
        query = input("📖 Book you like (or 'q' to quit): ").strip()
        if query.lower() == "q":
            break

        idx, book_row = find_book_index_by_title(query)
        if idx is None:
            print("❌ No match. Try 'Hunger Games', 'Harry Potter', '1984'")
            continue

        print(f"\n✅ Found: {book_row['title']} by {book_row['authors']} (⭐{book_row['average_rating']:.1f})")
        
        print("\n🔍 [TOOL CALL] Retrieval...")
        candidates = tool_recommend_books(idx, top_n=15)

        print("🧠 [REASONING] Reflection...")
        final_recs, notes = reflect_and_adjust(candidates)
        for n in notes:
            print(f"   {n}")

        print("\n📚 RECOMMENDATIONS:")
        for i, (_, r) in enumerate(final_recs.iterrows(), 1):
            print(f"  {i}. {r['title']} by {r['authors']} (⭐{r['average_rating']:.1f})")

        print("\n👍 [FEEDBACK] Logging to feedback.csv...")
        total_recs = len(final_recs)
        likes = 0
        for _, r in final_recs.iterrows():
            while True:
                ans = input(f"  Like '{r['title'][:30]}...' (y/n): ").strip().lower()
                if ans in ["y", "n"]:
                    liked = (ans == "y")
                    log_feedback(user_id, book_row["title"], r["title"], liked)
                    if liked:
                        likes += 1
                    print(f"     → {'👍' if liked else '👎'} Logged")
                    break
                print("     Enter 'y' or 'n'")
        
        print(f"\n✅ Session complete: {likes}/{total_recs} liked ({likes/total_recs:.0%})")
        print("-" * 60)
