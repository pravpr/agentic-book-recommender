import pandas as pd

FEEDBACK_FILE = "feedback.csv"

def main():
    try:
        fb = pd.read_csv(FEEDBACK_FILE)
    except FileNotFoundError:
        print("No feedback.csv found yet. Run the agent and give feedback first.")
        return

    if fb.empty:
        print("feedback.csv is empty.")
        return

    total = len(fb)
    liked = fb["liked"].sum()
    acceptance_rate = liked / total

    # Average number of likes per query_title (rough measure)
    likes_per_query = fb.groupby("query_title")["liked"].mean()

    # Diversity: number of unique recommended titles and authors
    unique_recs = fb["recommended_title"].nunique()

    print(f"Total feedback rows: {total}")
    print(f"Total likes: {liked}")
    print(f"Acceptance rate: {acceptance_rate:.2f}")
    print(f"Unique recommended titles: {unique_recs}")
    print("\nSample likes per query_title:")
    print(likes_per_query.head())

if __name__ == "__main__":
    main()
