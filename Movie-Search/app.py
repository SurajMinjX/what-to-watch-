import json
import re
import time
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# ─── CONFIG ─────────────────────────────────────────────
RRF_K = 60
TOP_K = 5

# ─── LOAD MOVIES ────────────────────────────────────────
def load_movies(path="movies.jsonl"):
    movies = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                movies.append(json.loads(line))
    return movies

# ─── TF-IDF ENCODER (no torch needed) ──────────────────
class TFIDFEncoder:
    def __init__(self, texts):
        self.vec = TfidfVectorizer()
        self.embeddings = self.vec.fit_transform(texts).toarray()
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

    def encode(self, query):
        vec = self.vec.transform([query]).toarray()[0]
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

# ─── CONSTRAINT PARSER ──────────────────────────────────
def extract_constraints(query):
    q = query.lower()
    c = {}
    if any(w in q for w in ["indian", "hindi", "bollywood", "telugu", "kannada", "tamil"]):
        c["language"] = "Indian"
    if "english" in q or "hollywood" in q or "western" in q:
        c["language"] = "English"
    if any(w in q for w in ["light", "funny", "fun", "feel good", "chill", "relax", "not too dumb"]):
        c["mood"] = "light"
    if any(w in q for w in ["dark", "serious", "tense", "intense", "scary", "thrilling"]):
        c["mood"] = "dark"
    if "action" in q:
        c["genre"] = "Action"
    if "comedy" in q or "funny" in q:
        c["genre"] = "Comedy"
    if "thriller" in q:
        c["genre"] = "Thriller"
    if "romance" in q or "love" in q:
        c["genre"] = "Romance"
    if "sci-fi" in q or "scifi" in q or "science fiction" in q:
        c["genre"] = "Sci-Fi"
    if "drama" in q:
        c["genre"] = "Drama"
    if "horror" in q:
        c["genre"] = "Horror"
    if "animation" in q or "animated" in q or "anime" in q:
        c["genre"] = "Animation"
    m = re.search(r"under (\d+) (?:hours?|hrs?|min(?:utes)?)", q)
    if m:
        val = int(m.group(1))
        c["max_duration"] = val * 60 if val < 20 else val
    m = re.search(r"(?:rated|above|over) (\d+\.?\d*)", q)
    if m:
        c["min_rating"] = float(m.group(1))
    return c

# ─── CONSTRAINT BONUS ───────────────────────────────────
def constraint_bonus(movie, constraints):
    bonus = 0.0
    if "language" in constraints:
        if constraints["language"] == "Indian":
            if movie["language"] in ["Hindi", "Telugu", "Kannada", "Tamil", "Bengali", "Malayalam"]:
                bonus += 0.3
        elif constraints["language"] == "English":
            if movie["language"] == "English":
                bonus += 0.3
    if "mood" in constraints:
        if movie["mood"] == constraints["mood"]:
            bonus += 0.2
    if "genre" in constraints:
        if constraints["genre"] in movie["genre"]:
            bonus += 0.2
    if "max_duration" in constraints:
        if movie["duration"] <= constraints["max_duration"]:
            bonus += 0.15
    if "min_rating" in constraints:
        if movie["vote"] >= constraints["min_rating"]:
            bonus += 0.15
    return bonus

# ─── EXPLAIN MATCH ──────────────────────────────────────
def explain_match(movie, constraints):
    reasons = []
    if "language" in constraints:
        if constraints["language"] == "Indian" and movie["language"] in ["Hindi", "Telugu", "Kannada", "Tamil", "Bengali", "Malayalam"]:
            reasons.append(f"{movie['language']} film")
        elif constraints["language"] == "English" and movie["language"] == "English":
            reasons.append("English film")
    if "mood" in constraints and movie["mood"] == constraints["mood"]:
        reasons.append("Light & fun" if movie["mood"] == "light" else "Dark & intense")
    if "genre" in constraints and constraints["genre"] in movie["genre"]:
        reasons.append(constraints["genre"])
    if "max_duration" in constraints and movie["duration"] <= constraints["max_duration"]:
        reasons.append(f"Only {movie['duration']} min")
    if movie["vote"] >= 8.3:
        reasons.append(f"Highly rated ({movie['vote']})")
    return " | ".join(reasons[:3]) if reasons else "Good match"

# ─── SEARCH ENGINE ──────────────────────────────────────
class MovieSearchEngine:
    def __init__(self, movies):
        self.movies = movies
        self.search_texts = [
            f"{m['title']} {m['genre']} {m['plot']} {m['language']} {m['mood']}"
            for m in movies
        ]
        self.tokenized = [t.lower().split() for t in self.search_texts]
        self.bm25 = BM25Okapi(self.tokenized)
        self.encoder = TFIDFEncoder(self.search_texts)
        print(f"✅ Indexed {len(movies)} movies")

    def search(self, query, top_k=TOP_K):
        t0 = time.perf_counter()
        constraints = extract_constraints(query)

        # BM25
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k * 5]

        # TF-IDF Vector
        q_emb = self.encoder.encode(query)
        cos_scores = self.encoder.embeddings @ q_emb
        vec_top = np.argsort(cos_scores)[::-1][:top_k * 5]

        # RRF + constraint bonus
        rrf = defaultdict(float)
        for rank, idx in enumerate(bm25_top, 1):
            rrf[int(idx)] += 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(vec_top, 1):
            rrf[int(idx)] += 1.0 / (RRF_K + rank)

        for idx in rrf:
            rrf[idx] += constraint_bonus(self.movies[idx], constraints)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ms = (time.perf_counter() - t0) * 1000

        results = []
        for idx, score in ranked:
            m = self.movies[idx]
            results.append({
                "rank": len(results) + 1,
                "title": m["title"],
                "year": m["year"],
                "genre": m["genre"],
                "rating": m["vote"],
                "duration": m["duration"],
                "language": m["language"],
                "plot": m["plot"],
                "why": explain_match(m, constraints)
            })
        return results, ms, constraints

# ─── MAIN ───────────────────────────────────────────────
if __name__ == "__main__":
    movies = load_movies()
    engine = MovieSearchEngine(movies)

    print("\n" + "═" * 55)
    print("  🎬  WHAT TO WATCH TONIGHT")
    print("  Hybrid Movie Finder (BM25 + TF-IDF + RRF)")
    print("═" * 55)
    print("  Try:")
    print("    • indian action, under 2 hours")
    print("    • light comedy, not too long")
    print("    • dark thriller, english, rated 8+")
    print("    • sci-fi about space")
    print("    • feel good indian film")
    print("  Type 'quit' to exit\n")

    while True:
        try:
            q = input("🔍 ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ('quit', 'exit', 'q'):
            print("👋 Good night! Pick a movie and enjoy.")
            break
        if not q:
            continue

        results, ms, constraints = engine.search(q)
        print(f"\n⚡ {ms:.0f}ms" + (f"  |  Filters: {constraints}" if constraints else ""))
        print("─" * 55)
        for r in results:
            print(f"  {r['rank']}. {r['title']} ({r['year']})  ⭐ {r['rating']}  |  {r['duration']} min  |  {r['language']}")
            print(f"     [{r['genre']}]")
            print(f"     {r['plot'][:90]}...")
            print(f"     ✅ Why: {r['why']}")
            print()   