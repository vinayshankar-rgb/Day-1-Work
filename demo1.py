from sentence_transformers import SentenceTransformer

# 1. Load the model
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define your documents (The Database)
documents = [
    "Python is an interpreted, high-level programming language created by Guido van Rossum.",
    "Machine Learning focuses on developing algorithms that allow computers to learn from data.",
    "Deep Learning uses multi-layered artificial neural networks for complex pattern recognition.",
    "Natural Language Processing (NLP) helps computers understand and interact with human speech.",
    "Chennai, the capital of Tamil Nadu, is a major cultural and information technology hub.",
    "Retrieval-Augmented Generation (RAG) optimizes LLM outputs by querying external knowledge bases.",
    "Vector databases index high-dimensional embeddings to perform rapid semantic similarity searches."
]

# 3. Define your search query
query = "How do computers understand human talk?"

# 4. Compute embeddings
print("Computing vector embeddings...")
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# 5. Calculate semantic similarity scores
similarities = model.similarity(query_embedding, doc_embeddings)[0]

print(f"\n--- Semantic Search Results for Query: '{query}' ---")
# Sort and print results from highest to lowest similarity score
results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
for doc, score in results:
    print(f"\nScore: {score.item() * 100:.2f}%")
    print(f"Match: {doc}")
