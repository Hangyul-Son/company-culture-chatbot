from embedding_pipeline import EmbeddingPipeline

# Initialize the pipeline (it will use distilbert-base-uncased by default)
pipeline = EmbeddingPipeline()

# Add some example texts
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Natural language processing helps computers understand text"
]

# Add texts to the vector database
pipeline.add_texts(texts)

# Search for similar texts
query = "AI and machine learning"
results = pipeline.search(query, k=2)

# Print results
for text, distance in zip(results["texts"], results["distances"]):
    print(f"Distance: {distance:.4f} | Text: {text}")
