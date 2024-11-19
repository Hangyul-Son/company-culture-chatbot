from embedding_pipeline import SentenceTransformerModel
from vector_db import QdrantDB
from vector_search_system import VectorSearchSystem

# Initialize components
embedding_model = SentenceTransformerModel('distilbert-base-uncased')
vector_db = QdrantDB(dimension=768)  # distilbert's output dimension
search_system = VectorSearchSystem(embedding_model, vector_db)

# Add some example texts
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Natural language processing helps computers understand text"
]

# Add texts to the vector database
search_system.add_texts(texts)

# Search for similar texts
query = "AI and machine learning"
results = search_system.search(query, limit=2)

# Print results
for result in results:
    print(f"Distance: {result['score']:.4f} | Text: {result['text']}")
