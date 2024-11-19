from typing import List, Dict, Optional
from embedding_pipeline import EmbeddingModel
from vector_db import VectorDB

# Main System
class VectorSearchSystem:
    def __init__(self, embedding_model: EmbeddingModel, vector_db: VectorDB):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
    
    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        embeddings = self.embedding_model.encode(texts)
        self.vector_db.add_texts(texts, embeddings, metadata)
    
    def search(self, query: str, limit: int = 5, filter_conditions: Optional[Dict] = None) -> List[Dict]:
        query_vector = self.embedding_model.encode([query])[0]
        return self.vector_db.search(query_vector, limit, filter_conditions)