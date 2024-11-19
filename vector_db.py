from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Vector Database Interface
class VectorDB(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadata: Optional[List[Dict]] = None) -> None:
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, limit: int = 5, filter_conditions: Optional[Dict] = None) -> List[Dict]:
        pass

# Concrete Vector DB - Qdrant
class QdrantDB(VectorDB):
    def __init__(self, dimension: int, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "text_store"
        
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
    
    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadata: Optional[List[Dict]] = None) -> None:
        points = []
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            payload = {"text": text}
            if metadata and len(metadata) > idx:
                payload.update(metadata[idx])
            
            points.append(PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=payload
            ))
        
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query_vector: np.ndarray, limit: int = 5, filter_conditions: Optional[Dict] = None) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {k:v for k,v in hit.payload.items() if k != "text"}
            }
            for hit in results
        ]
