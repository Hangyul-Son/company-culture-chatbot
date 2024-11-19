from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import faiss

class VectorDB(ABC):
    """Abstract base class for vector database implementations."""
    
    @abstractmethod
    def initialize(self, dimension: int) -> None:
        """Initialize the vector database with given dimension."""
        pass
    
    @abstractmethod
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the database."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> Optional[int]:
        """Get dimension of vectors in the database."""
        pass

class FaissVectorDB(VectorDB):
    """FAISS implementation of vector database."""
    
    def __init__(self, index_type: str = "flat"):
        self.index = None
        self.index_type = index_type
    
    def initialize(self, dimension: int) -> None:
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add(self, embeddings: np.ndarray) -> None:
        self.index.add(embeddings)
    
    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(query_embedding.reshape(1, -1))
        return self.index.search(query, k)
    
    def get_dimension(self) -> Optional[int]:
        return self.index.d if self.index else None 