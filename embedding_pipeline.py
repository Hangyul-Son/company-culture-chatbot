from typing import List, Union, Dict, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from vector_db import VectorDB, FaissVectorDB

class EmbeddingPipeline:
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        vector_db: Optional[VectorDB] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.vector_db = vector_db or FaissVectorDB()
        self.text_store = []
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for single text or list of texts."""
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy().astype('float32')
    
    def initialize_vector_db(self, dimension: int = None) -> None:
        """Initialize vector database."""
        if dimension is None and len(self.text_store) > 0:
            sample_embedding = self.embed_text(self.text_store[0])
            dimension = sample_embedding.shape[1]
        elif dimension is None:
            raise ValueError("Must provide dimension if no texts have been indexed")
            
        self.vector_db.initialize(dimension)
        
    def add_texts(self, texts: Union[str, List[str]]) -> None:
        """Add texts to the vector database."""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.embed_text(texts)
        
        if self.vector_db.get_dimension() is None:
            self.initialize_vector_db(embeddings.shape[1])
            
        self.vector_db.add(embeddings)
        self.text_store.extend(texts)
        
    def search(self, query: str, k: int = 5) -> Dict[str, Union[List[str], List[float]]]:
        """Search for similar texts."""
        if not self.text_store:
            raise ValueError("No texts have been indexed yet")
            
        query_embedding = self.embed_text(query)
        distances, indices = self.vector_db.search(query_embedding, k)
        
        return {
            "texts": [self.text_store[i] for i in indices[0]],
            "distances": distances[0].tolist()
        }
    
    def change_model(self, new_model_name: str) -> None:
        """Change the underlying model and tokenizer."""
        self.model_name = new_model_name
        self._initialize_model()
        
        if self.text_store:
            stored_texts = self.text_store.copy()
            self.text_store = []
            self.vector_db = type(self.vector_db)()  # Create new instance of same type
            self.add_texts(stored_texts)
    
    def change_vector_db(self, new_vector_db: VectorDB) -> None:
        """Change the vector database implementation."""
        self.vector_db = new_vector_db
        
        if self.text_store:
            stored_texts = self.text_store.copy()
            self.text_store = []
            self.add_texts(stored_texts)