import os
from sentence_transformers import SentenceTransformer
from config.settings import Config

def get_local_embedder():
    """Initialize local embedding model"""
    model_path = Config.EMBEDDING_MODEL_PATH
    print(f"Loading local embedding model from: {model_path}")
    
    try:
        local_embedder = SentenceTransformer(model_path, device="cuda", trust_remote_code=True)
        print("Local embedding model loaded successfully on CUDA")
        return local_embedder
    except Exception as e:
        print(f"Error loading local model, falling back to CPU: {e}")
        local_embedder = SentenceTransformer(model_path, device="cpu", trust_remote_code=True)
        return local_embedder

class LocalEmbeddings:
    """Wrapper for local embedding model compatible with neo4j-graphrag"""
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()
    
    def embed_documents(self, texts):
        return [self.model.encode([text])[0].tolist() for text in texts]