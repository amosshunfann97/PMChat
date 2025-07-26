from sentence_transformers import SentenceTransformer
from config.settings import Config
from utils.logging_utils import log

config = Config()

def get_local_embedder(device="cpu"):
    """Initialize local embedding model on specified device"""
    model_path = config.EMBEDDING_MODEL_PATH
    log(f"Loading local embedding model from: {model_path} on device: {device.upper()}", level="debug")
    try:
        local_embedder = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        log(f"Local embedding model loaded successfully on {device.upper()}", level="debug")
        return local_embedder
    except Exception as e:
        log(f"Error loading local model on {device}: {e}", level="error")
        return None

class LocalEmbeddings:
    """Wrapper for local embedding model compatible with neo4j-graphrag"""
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()
    
    def embed_documents(self, texts):
        return [self.model.encode([text])[0].tolist() for text in texts]