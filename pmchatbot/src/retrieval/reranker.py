import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from utils.logging_utils import log

class Reranker:
    """Generic reranker for improving retrieval quality"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        log(f"Loading Reranker from: {model_path} on device: {self.device.upper()}", level="debug")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token or "[PAD]"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.model.eval()
            log(f"Reranker loaded successfully on {self.device.upper()}", level="debug")
        except Exception as e:
            log(f"Error loading reranker model: {e}", level="error")
            raise
    
    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def rerank(self, query: str, chunks: List[Tuple[str, dict]], top_k: int = None) -> List[Tuple[str, dict, float]]:
        if not chunks:
            return []
        try:
            pairs = [(query, chunk[0]) for chunk in chunks]
            scores = self._score_pairs(pairs)
            reranked_results = []
            for i, (text, metadata) in enumerate(chunks):
                rerank_score = float(scores[i])
                enhanced_metadata = metadata.copy() if metadata else {}
                enhanced_metadata['rerank_score'] = rerank_score
                enhanced_metadata['original_score'] = enhanced_metadata.get('score', 0.0)
                enhanced_metadata['score'] = rerank_score
                reranked_results.append((text, enhanced_metadata, rerank_score))
            reranked_results.sort(key=lambda x: x[2], reverse=True)
            if top_k:
                reranked_results = reranked_results[:top_k]
            return reranked_results
        except Exception as e:
            log(f"Error during reranking: {e}", level="error")
            return [(text, metadata, 0.0) for text, metadata in chunks]
    
    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        scores = []
        batch_size = 1
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = self._score_batch(batch_pairs)
            scores.extend(batch_scores)
        return scores
    
    def _score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        try:
            queries = [q for q, d in pairs]
            docs = [d for q, d in pairs]
            inputs = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.sigmoid(outputs.logits.squeeze(-1))
            scores = scores.cpu().numpy()
            if scores.ndim > 1:
                scores = scores.flatten()
            return scores.tolist()
        except Exception as e:
            log(f"Error scoring batch in reranker: {e}", level="error")
            return [0.0] * len(pairs)
    
    def __del__(self):
        if hasattr(self, 'model') and self.device == "cuda":
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

def get_reranker(device: str = "auto") -> Reranker:
    model_path = os.getenv("RERANKER_MODEL_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    return Reranker(model_path=model_path, device=device)