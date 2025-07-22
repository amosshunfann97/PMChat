import traceback
from typing import List, Optional
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem, RetrieverResult
from .retriever_setup import setup_retriever
from .reranker import get_reranker
from config.settings import Config

config = Config()

class EnhancedRetriever:
    """Enhanced retriever with reranking capability"""
    
    def __init__(self, base_retriever: HybridCypherRetriever, use_reranker: bool = True):
        self.base_retriever = base_retriever
        self.use_reranker = use_reranker
        self.reranker = None
        
        if self.use_reranker:
            try:
                self.reranker = get_reranker(device=config.RERANKER_DEVICE)
            except Exception as e:
                self.use_reranker = False

    def search(self, query: str, top_k: int = None) -> RetrieverResult:
        """
        Search with optional reranking

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            RetrieverResult with potentially reranked results
        """
        try:
            # Use provided top_k or fallback to config
            if top_k is None:
                top_k = config.RETRIEVER_TOP_K
            base_result = self.base_retriever.search(query, top_k=top_k)

            if not self.use_reranker or not self.reranker or not base_result.items:
                return base_result

            # Prepare chunks for reranking
            chunks = []
            for item in base_result.items:
                text = self._extract_text_content(item.content)
                metadata = item.metadata or {}
                chunks.append((text, metadata))

            # Rerank and select RERANKER_TOP_K best
            reranked_results = self.reranker.rerank(query, chunks, top_k=config.RERANKER_TOP_K)

            # Convert back to RetrieverResultItem format
            reranked_items = []
            for text, metadata, rerank_score in reranked_results:
                metadata = metadata.copy() if metadata else {}
                metadata['rerank_score'] = rerank_score
                item = RetrieverResultItem(
                    content=text,
                    metadata=metadata
                )
                reranked_items.append(item)

            enhanced_result = RetrieverResult(
                items=reranked_items,
                metadata={"reranked": True, "original_count": len(base_result.items)}
            )
            return enhanced_result
        except Exception as e:
            traceback.print_exc()
            return self.base_retriever.search(query, top_k=top_k)
    
    def _extract_text_content(self, content) -> str:
        """Extract text content from various content formats"""
        content_str = str(content)
        if content_str.startswith('<Record text="') and content_str.endswith('">'):
            return content_str[14:-2]
        return content_str

def setup_enhanced_retriever(driver, chunk_type: str, local_embedder, use_reranker: bool = None) -> Optional[EnhancedRetriever]:
    """
    Setup enhanced retriever with optional reranking
    
    Args:
        driver: Neo4j driver
        chunk_type: Type of chunks to retrieve
        local_embedder: Local embedding model
        use_reranker: Whether to use reranker (None for config default)
        
    Returns:
        EnhancedRetriever instance or None if setup failed
    """
    try:
        # Setup base retriever
        base_retriever = setup_retriever(driver, chunk_type, local_embedder)
        if not base_retriever:
            return None
        
        # Use config default if not specified
        if use_reranker is None:
            use_reranker = config.USE_RERANKER
        
        # Create enhanced retriever
        enhanced_retriever = EnhancedRetriever(
            base_retriever=base_retriever,
            use_reranker=use_reranker
        )
        
        return enhanced_retriever
        
    except Exception as e:
        traceback.print_exc()