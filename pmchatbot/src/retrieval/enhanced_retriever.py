import traceback
from typing import List, Optional
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem, RetrieverResult
from .retriever_setup import setup_retriever
from .reranker import get_reranker
from config.settings import Config
from utils.logging_utils import log

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
            
            # Log the ranker settings for debugging
            log(f"Using hybrid ranker: {config.HYBRID_RANKER}", level="debug")
            if config.HYBRID_RANKER.lower() == "linear":
                log(f"Using alpha: {config.HYBRID_ALPHA}", level="debug")
            
            # Try to use ranker parameters if the retriever supports them
            base_result = self._search_with_ranker_support(query, top_k)

            if not self.use_reranker or not self.reranker or not base_result.items:
                return base_result

            # Prepare chunks for reranking
            chunks = []
            for item in base_result.items:
                text = self._extract_text_content(item.content)
                metadata = item.metadata or {}
                # Preserve the hybrid search score
                if 'score' in metadata:
                    metadata['hybrid_score'] = metadata['score']
                chunks.append((text, metadata))

            # Log original vs hybrid scores for debugging
            log(f"Retrieved {len(chunks)} chunks for reranking", level="debug")
            for i, (text, metadata) in enumerate(chunks[:3]):  # Log first 3 for debugging
                score = metadata.get('score', 'N/A')
                log(f"Chunk {i+1} hybrid_score: {score}", level="debug")

            # Rerank and select RERANKER_TOP_K best
            reranked_results = self.reranker.rerank(query, chunks, top_k=config.RERANKER_TOP_K)

            # Convert back to RetrieverResultItem format
            reranked_items = []
            for text, metadata, rerank_score in reranked_results:
                metadata = metadata.copy() if metadata else {}
                metadata['rerank_score'] = rerank_score
                metadata['final_score'] = rerank_score
                # Log scores for debugging
                hybrid_score = metadata.get('hybrid_score', 'N/A')
                log(f"Item hybrid_score: {hybrid_score}, rerank_score: {rerank_score}", level="debug")
                
                item = RetrieverResultItem(
                    content=text,
                    metadata=metadata
                )
                reranked_items.append(item)

            enhanced_result = RetrieverResult(
                items=reranked_items,
                metadata={
                    "reranked": True, 
                    "original_count": len(base_result.items),
                    "hybrid_ranker": config.HYBRID_RANKER,
                    "hybrid_alpha": config.HYBRID_ALPHA if config.HYBRID_RANKER.lower() == "linear" else None
                }
            )
            return enhanced_result
        except Exception as e:
            traceback.print_exc()
            # Fallback to basic search
            return self.base_retriever.search(query, top_k=top_k)
    
    def _search_with_ranker_support(self, query: str, top_k: int) -> RetrieverResult:
        """
        Try different ways to use ranker parameters with the retriever
        """
        try:
            # Method 1: Try with keyword arguments if supported
            try:
                from neo4j_graphrag.types import HybridSearchRanker
                ranker = HybridSearchRanker(config.HYBRID_RANKER.lower())
                
                if config.HYBRID_RANKER.lower() == "linear":
                    result = self.base_retriever.search(
                        query, 
                        top_k=top_k, 
                        ranker=ranker, 
                        alpha=config.HYBRID_ALPHA
                    )
                    log("Used search with linear ranker and alpha", level="debug")
                    return result
                else:
                    result = self.base_retriever.search(
                        query, 
                        top_k=top_k, 
                        ranker=ranker
                    )
                    log("Used search with naive ranker", level="debug")
                    return result
            except TypeError as e:
                log(f"Ranker parameters not supported in search method: {e}", level="debug")
                pass
            
            # Method 2: Try with search model if supported
            try:
                from neo4j_graphrag.types import HybridSearchModel, HybridSearchRanker
                search_model = HybridSearchModel(
                    query_text=query,
                    top_k=top_k,
                    ranker=HybridSearchRanker(config.HYBRID_RANKER.lower()),
                    alpha=config.HYBRID_ALPHA if config.HYBRID_RANKER.lower() == "linear" else None
                )
                
                # Check if retriever has a method to accept search model
                if hasattr(self.base_retriever, 'search') and callable(getattr(self.base_retriever, 'search')):
                    # Try passing the search model
                    result = self.base_retriever.search(search_model)
                    log("Used search with HybridSearchModel", level="debug")
                    return result
            except Exception as e:
                log(f"HybridSearchModel not supported: {e}", level="debug")
                pass
            
            # Method 3: Fallback to basic search
            log("Falling back to basic search (ranker settings may not be applied)", level="debug")
            return self.base_retriever.search(query, top_k=top_k)
            
        except Exception as e:
            log(f"Error in ranker search: {e}", level="debug")
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