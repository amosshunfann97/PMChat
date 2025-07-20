import traceback
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

def setup_retriever(driver, chunk_type, local_embedder):
    """Setup retriever for specific chunk type"""
    try:
        # Check index status
        vector_index_name, vector_index_ready, fulltext_index_name, fulltext_index_ready = _check_index_status(
            driver, chunk_type
        )
        
        # Create local embedder wrapper
        embedder = _create_embedder_wrapper(local_embedder)
        
        # Check if indexes are ready
        if not (vector_index_ready and fulltext_index_ready and vector_index_name and fulltext_index_name):
            print(f"Error: Both vector and fulltext indexes are required for {chunk_type} HybridCypherRetriever")
            return None
        
        # Get retrieval query for chunk type
        retrieval_query = _get_retrieval_query(chunk_type)
        
        # Create retriever
        retriever = HybridCypherRetriever(
            driver=driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=_custom_result_formatter
        )
        return retriever
    except Exception as e:
        print(f"Error setting up retriever for {chunk_type}: {e}")
        traceback.print_exc()
        return None

def _check_index_status(driver, chunk_type):
    """Check the status of vector and fulltext indexes"""
    with driver.session() as session:
        result = session.run("SHOW INDEXES YIELD name, state, type, labelsOrTypes, properties")
        all_indexes = list(result)
        
        vector_index_name = None
        vector_index_ready = False
        fulltext_index_name = None
        fulltext_index_ready = False
        
        for idx in all_indexes:
            if (idx['type'] == 'VECTOR' and 
                chunk_type in str(idx['labelsOrTypes']) and 
                'embedding' in str(idx['properties'])):
                vector_index_name = idx['name']
                vector_index_ready = idx['state'] == 'ONLINE'
            if (idx['type'] == 'FULLTEXT' and 
                chunk_type in str(idx['labelsOrTypes'])):
                fulltext_index_name = idx['name']
                fulltext_index_ready = idx['state'] == 'ONLINE'
    
    return vector_index_name, vector_index_ready, fulltext_index_name, fulltext_index_ready

def _create_embedder_wrapper(local_embedder):
    """Create wrapper for local embedding model"""
    class LocalEmbeddings:
        def __init__(self, model):
            self.model = model
        
        def embed_query(self, text):
            return self.model.encode([text])[0].tolist()
        
        def embed_documents(self, texts):
            return [self.model.encode([text])[0].tolist() for text in texts]
    
    return LocalEmbeddings(local_embedder)

def _get_retrieval_query(chunk_type):
    """Get the appropriate retrieval query for chunk type"""
    if chunk_type == "ActivityChunk":
        return """
            MATCH (node:ActivityChunk)-[:DESCRIBES]->(activity:Activity)
            RETURN node.text AS text,
                   score AS score,
                   {
                       activity_name: activity.name,
                       is_start: activity.is_start,
                       is_end: activity.is_end,
                       start_count: activity.start_count,
                       end_count: activity.end_count,
                       type: node.type,
                       source: node.source
                   } AS metadata
        """
    elif chunk_type == "ProcessChunk":
        return """
            MATCH (node)-[:DESCRIBES]->(path:ProcessPath)
            RETURN node.text AS text,
                   score AS score,
                   {
                       path_string: path.path_string,
                       rank: path.rank,
                       frequency: path.frequency,
                       type: node.type,
                       source: node.source
                   } AS metadata
        """
    elif chunk_type == "VariantChunk":
        return """
            MATCH (node)-[:DESCRIBES]->(variant:CaseVariant)
            RETURN node.text AS text,
                   score AS score,
                   {
                       variant_string: variant.variant_string,
                       rank: variant.rank,
                       frequency: variant.frequency,
                       type: node.type,
                       source: node.source
                   } AS metadata
        """
    else:
        raise ValueError(f"Unknown chunk_type: {chunk_type}")

def _custom_result_formatter(record):
    """Custom formatter for retrieval results"""
    score = record.get("score", None)
    record_metadata = record.get("metadata", {})
    metadata = {}
    
    if score is not None:
        metadata["score"] = score
    if record_metadata:
        metadata.update(record_metadata)
    
    content = record.get("text", str(record))
    
    return RetrieverResultItem(
        content=content,
        metadata=metadata if metadata else None,
    )