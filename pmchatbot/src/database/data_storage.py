import time
import traceback

def store_chunks_in_neo4j(driver, dfg, start_activities, end_activities, activity_chunks, process_chunks, variant_chunks, frequent_paths, variant_stats, local_embedder):
    """Store all chunks and process data in Neo4j"""
    print("Storing activity-based, process-based, and variant-based chunks and RAG data in Neo4j...")
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        print("   - Cleared existing data")
        
        # Drop existing indexes
        try:
            session.run("DROP INDEX activity_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX activity_chunk_fulltext_index IF EXISTS")
            session.run("DROP INDEX process_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX process_chunk_fulltext_index IF EXISTS")
            session.run("DROP INDEX case_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX case_chunk_fulltext_index IF EXISTS")
            session.run("DROP INDEX variant_chunk_vector_index IF EXISTS")
            session.run("DROP INDEX variant_chunk_fulltext_index IF EXISTS")
            print("   - Dropped existing indexes")
        except Exception as e:
            print(f"   - Note: {e}")
        
        # Store DFG and activities
        _store_dfg_data(session, dfg, start_activities, end_activities)
        
        # Store activity chunks
        _store_activity_chunks(session, activity_chunks, local_embedder)
        
        # Store process paths and chunks
        _store_process_chunks(session, process_chunks, frequent_paths, local_embedder)
        
        # Store case variants and chunks
        _store_variant_chunks(session, variant_chunks, variant_stats, local_embedder)
        
        # Create indexes
        _create_indexes(session)

def _store_dfg_data(session, dfg, start_activities, end_activities):
    """Store DFG and activity data"""
    dfg_data = [{"src": s, "tgt": t, "count": c} for (s, t), c in dfg.items()]
    session.run("""
        UNWIND $rows AS r
        MERGE (a:Activity {name: r.src})
        MERGE (b:Activity {name: r.tgt})
        MERGE (a)-[f:NEXT]->(b)
        ON CREATE SET f.count = r.count
        ON MATCH SET f.count = r.count
    """, rows=dfg_data)
    print(f"   - Created {len(dfg_data)} process paths")
    
    if start_activities:
        session.run("""
            UNWIND $starts AS start
            MATCH (a:Activity {name: start.activity})
            SET a.is_start = true, a.start_count = start.count
        """, starts=[{"activity": act, "count": count} for act, count in start_activities.items()])
        print(f"   - Marked {len(start_activities)} start activities")
    
    if end_activities:
        session.run("""
            UNWIND $ends AS end
            MATCH (a:Activity {name: end.activity})
            SET a.is_end = true, a.end_count = end.count
        """, ends=[{"activity": act, "count": count} for act, count in end_activities.items()])
        print(f"   - Marked {len(end_activities)} end activities")

def _store_activity_chunks(session, activity_chunks, local_embedder):
    """Store activity chunks with embeddings"""
    print("   - Creating activity chunk embeddings with local model...")
    for i, chunk in enumerate(activity_chunks):
        try:
            embedding = local_embedder.encode([chunk["text"]])[0].tolist()
            session.run("""
                CREATE (ac:ActivityChunk {
                    id: $id,
                    text: $text,
                    activity_name: $activity_name,
                    type: $type,
                    source: $source,
                    embedding: $embedding
                })
            """, 
            id=i, 
            text=chunk["text"], 
            activity_name=chunk["activity_name"],
            type=chunk["type"],
            source=chunk["source"],
            embedding=embedding)
            
            session.run("""
                MATCH (ac:ActivityChunk {id: $chunk_id})
                MATCH (a:Activity {name: $activity_name})
                MERGE (ac)-[:DESCRIBES]->(a)
            """, chunk_id=i, activity_name=chunk["activity_name"])
            
            print(f"   - Created activity chunk {i+1}/{len(activity_chunks)} for '{chunk['activity_name']}'")
        except Exception as e:
            print(f"   - Error creating activity chunk {i}: {e}")

def _store_process_chunks(session, process_chunks, frequent_paths, local_embedder):
    """Store process paths and chunks with performance metrics as properties"""
    # Create ProcessPath nodes (unchanged)
    for i, (path, frequency) in enumerate(frequent_paths):
        path_str = " → ".join(path)
        session.run("""
            CREATE (pp:ProcessPath {
                id: $id,
                path_string: $path_string,
                frequency: $frequency,
                length: $length,
                rank: $rank
            })
        """, 
        id=i,
        path_string=path_str,
        frequency=frequency,
        length=len(path),
        rank=i + 1)
        
        for j, activity in enumerate(path):
            session.run("""
                MATCH (pp:ProcessPath {id: $path_id})
                MATCH (a:Activity {name: $activity})
                MERGE (pp)-[:CONTAINS {position: $position}]->(a)
            """, path_id=i, activity=activity, position=j)
    
    print(f"   - Created {len(frequent_paths)} ProcessPath nodes")
    
    # Store process chunks with embeddings and performance metrics
    print("   - Creating process chunk embeddings with local model...")
    for i, chunk in enumerate(process_chunks):
        try:
            embedding = local_embedder.encode([chunk["text"]])[0].tolist()
            perf = chunk["data"]["performance"]
            session.run("""
                CREATE (pc:ProcessChunk {
                    id: $id,
                    text: $text,
                    path_string: $path_string,
                    type: $type,
                    source: $source,
                    embedding: $embedding,
                    mean_duration: $mean,
                    min_duration: $min,
                    max_duration: $max,
                    frequency: $frequency,
                    rank: $rank
                })
            """, 
            id=i, 
            text=chunk["text"], 
            path_string=chunk["path_string"],
            type=chunk["type"],
            source=chunk["source"],
            embedding=embedding,
            mean=perf["mean"],
            min=perf["min"],
            max=perf["max"],
            frequency=chunk["data"]["frequency"],
            rank=chunk["data"]["rank"])
            
            session.run("""
                MATCH (pc:ProcessChunk {id: $chunk_id})
                MATCH (pp:ProcessPath {id: $path_id})
                MERGE (pc)-[:DESCRIBES]->(pp)
            """, chunk_id=i, path_id=i)
            
            print(f"   - Created process chunk {i+1}/{len(process_chunks)} for '{chunk['path_string']}'")
        except Exception as e:
            print(f"   - Error creating process chunk {i}: {e}")

def _store_variant_chunks(session, variant_chunks, variant_stats, local_embedder):
    """Store case variants and chunks with performance metrics as properties"""
    # Create CaseVariant nodes (unchanged)
    for i, stats in enumerate(variant_stats):
        variant = stats['variant']
        variant_str = " → ".join(variant)
        
        variant_props = {
            'id': i,
            'variant_string': variant_str,
            'frequency': stats['frequency'],
            'length': len(variant),
            'rank': i + 1,
            'avg_activities': stats['avg_activities'],
            'avg_unique_activities': stats['avg_unique_activities']
        }
        
        session.run("""
            CREATE (cv:CaseVariant $props)
        """, props=variant_props)
        
        for j, activity in enumerate(variant):
            session.run("""
                MATCH (cv:CaseVariant {id: $variant_id})
                MATCH (a:Activity {name: $activity})
                MERGE (cv)-[:EXECUTES {position: $position}]->(a)
            """, variant_id=i, activity=activity, position=j)
        
        for case_id in stats['cases'][:10]:
            session.run("""
                MATCH (cv:CaseVariant {id: $variant_id})
                CREATE (c:Case {id: $case_id})
                MERGE (c)-[:FOLLOWS]->(cv)
            """, variant_id=i, case_id=case_id)
    
    print(f"   - Created {len(variant_stats)} CaseVariant nodes")
    
    # Store variant chunks with embeddings and performance metrics
    print("   - Creating variant-based chunk embeddings with local model...")
    for i, chunk in enumerate(variant_chunks):
        try:
            embedding = local_embedder.encode([chunk["text"]])[0].tolist()
            perf = chunk["data"]["performance"]
            session.run("""
                CREATE (vc:VariantChunk {
                    id: $id,
                    text: $text,
                    variant_string: $variant_string,
                    type: $type,
                    source: $source,
                    embedding: $embedding,
                    avg_duration: $avg_duration,
                    min_duration: $min_duration,
                    max_duration: $max_duration,
                    total_duration: $total_duration,
                    frequency: $frequency,
                    rank: $rank
                })
            """, 
            id=i, 
            text=chunk["text"], 
            variant_string=chunk["variant_string"],
            type=chunk["type"],
            source=chunk["source"],
            embedding=embedding,
            avg_duration=perf["avg_duration"],
            min_duration=perf["min_duration"],
            max_duration=perf["max_duration"],
            total_duration=perf["total_duration"],
            frequency=chunk["data"]["frequency"],
            rank=chunk["data"]["rank"])
            
            session.run("""
                MATCH (vc:VariantChunk {id: $chunk_id})
                MATCH (cv:CaseVariant {id: $variant_id})
                MERGE (vc)-[:DESCRIBES]->(cv)
            """, chunk_id=i, variant_id=i)
            
            print(f"   - Created variant chunk {i+1}/{len(variant_chunks)} for '{chunk['variant_string']}'")
        except Exception as e:
            print(f"   - Error creating variant chunk {i}: {e}")

def _create_indexes(session):
    """Create fulltext and vector indexes"""
    time.sleep(2)
    
    # Create fulltext indexes
    try:
        session.run("""
            CREATE FULLTEXT INDEX activity_chunk_fulltext_index IF NOT EXISTS
            FOR (ac:ActivityChunk) ON EACH [ac.text, ac.activity_name]
        """)
        session.run("""
            CREATE FULLTEXT INDEX process_chunk_fulltext_index IF NOT EXISTS
            FOR (pc:ProcessChunk) ON EACH [pc.text, pc.path_string]
        """)
        session.run("""
            CREATE FULLTEXT INDEX variant_chunk_fulltext_index IF NOT EXISTS
            FOR (vc:VariantChunk) ON EACH [vc.text, vc.variant_string]
        """)
        print("   - Created fulltext indexes")
    except Exception as e:
        print(f"   - Warning: Fulltext index creation: {e}")
    
    # Create vector indexes
    try:
        # Create vector indexes with 1024 dimensions for E5-large
        result = session.run("MATCH (ac:ActivityChunk) RETURN count(ac) as count")
        node_count = result.single()["count"]
        if node_count > 0:
            session.run("""
                CREATE VECTOR INDEX activity_chunk_vector_index IF NOT EXISTS
                FOR (ac:ActivityChunk) ON (ac.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
        
        result = session.run("MATCH (pc:ProcessChunk) RETURN count(pc) as count")
        node_count = result.single()["count"]
        if node_count > 0:
            session.run("""
                CREATE VECTOR INDEX process_chunk_vector_index IF NOT EXISTS
                FOR (pc:ProcessChunk) ON (pc.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
        
        result = session.run("MATCH (vc:VariantChunk) RETURN count(vc) as count")
        node_count = result.single()["count"]
        if node_count > 0:
            session.run("""
                CREATE VECTOR INDEX variant_chunk_vector_index IF NOT EXISTS
                FOR (vc:VariantChunk) ON (vc.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
        print("   - Created vector indexes")
    except Exception as e:
        print(f"   - Critical error in vector index creation: {e}")
        traceback.print_exc()