import neo4j
import time
from config.settings import Config

def connect_neo4j():
    """Connect to Neo4j database"""
    try:
        driver = neo4j.GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            session.run("RETURN 1")
        print("Connected to Neo4j successfully!")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def force_clean_neo4j_indexes(driver):
    """Force clean all indexes in Neo4j"""
    print("Force cleaning all Neo4j indexes...")
    with driver.session() as session:
        try:
            # Get all existing indexes
            result = session.run("SHOW INDEXES")
            indexes = list(result)
            
            for index in indexes:
                index_name = index.get('name')
                if index_name and ('chunk' in index_name.lower() or 'vector' in index_name.lower() or 'fulltext' in index_name.lower()):
                    try:
                        session.run(f"DROP INDEX `{index_name}` IF EXISTS")
                        print(f"   - Force dropped index: {index_name}")
                    except Exception as e:
                        print(f"   - Could not drop {index_name}: {e}")
            
            # Wait for cleanup
            time.sleep(5)
            print("   - All indexes cleaned successfully")
        except Exception as e:
            print(f"   - Error cleaning indexes: {e}")