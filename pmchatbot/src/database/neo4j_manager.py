import neo4j
import time
from config.settings import Config
from utils.logging_utils import log
config = Config()

def connect_neo4j():
    """Connect to Neo4j database"""
    try:
        driver = neo4j.GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            session.run("RETURN 1")
        log("Connected to Neo4j successfully!", level="info")
        return driver
    except Exception as e:
        log(f"Failed to connect to Neo4j: {e}", level="error")
        return None

def force_clean_neo4j_indexes(driver):
    """Force clean all indexes in Neo4j"""
    log("Force cleaning all Neo4j indexes...", level="info")
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
                        log(f"   - Force dropped index: {index_name}", level="debug")
                    except Exception as e:
                        log(f"   - Could not drop {index_name}: {e}", level="warning")
            
            # Wait for cleanup
            time.sleep(5)
            log("   - All indexes cleaned successfully", level="info")
        except Exception as e:
            log(f"   - Error cleaning indexes: {e}", level="error")