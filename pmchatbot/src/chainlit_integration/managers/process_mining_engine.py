"""
Process Mining Engine Wrapper for Chainlit Integration.

This module provides an async wrapper around the existing PM4py functionality
from main.py, enabling integration with Chainlit's async event system.
"""

import asyncio
import os
import tempfile
import torch
import gc
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import chainlit as cl
from data.data_loader import load_csv_data, build_activity_case_map, filter_by_part_desc
from data.data_processor import prepare_pm4py_log, discover_process_model, extract_case_variants, extract_process_paths
from chunking.activity_chunker import generate_activity_based_chunks
from chunking.process_chunker import generate_process_based_chunks
from chunking.variant_chunker import generate_variant_based_chunks
from embeddings.local_embedder import get_local_embedder
from database.neo4j_manager import connect_neo4j, force_clean_neo4j_indexes
from database.data_storage import store_chunks_in_neo4j
from retrieval.enhanced_retriever import setup_enhanced_retriever
from visualization.dfg_visualization import visualize_dfg, export_dfg_data
from visualization.performance_dfg_visualization import visualize_performance_dfg
from utils.logging_utils import log
from config.settings import Config

from ..models import ProcessingResult, ErrorContext
from ..utils.async_helpers import run_in_executor


class ProcessMiningEngine:
    """
    Async wrapper for PM4py pipeline functionality.
    
    This class wraps the existing synchronous PM4py operations from main.py
    into async methods suitable for Chainlit integration, while maintaining
    the same functionality and adding progress indicators.
    """
    
    def __init__(self, session_state):
        self.session_state = session_state
        self.config = Config()
        self.driver = None
        self.temp_files: List[str] = []
        self._connection_retries = 3
        self._connection_timeout = 30
        
    async def initialize_neo4j(self) -> bool:
        """
        Initialize Neo4j connection asynchronously with retry logic.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        for attempt in range(self._connection_retries):
            try:
                # Run Neo4j connection in executor to avoid blocking
                self.driver = await asyncio.wait_for(
                    run_in_executor(connect_neo4j),
                    timeout=self._connection_timeout
                )
                
                if not self.driver:
                    log(f"Neo4j connection attempt {attempt + 1} failed", level="warning")
                    if attempt < self._connection_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        log("Failed to connect to Neo4j after all retries", level="error")
                        return False
                
                # Test the connection
                await self._test_neo4j_connection()
                log("Neo4j connection established and tested", level="info")
                return True
                
            except asyncio.TimeoutError:
                log(f"Neo4j connection timeout on attempt {attempt + 1}", level="warning")
                if attempt < self._connection_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    log("Neo4j connection timed out after all retries", level="error")
                    return False
                    
            except Exception as e:
                log(f"Error connecting to Neo4j (attempt {attempt + 1}): {str(e)}", level="error")
                if attempt < self._connection_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return False
        
        return False
    
    async def _test_neo4j_connection(self) -> None:
        """
        Test Neo4j connection by running a simple query.
        
        Raises:
            Exception: If connection test fails
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        def test_query():
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        
        success = await run_in_executor(test_query)
        if not success:
            raise RuntimeError("Neo4j connection test failed")
    
    async def process_data(self, selected_part: str) -> ProcessingResult:
        """
        Process data for the selected part using PM4py pipeline.
        
        This method wraps the main PM4py processing logic from main.py
        into an async operation with progress indicators.
        
        Args:
            selected_part: The part description to filter data by
            
        Returns:
            ProcessingResult: Results of the processing operation
        """
        try:
            # Show initial progress
            progress_msg = await cl.Message(
                content="🔄 **Starting data processing...**\n\n• Loading CSV data..."
            ).send()
            
            # Load and filter data
            df = await run_in_executor(load_csv_data)
            if selected_part:
                df = await run_in_executor(filter_by_part_desc, df, selected_part)
            
            await progress_msg.update(
                "🔄 **Processing data...**\n\n• ✅ Data loaded\n• Preparing PM4py event log..."
            )
            
            # Prepare PM4py log
            event_log = await run_in_executor(prepare_pm4py_log, df)
            
            await progress_msg.update(
                "🔄 **Processing data...**\n\n• ✅ Data loaded\n• ✅ Event log prepared\n• Discovering process model..."
            )
            
            # Discover process model
            dfg, start_activities, end_activities, performance_dfgs, rework_cases = await run_in_executor(
                discover_process_model, event_log
            )
            
            await progress_msg.update(
                "🔄 **Processing data...**\n\n• ✅ Data loaded\n• ✅ Event log prepared\n• ✅ Process model discovered\n• Generating chunks..."
            )
            
            # Generate chunks
            activity_case_map = await run_in_executor(build_activity_case_map, df)
            
            activity_chunks = await run_in_executor(
                generate_activity_based_chunks,
                dfg, start_activities, end_activities, activity_case_map, rework_cases
            )
            
            # Extract process paths for process chunks
            frequent_paths, path_performance = await run_in_executor(
                extract_process_paths, dfg, performance_dfgs, 1  # min_frequency=1
            )
            
            process_chunks = await run_in_executor(
                generate_process_based_chunks, frequent_paths, path_performance
            )
            
            # Extract case variants for variant chunks
            variant_stats = await run_in_executor(
                extract_case_variants, event_log, 1  # min_cases_per_variant=1
            )
            
            variant_chunks = await run_in_executor(
                generate_variant_based_chunks, dfg, start_activities, end_activities, variant_stats
            )
            
            await progress_msg.update(
                "🔄 **Processing data...**\n\n• ✅ Data loaded\n• ✅ Event log prepared\n• ✅ Process model discovered\n• ✅ Chunks generated\n• Storing in Neo4j..."
            )
            
            # Store chunks in Neo4j
            await self._store_chunks_in_neo4j(
                dfg, start_activities, end_activities,
                activity_chunks, process_chunks, variant_chunks,
                frequent_paths, variant_stats
            )
            
            await progress_msg.update(
                "✅ **Data processing complete!**\n\n• Data loaded and filtered\n• Process model discovered\n• Chunks generated and stored\n• Ready for visualization and queries"
            )
            
            # Return processing results
            chunk_counts = {
                "activity": len(activity_chunks),
                "process": len(process_chunks),
                "variant": len(variant_chunks)
            }
            
            return ProcessingResult(
                success=True,
                dfg_data=dfg,
                performance_data=performance_dfgs,
                start_activities=start_activities,
                end_activities=end_activities,
                chunk_counts=chunk_counts
            )
            
        except Exception as e:
            error_msg = f"Error during data processing: {str(e)}"
            log(error_msg, level="error")
            
            # Update progress message with error
            if 'progress_msg' in locals():
                await progress_msg.update(
                    f"❌ **Processing failed**\n\nError: {str(e)}"
                )
            
            return ProcessingResult(
                success=False,
                error_message=error_msg
            )
    
    async def _store_chunks_in_neo4j(
        self, 
        dfg: Dict, 
        start_activities: List[str], 
        end_activities: List[str],
        activity_chunks: List[Dict],
        process_chunks: List[Dict], 
        variant_chunks: List[Dict],
        frequent_paths: List[Dict],
        variant_stats: Dict
    ) -> None:
        """
        Store processed chunks in Neo4j database with error handling.
        
        This method handles the Neo4j storage operations asynchronously,
        including index cleanup and embedding generation with proper
        connection management and error recovery.
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        local_embedder_gpu = None
        
        try:
            # Test connection before proceeding
            await self._test_neo4j_connection()
            
            # Clean existing indexes with timeout
            await asyncio.wait_for(
                run_in_executor(force_clean_neo4j_indexes, self.driver),
                timeout=60  # 1 minute timeout for index cleanup
            )
            log("Neo4j indexes cleaned successfully", level="debug")
            
            # Get embedder for GPU processing
            local_embedder_gpu = await run_in_executor(get_local_embedder, "cuda")
            log("GPU embedder initialized", level="debug")
            
            # Store chunks in Neo4j with timeout
            await asyncio.wait_for(
                run_in_executor(
                    store_chunks_in_neo4j,
                    self.driver, dfg, start_activities, end_activities,
                    activity_chunks, process_chunks, variant_chunks,
                    frequent_paths, variant_stats, local_embedder_gpu
                ),
                timeout=300  # 5 minute timeout for chunk storage
            )
            log("Chunks stored in Neo4j successfully", level="info")
            
        except asyncio.TimeoutError as e:
            error_msg = "Neo4j storage operation timed out"
            log(error_msg, level="error")
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Error storing chunks in Neo4j: {str(e)}"
            log(error_msg, level="error")
            raise RuntimeError(error_msg) from e
            
        finally:
            # Clean up GPU resources
            if local_embedder_gpu is not None:
                del local_embedder_gpu
                log("GPU embedder cleaned up", level="debug")
            
            gc.collect()
            torch.cuda.empty_cache()
            log("GPU memory cleared", level="debug")
    
    async def generate_automatic_visualizations(
        self, 
        dfg_data: Dict, 
        performance_data: Dict,
        start_activities: List[str],
        end_activities: List[str]
    ) -> Tuple[cl.Image, cl.Image]:
        """
        Generate both frequency and performance DFG visualizations automatically.
        
        This method creates optimized visualizations for web display with
        automatic image conversion and proper sizing for Chainlit interface.
        
        Args:
            dfg_data: DFG data from process discovery
            performance_data: Performance metrics data
            start_activities: List of start activities
            end_activities: List of end activities
            
        Returns:
            Tuple of Chainlit Image objects (frequency_dfg, performance_dfg)
        """
        try:
            # Show progress for visualization generation
            viz_progress = await cl.Message(
                content="🎨 **Generating visualizations...**\n\n• Creating frequency DFG..."
            ).send()
            
            # Create temporary files for visualizations with better naming
            timestamp = int(asyncio.get_event_loop().time())
            freq_path = os.path.join(tempfile.gettempdir(), f"freq_dfg_{timestamp}.png")
            perf_path = os.path.join(tempfile.gettempdir(), f"perf_dfg_{timestamp}.png")
            export_path = os.path.join(tempfile.gettempdir(), f"dfg_relationships_{timestamp}.csv")
            
            # Track temp files for cleanup
            self.temp_files.extend([freq_path, perf_path, export_path])
            
            # Generate frequency DFG with timeout
            await asyncio.wait_for(
                run_in_executor(
                    visualize_dfg, 
                    dfg_data, start_activities, end_activities, freq_path
                ),
                timeout=60  # 1 minute timeout
            )
            
            await viz_progress.update(
                "🎨 **Generating visualizations...**\n\n• ✅ Frequency DFG created\n• Creating performance DFG..."
            )
            
            # Generate performance DFG with timeout
            await asyncio.wait_for(
                run_in_executor(
                    visualize_performance_dfg,
                    performance_data['mean'], start_activities, end_activities, perf_path
                ),
                timeout=60  # 1 minute timeout
            )
            
            await viz_progress.update(
                "🎨 **Generating visualizations...**\n\n• ✅ Frequency DFG created\n• ✅ Performance DFG created\n• Exporting data..."
            )
            
            # Export DFG relationships data
            await asyncio.wait_for(
                run_in_executor(
                    export_dfg_data,
                    dfg_data, start_activities, end_activities, export_path
                ),
                timeout=30  # 30 second timeout
            )
            
            # Verify files were created
            if not os.path.exists(freq_path):
                raise RuntimeError("Frequency DFG visualization file not created")
            if not os.path.exists(perf_path):
                raise RuntimeError("Performance DFG visualization file not created")
            
            # Optimize images for web display
            freq_path_optimized = await self._optimize_image_for_web(freq_path)
            perf_path_optimized = await self._optimize_image_for_web(perf_path)
            
            # Create Chainlit Image objects with better configuration
            freq_image = cl.Image(
                name="frequency_dfg",
                display="inline",
                path=freq_path_optimized,
                size="large"
            )
            
            perf_image = cl.Image(
                name="performance_dfg", 
                display="inline",
                path=perf_path_optimized,
                size="large"
            )
            
            await viz_progress.update(
                "✅ **Visualizations ready!**\n\n• Frequency DFG generated\n• Performance DFG generated\n• Data exported for download"
            )
            
            log("DFG visualizations generated successfully", level="info")
            return freq_image, perf_image
            
        except asyncio.TimeoutError:
            error_msg = "Visualization generation timed out"
            log(error_msg, level="error")
            if 'viz_progress' in locals():
                await viz_progress.update(f"❌ **Visualization failed**: {error_msg}")
            raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"Error generating visualizations: {str(e)}"
            log(error_msg, level="error")
            if 'viz_progress' in locals():
                await viz_progress.update(f"❌ **Visualization failed**: {error_msg}")
            raise RuntimeError(error_msg)
    
    async def _optimize_image_for_web(self, image_path: str) -> str:
        """
        Optimize image for web display by resizing and compressing if needed.
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Path to the optimized image
        """
        try:
            # For now, return the original path
            # In a production environment, you might want to:
            # - Resize large images to reasonable dimensions
            # - Compress images to reduce file size
            # - Convert to web-optimized formats
            
            # Check if image file exists and has reasonable size
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:  # 10MB limit
                    log(f"Warning: Large image file ({file_size} bytes) at {image_path}", level="warning")
                
                log(f"Image optimized for web display: {image_path} ({file_size} bytes)", level="debug")
                return image_path
            else:
                raise RuntimeError(f"Image file not found: {image_path}")
                
        except Exception as e:
            log(f"Error optimizing image {image_path}: {str(e)}", level="warning")
            return image_path  # Return original path as fallback
    
    async def export_dfg_data_for_download(self) -> Optional[cl.File]:
        """
        Create downloadable CSV file with DFG relationships data.
        
        Returns:
            Chainlit File object for download, or None if no export available
        """
        try:
            # Find the most recent export file
            export_files = [f for f in self.temp_files if f.endswith('.csv') and 'dfg_relationships' in f]
            
            if not export_files:
                log("No DFG export file available", level="warning")
                return None
            
            # Use the most recent export file
            export_path = max(export_files, key=lambda f: os.path.getctime(f) if os.path.exists(f) else 0)
            
            if not os.path.exists(export_path):
                log(f"Export file not found: {export_path}", level="warning")
                return None
            
            # Create Chainlit File object
            export_file = cl.File(
                name="dfg_relationships.csv",
                path=export_path,
                display="inline"
            )
            
            log(f"DFG export file prepared: {export_path}", level="info")
            return export_file
            
        except Exception as e:
            log(f"Error preparing export file: {str(e)}", level="error")
            return None
    
    async def setup_retrievers(self) -> Tuple[Any, Any, Any]:
        """
        Set up GraphRAG retrievers after data processing with comprehensive error handling.
        
        Returns:
            Tuple of (activity_retriever, process_retriever, variant_retriever)
            
        Raises:
            RuntimeError: If retriever setup fails
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        local_embedder_cpu = None
        
        try:
            # Test connection before proceeding
            await self._test_neo4j_connection()
            
            # Get CPU embedder for retrieval
            local_embedder_cpu = await asyncio.wait_for(
                run_in_executor(get_local_embedder, "cpu"),
                timeout=60  # 1 minute timeout for embedder initialization
            )
            log("CPU embedder initialized for retrievers", level="debug")
            
            # Setup retrievers with individual error handling
            retrievers = []
            chunk_types = ["ActivityChunk", "ProcessChunk", "VariantChunk"]
            
            for chunk_type in chunk_types:
                try:
                    retriever = await asyncio.wait_for(
                        run_in_executor(
                            setup_enhanced_retriever,
                            self.driver, chunk_type, local_embedder_cpu, 
                            self.config.USE_RERANKER if hasattr(self.config, 'USE_RERANKER') else None
                        ),
                        timeout=120  # 2 minute timeout per retriever
                    )
                    
                    if not retriever:
                        raise RuntimeError(f"Failed to setup {chunk_type} retriever")
                    
                    retrievers.append(retriever)
                    log(f"{chunk_type} retriever setup successfully", level="debug")
                    
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Timeout setting up {chunk_type} retriever")
                except Exception as e:
                    raise RuntimeError(f"Error setting up {chunk_type} retriever: {str(e)}")
            
            # Verify all retrievers were created
            if len(retrievers) != 3:
                raise RuntimeError(f"Expected 3 retrievers, got {len(retrievers)}")
            
            log("All GraphRAG retrievers setup successfully", level="info")
            return tuple(retrievers)
            
        except Exception as e:
            error_msg = f"Error setting up retrievers: {str(e)}"
            log(error_msg, level="error")
            raise RuntimeError(error_msg) from e
            
        finally:
            # Clean up CPU embedder
            if local_embedder_cpu is not None:
                del local_embedder_cpu
                log("CPU embedder cleaned up", level="debug")
            
            torch.cuda.empty_cache()
            log("GPU memory cleared after retriever setup", level="debug")
    
    def cleanup_resources(self) -> None:
        """
        Clean up resources including Neo4j connection and temporary files.
        """
        # Close Neo4j connection
        if self.driver:
            self.driver.close()
            self.driver = None
            log("Neo4j connection closed", level="info")
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                log(f"Error cleaning up temp file {temp_file}: {str(e)}", level="warning")
        
        self.temp_files.clear()
        
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()