"""
Visualization Manager for Chainlit Integration.

This module handles visualization generation, display, and management for the
process mining chatbot, including zoomable image display and visualization
summaries with smooth transitions to the query interface.
"""

import asyncio
import os
import tempfile
import base64
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image, ImageEnhance
import io

import chainlit as cl
import pandas as pd
from visualization.dfg_visualization import visualize_dfg, export_dfg_data
from visualization.performance_dfg_visualization import visualize_performance_dfg
from utils.logging_utils import log

from ..models import VisualizationData, ErrorContext
from ..utils.async_helpers import run_in_executor


@dataclass
class VisualizationSummary:
    """Data model for process structure summaries."""
    total_activities: int
    total_transitions: int
    start_activities: List[str]
    end_activities: List[str]
    most_frequent_path: Optional[str] = None
    average_performance: Optional[str] = None
    complexity_score: Optional[str] = None


class VisualizationManager:
    """
    Manager for handling DFG visualizations with zoomable display functionality.
    
    This class provides zoomable image components using Chainlit, generates
    process structure summaries, and manages smooth transitions to the query
    interface with proper state management.
    """
    
    def __init__(self, session_state):
        self.session_state = session_state
        self.visualization_data = VisualizationData()
        self.temp_files: List[str] = []
        self._max_image_size = (1920, 1080)  # Max dimensions for web display
        self._compression_quality = 85  # JPEG quality for optimization
        
    async def create_zoomable_image(
        self, 
        image_path: str, 
        title: str,
        description: Optional[str] = None
    ) -> cl.Image:
        """
        Create zoomable image components using Chainlit with optimization.
        
        This method creates optimized images for web display with proper sizing
        and compression while maintaining visual quality for process analysis.
        
        Args:
            image_path: Path to the source image file
            title: Title for the image display
            description: Optional description for the image
            
        Returns:
            cl.Image: Chainlit Image object with zoomable functionality
            
        Raises:
            RuntimeError: If image processing fails
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Optimize image for web display
            optimized_path = await self._optimize_image_for_web(image_path, title)
            
            # Create Chainlit Image with zoomable configuration
            image = cl.Image(
                name=title.lower().replace(" ", "_"),
                display="inline",
                path=optimized_path,
                size="large"
            )
            
            log(f"Zoomable image created: {title} from {image_path}", level="debug")
            return image
            
        except Exception as e:
            error_msg = f"Error creating zoomable image '{title}': {str(e)}"
            log(error_msg, level="error")
            raise RuntimeError(error_msg) from e
    
    async def _optimize_image_for_web(self, image_path: str, title: str) -> str:
        """
        Optimize image for web display with resizing and compression.
        
        Args:
            image_path: Path to the original image
            title: Title for naming the optimized image
            
        Returns:
            Path to the optimized image
        """
        try:
            # Create optimized filename
            timestamp = int(asyncio.get_event_loop().time())
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_').lower()
            optimized_path = os.path.join(
                tempfile.gettempdir(), 
                f"{safe_title}_optimized_{timestamp}.png"
            )
            
            # Track temp file for cleanup
            self.temp_files.append(optimized_path)
            
            def optimize_image():
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Create white background for transparency
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = background
                    
                    # Resize if image is too large
                    if img.size[0] > self._max_image_size[0] or img.size[1] > self._max_image_size[1]:
                        img.thumbnail(self._max_image_size, Image.Resampling.LANCZOS)
                        log(f"Image resized to {img.size} for web display", level="debug")
                    
                    # Enhance image quality for better readability
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.1)  # Slight sharpening
                    
                    # Save optimized image
                    img.save(optimized_path, 'PNG', optimize=True)
                    
                return optimized_path
            
            # Run optimization in executor to avoid blocking
            result_path = await run_in_executor(optimize_image)
            
            # Verify optimization
            original_size = os.path.getsize(image_path)
            optimized_size = os.path.getsize(result_path)
            
            log(f"Image optimized: {original_size} -> {optimized_size} bytes ({title})", level="debug")
            return result_path
            
        except Exception as e:
            log(f"Error optimizing image {image_path}: {str(e)}", level="warning")
            # Return original path as fallback
            return image_path
    
    async def generate_automatic_dfgs(
        self,
        dfg_data: Dict,
        performance_data: Dict,
        start_activities: List[str],
        end_activities: List[str]
    ) -> Tuple[cl.Image, cl.Image]:
        """
        Generate both frequency and performance DFG visualizations automatically.
        
        Creates optimized visualizations for web display with automatic image
        conversion and proper sizing for Chainlit interface with zoomable functionality.
        
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
                content="🎨 **Generating process visualizations...**\n\n• Creating frequency DFG..."
            ).send()
            
            # Create temporary files for visualizations
            timestamp = int(asyncio.get_event_loop().time())
            freq_path = os.path.join(tempfile.gettempdir(), f"freq_dfg_{timestamp}.png")
            perf_path = os.path.join(tempfile.gettempdir(), f"perf_dfg_{timestamp}.png")
            export_path = os.path.join(tempfile.gettempdir(), f"dfg_relationships_{timestamp}.csv")
            
            # Track temp files for cleanup
            self.temp_files.extend([freq_path, perf_path, export_path])
            self.visualization_data.temp_files.extend([freq_path, perf_path, export_path])
            
            # Generate frequency DFG with timeout
            await asyncio.wait_for(
                run_in_executor(
                    visualize_dfg, 
                    dfg_data, start_activities, end_activities, freq_path
                ),
                timeout=60  # 1 minute timeout
            )
            
            await viz_progress.update(
                "🎨 **Generating process visualizations...**\n\n• ✅ Frequency DFG created\n• Creating performance DFG..."
            )
            
            # Generate performance DFG with timeout
            await asyncio.wait_for(
                run_in_executor(
                    visualize_performance_dfg,
                    performance_data.get('mean', performance_data), 
                    start_activities, end_activities, perf_path
                ),
                timeout=60  # 1 minute timeout
            )
            
            await viz_progress.update(
                "🎨 **Generating process visualizations...**\n\n• ✅ Frequency DFG created\n• ✅ Performance DFG created\n• Optimizing for display..."
            )
            
            # Export DFG relationships data
            await asyncio.wait_for(
                run_in_executor(
                    export_dfg_data,
                    dfg_data, start_activities, end_activities, export_path
                ),
                timeout=30  # 30 second timeout
            )
            
            # Store paths in visualization data
            self.visualization_data.frequency_dfg_path = freq_path
            self.visualization_data.performance_dfg_path = perf_path
            self.visualization_data.export_csv_path = export_path
            
            # Verify files were created
            if not os.path.exists(freq_path):
                raise RuntimeError("Frequency DFG visualization file not created")
            if not os.path.exists(perf_path):
                raise RuntimeError("Performance DFG visualization file not created")
            
            # Create zoomable images
            freq_image = await self.create_zoomable_image(
                freq_path, 
                "Frequency DFG",
                "Process flow diagram showing activity frequencies and transitions"
            )
            
            perf_image = await self.create_zoomable_image(
                perf_path,
                "Performance DFG", 
                "Process flow diagram showing performance metrics and timing"
            )
            
            await viz_progress.update(
                "✅ **Visualizations ready!**\n\n• Frequency DFG generated with zoom functionality\n• Performance DFG generated with zoom functionality\n• Data exported for download"
            )
            
            log("DFG visualizations with zoom functionality generated successfully", level="info")
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
    
    async def generate_process_summary(
        self,
        dfg_data: Dict,
        performance_data: Dict,
        start_activities: List[str],
        end_activities: List[str]
    ) -> VisualizationSummary:
        """
        Generate brief process structure summaries for display.
        
        Analyzes the process data to create meaningful summaries that help
        users understand the process structure before asking questions.
        
        Args:
            dfg_data: DFG data from process discovery
            performance_data: Performance metrics data
            start_activities: List of start activities
            end_activities: List of end activities
            
        Returns:
            VisualizationSummary: Structured summary of process characteristics
        """
        try:
            # Calculate basic metrics
            total_activities = len(set(
                [activity for (source, target) in dfg_data.keys() for activity in [source, target]]
            ))
            total_transitions = len(dfg_data)
            
            # Find most frequent transition
            most_frequent_transition = None
            if dfg_data:
                most_frequent = max(dfg_data.items(), key=lambda x: x[1])
                most_frequent_transition = f"{most_frequent[0][0]} → {most_frequent[0][1]} ({most_frequent[1]} times)"
            
            # Calculate average performance if available
            average_performance = None
            if performance_data and isinstance(performance_data, dict):
                if 'mean' in performance_data:
                    perf_values = list(performance_data['mean'].values())
                elif isinstance(performance_data, dict) and performance_data:
                    perf_values = list(performance_data.values())
                else:
                    perf_values = []
                
                if perf_values:
                    avg_seconds = sum(perf_values) / len(perf_values)
                    average_performance = self._format_duration(avg_seconds)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(
                total_activities, total_transitions, len(start_activities), len(end_activities)
            )
            
            return VisualizationSummary(
                total_activities=total_activities,
                total_transitions=total_transitions,
                start_activities=start_activities,
                end_activities=end_activities,
                most_frequent_path=most_frequent_transition,
                average_performance=average_performance,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            log(f"Error generating process summary: {str(e)}", level="error")
            # Return basic summary on error
            return VisualizationSummary(
                total_activities=0,
                total_transitions=0,
                start_activities=start_activities or [],
                end_activities=end_activities or []
            )
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        else:
            days = seconds / 86400
            return f"{days:.1f} days"
    
    def _calculate_complexity_score(
        self, 
        activities: int, 
        transitions: int, 
        start_count: int, 
        end_count: int
    ) -> str:
        """Calculate and categorize process complexity."""
        try:
            if activities == 0:
                return "Unknown"
            
            # Simple complexity calculation based on various factors
            transition_ratio = transitions / activities if activities > 0 else 0
            entry_exit_ratio = (start_count + end_count) / activities if activities > 0 else 0
            
            # Complexity scoring
            complexity_points = 0
            
            # Activity count factor
            if activities <= 5:
                complexity_points += 1
            elif activities <= 15:
                complexity_points += 2
            else:
                complexity_points += 3
            
            # Transition density factor
            if transition_ratio <= 1.2:
                complexity_points += 1
            elif transition_ratio <= 2.0:
                complexity_points += 2
            else:
                complexity_points += 3
            
            # Entry/exit points factor
            if entry_exit_ratio <= 0.3:
                complexity_points += 1
            elif entry_exit_ratio <= 0.6:
                complexity_points += 2
            else:
                complexity_points += 3
            
            # Categorize complexity
            if complexity_points <= 4:
                return "Simple"
            elif complexity_points <= 7:
                return "Moderate"
            else:
                return "Complex"
                
        except Exception:
            return "Unknown"
    
    async def display_visualization_summary(self, summary: VisualizationSummary) -> None:
        """
        Display process structure summary with formatting.
        
        Creates a well-formatted summary message that provides users with
        key insights about their process before they start asking questions.
        
        Args:
            summary: VisualizationSummary object with process metrics
        """
        try:
            # Build summary message
            summary_content = "📊 **Process Structure Summary**\n\n"
            
            # Basic metrics
            summary_content += f"• **Activities**: {summary.total_activities}\n"
            summary_content += f"• **Transitions**: {summary.total_transitions}\n"
            summary_content += f"• **Start Points**: {len(summary.start_activities)}\n"
            summary_content += f"• **End Points**: {len(summary.end_activities)}\n"
            
            # Complexity assessment
            if summary.complexity_score:
                summary_content += f"• **Complexity**: {summary.complexity_score}\n"
            
            # Performance information
            if summary.average_performance:
                summary_content += f"• **Average Duration**: {summary.average_performance}\n"
            
            # Most frequent path
            if summary.most_frequent_path:
                summary_content += f"• **Most Frequent Transition**: {summary.most_frequent_path}\n"
            
            # Start and end activities details
            if summary.start_activities:
                start_list = ", ".join(summary.start_activities[:3])  # Show first 3
                if len(summary.start_activities) > 3:
                    start_list += f" (and {len(summary.start_activities) - 3} more)"
                summary_content += f"\n**Starting Activities**: {start_list}\n"
            
            if summary.end_activities:
                end_list = ", ".join(summary.end_activities[:3])  # Show first 3
                if len(summary.end_activities) > 3:
                    end_list += f" (and {len(summary.end_activities) - 3} more)"
                summary_content += f"**Ending Activities**: {end_list}\n"
            
            # Send summary message
            await cl.Message(content=summary_content).send()
            
            log("Process structure summary displayed", level="debug")
            
        except Exception as e:
            log(f"Error displaying visualization summary: {str(e)}", level="error")
            # Send basic message on error
            await cl.Message(
                content="📊 **Process Structure Summary**\n\nProcess analysis complete. You can now ask questions about your process data."
            ).send()
    
    async def transition_to_query_interface(self) -> None:
        """
        Implement smooth transition to query interface with state management.
        
        Provides a clear transition from visualization display to the query
        interface, ensuring users understand they can now ask questions.
        """
        try:
            # Create transition message with clear instructions
            transition_content = (
                "🚀 **Ready for Analysis!**\n\n"
                "Your process visualizations are now displayed above. You can:\n\n"
                "• **Zoom in/out** on the diagrams for detailed analysis\n"
                "• **Ask questions** about your process using natural language\n"
                "• **Switch contexts** to focus on specific aspects\n"
                "• **Export data** for external analysis\n\n"
                "**Next Step**: Choose a query context to start asking questions about your process!"
            )
            
            # Create export actions if data is available
            actions = []
            if self.visualization_data.export_csv_path and os.path.exists(self.visualization_data.export_csv_path):
                actions.append(
                    cl.Action(
                        name="export_data",
                        value="csv",
                        label="📁 Export Process Data (CSV)",
                        description="Download process relationships and activities as CSV file"
                    )
                )
            
            # Send message with export actions if available
            if actions:
                await cl.Message(
                    content=transition_content,
                    actions=actions
                ).send()
            else:
                await cl.Message(content=transition_content).send()
            
            log("Smooth transition to query interface completed", level="debug")
            
        except Exception as e:
            log(f"Error in transition to query interface: {str(e)}", level="error")
            # Send basic transition message on error
            await cl.Message(
                content="✅ **Analysis Ready**\n\nYou can now ask questions about your process data."
            ).send()
    
    async def export_data(self, export_type: str = "csv") -> Optional[cl.File]:
        """
        Create downloadable files for process data export.
        
        Args:
            export_type: Type of export ("csv", "json", "summary")
            
        Returns:
            cl.File object for download, or None if export fails
        """
        try:
            if export_type == "csv" and self.visualization_data.export_csv_path:
                if os.path.exists(self.visualization_data.export_csv_path):
                    export_file = cl.File(
                        name="process_relationships.csv",
                        path=self.visualization_data.export_csv_path,
                        display="inline"
                    )
                    
                    log(f"Export file prepared: {self.visualization_data.export_csv_path}", level="info")
                    return export_file
            
            elif export_type == "json":
                return await self._export_json_data()
            
            elif export_type == "summary":
                return await self._export_process_summary()
            
            elif export_type == "images":
                return await self._export_visualization_images()
            
            log(f"No export file available for type: {export_type}", level="warning")
            return None
            
        except Exception as e:
            log(f"Error preparing export file: {str(e)}", level="error")
            return None
    
    async def _export_json_data(self) -> Optional[cl.File]:
        """Export process data as JSON format."""
        try:
            if not self.visualization_data.export_csv_path or not os.path.exists(self.visualization_data.export_csv_path):
                return None
            
            import pandas as pd
            import json
            
            # Read CSV data
            df = pd.read_csv(self.visualization_data.export_csv_path)
            
            # Convert to structured JSON
            json_data = {
                "process_data": {
                    "relationships": df.to_dict('records'),
                    "export_timestamp": pd.Timestamp.now().isoformat(),
                    "total_relationships": len(df),
                    "relationship_types": df['relationship_type'].unique().tolist() if 'relationship_type' in df.columns else []
                }
            }
            
            # Create temporary JSON file
            timestamp = int(time.time())
            json_path = os.path.join(tempfile.gettempdir(), f"process_data_{timestamp}.json")
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            self.temp_files.append(json_path)
            
            export_file = cl.File(
                name="process_data.json",
                path=json_path,
                display="inline"
            )
            
            log(f"JSON export file prepared: {json_path}", level="info")
            return export_file
            
        except Exception as e:
            log(f"Error creating JSON export: {str(e)}", level="error")
            return None
    
    async def _export_process_summary(self) -> Optional[cl.File]:
        """Export process summary as text file."""
        try:
            # Generate process summary
            summary = await self._generate_process_summary()
            
            if not summary:
                return None
            
            # Create summary text content
            summary_content = f"""Process Mining Analysis Summary
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PROCESS OVERVIEW
================
Process Complexity: {summary.complexity}
Total Activities: {summary.total_activities}
Total Transitions: {summary.total_transitions}

PERFORMANCE METRICS
==================
"""
            
            if summary.average_performance:
                summary_content += f"Average Duration: {summary.average_performance}\n"
            
            if summary.most_frequent_path:
                summary_content += f"Most Frequent Transition: {summary.most_frequent_path}\n"
            
            summary_content += f"""
PROCESS STRUCTURE
=================
Starting Activities: {', '.join(summary.start_activities) if summary.start_activities else 'None'}
Ending Activities: {', '.join(summary.end_activities) if summary.end_activities else 'None'}

DATA EXPORT INFORMATION
=======================
This summary was generated from process mining analysis.
For detailed relationship data, please export the CSV format.
"""
            
            # Create temporary text file
            timestamp = int(time.time())
            summary_path = os.path.join(tempfile.gettempdir(), f"process_summary_{timestamp}.txt")
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            self.temp_files.append(summary_path)
            
            export_file = cl.File(
                name="process_summary.txt",
                path=summary_path,
                display="inline"
            )
            
            log(f"Summary export file prepared: {summary_path}", level="info")
            return export_file
            
        except Exception as e:
            log(f"Error creating summary export: {str(e)}", level="error")
            return None
    
    async def _export_visualization_images(self) -> Optional[cl.File]:
        """Export visualization images as ZIP file."""
        try:
            import zipfile
            
            # Check if visualization images exist
            images_to_zip = []
            if self.visualization_data.frequency_dfg_path and os.path.exists(self.visualization_data.frequency_dfg_path):
                images_to_zip.append(("frequency_dfg.png", self.visualization_data.frequency_dfg_path))
            
            if self.visualization_data.performance_dfg_path and os.path.exists(self.visualization_data.performance_dfg_path):
                images_to_zip.append(("performance_dfg.png", self.visualization_data.performance_dfg_path))
            
            if not images_to_zip:
                return None
            
            # Create ZIP file
            timestamp = int(time.time())
            zip_path = os.path.join(tempfile.gettempdir(), f"process_visualizations_{timestamp}.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for filename, filepath in images_to_zip:
                    zipf.write(filepath, filename)
            
            self.temp_files.append(zip_path)
            
            export_file = cl.File(
                name="process_visualizations.zip",
                path=zip_path,
                display="inline"
            )
            
            log(f"Images export file prepared: {zip_path}", level="info")
            return export_file
            
        except Exception as e:
            log(f"Error creating images export: {str(e)}", level="error")
            return None
    
    async def show_export_options(self) -> None:
        """Display export options to the user."""
        try:
            export_actions = []
            
            # CSV export (always available if data exists)
            if self.visualization_data.export_csv_path and os.path.exists(self.visualization_data.export_csv_path):
                export_actions.append(
                    cl.Action(
                        name="export_data",
                        value="csv",
                        label="📊 Export CSV Data",
                        description="Download process relationships as CSV file"
                    )
                )
                
                export_actions.append(
                    cl.Action(
                        name="export_data",
                        value="json",
                        label="📋 Export JSON Data",
                        description="Download process data as structured JSON"
                    )
                )
                
                export_actions.append(
                    cl.Action(
                        name="export_data",
                        value="summary",
                        label="📄 Export Summary Report",
                        description="Download process analysis summary as text file"
                    )
                )
            
            # Images export (if visualizations exist)
            if ((self.visualization_data.frequency_dfg_path and os.path.exists(self.visualization_data.frequency_dfg_path)) or
                (self.visualization_data.performance_dfg_path and os.path.exists(self.visualization_data.performance_dfg_path))):
                export_actions.append(
                    cl.Action(
                        name="export_data",
                        value="images",
                        label="🖼️ Export Visualizations",
                        description="Download process diagrams as ZIP file"
                    )
                )
            
            if export_actions:
                await cl.Message(
                    content="📁 **Export Options Available**\n\nChoose the format you'd like to export:",
                    actions=export_actions
                ).send()
            else:
                await cl.Message(
                    content="❌ **No Export Data Available**\n\nPlease complete process analysis first to enable exports."
                ).send()
                
        except Exception as e:
            log(f"Error showing export options: {str(e)}", level="error")
            await cl.Message(
                content="❌ **Export Error**\n\nUnable to display export options. Please try again."
            ).send()
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary visualization files."""
        try:
            # Clean up manager temp files
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    log(f"Error cleaning up temp file {temp_file}: {str(e)}", level="warning")
            
            self.temp_files.clear()
            
            # Clean up visualization data temp files
            self.visualization_data.cleanup_temp_files()
            
            log("Visualization temporary files cleaned up", level="debug")
            
        except Exception as e:
            log(f"Error during temp file cleanup: {str(e)}", level="warning")
    
    def get_visualization_state(self) -> Dict[str, Any]:
        """
        Get current visualization display state for session management.
        
        Returns:
            Dictionary containing visualization state information
        """
        return {
            "has_frequency_dfg": self.visualization_data.frequency_dfg_path is not None,
            "has_performance_dfg": self.visualization_data.performance_dfg_path is not None,
            "has_export_data": self.visualization_data.export_csv_path is not None,
            "temp_files_count": len(self.temp_files)
        }