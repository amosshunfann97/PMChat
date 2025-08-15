"""
Part Selection Manager for Chainlit integration.

This module handles part selection functionality including CSV data loading,
part extraction, searchable dropdown interface, and part selection validation.
"""

import os
import pandas as pd
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

# Conditional import for Chainlit to avoid import errors in testing
try:
    import chainlit as cl
except ImportError:
    # Create a mock cl module for testing purposes
    class MockChainlit:
        class Action:
            def __init__(self, name: str, value: str, label: str):
                self.name = name
                self.value = value
                self.label = label
        
        class Message:
            def __init__(self, content: str, actions=None):
                self.content = content
                self.actions = actions or []
            
            async def send(self):
                return None
        
        class AskUserMessage:
            def __init__(self, content: str, timeout: int = 60):
                self.content = content
                self.timeout = timeout
            
            async def send(self):
                return None
        
        @staticmethod
        def action_callback(name: str):
            def decorator(func):
                return func
            return decorator
    cl = MockChainlit()

from ..interfaces import PartSelectionManagerInterface
from ..models import SessionState, ErrorContext


class PartSelectionManager(PartSelectionManagerInterface):
    """
    Manager for handling part selection from CSV data.
    
    This class provides functionality to:
    - Load and cache CSV data
    - Extract unique parts from the data
    - Provide searchable dropdown interface
    - Handle part selection and validation
    """
    
    def __init__(self, session_state: SessionState, csv_file_path: Optional[str] = None):
        """
        Initialize the Part Selection Manager.
        
        Args:
            session_state: Current session state
            csv_file_path: Path to CSV file (optional, will use default if not provided)
        """
        super().__init__(session_state)
        self.csv_file_path = csv_file_path or self._get_default_csv_path()
        self._cached_parts: Optional[List[str]] = None
        self._cached_data: Optional[pd.DataFrame] = None
        self.logger = logging.getLogger(__name__)
    
    def _get_default_csv_path(self) -> str:
        """
        Get the default CSV file path.
        
        Returns:
            Default path to the CSV file
        """
        # Look for CSV file in the project structure
        possible_paths = [
            "pmchatbot/Production_Event_Log.csv",
            "Production_Event_Log.csv",
            "../Production_Event_Log.csv",
            "../../Production_Event_Log.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Default fallback
        return "pmchatbot/Production_Event_Log.csv"
    
    async def initialize(self) -> bool:
        """
        Initialize the manager by loading available parts.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            parts = await self.load_available_parts()
            self.session_state.available_parts = parts
            self.session_state.data_loaded = True
            self.logger.info(f"Initialized Part Selection Manager with {len(parts)} parts")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Part Selection Manager: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up manager resources."""
        self._cached_parts = None
        self._cached_data = None
        self.logger.info("Part Selection Manager cleaned up")
    
    async def load_available_parts(self) -> List[str]:
        """
        Load available parts from CSV data with caching.
        
        Returns:
            List of unique part names from the CSV data
            
        Raises:
            FileNotFoundError: If CSV file is not found
            ValueError: If CSV file is invalid or missing required columns
        """
        # Return cached parts if available
        if self._cached_parts is not None:
            return self._cached_parts
        
        try:
            # Load CSV data
            if not os.path.exists(self.csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
            
            # Read CSV with error handling
            try:
                df = pd.read_csv(self.csv_file_path)
            except pd.errors.EmptyDataError:
                raise ValueError("CSV file is empty")
            except pd.errors.ParserError as e:
                raise ValueError(f"CSV file parsing error: {e}")
            
            # Validate required columns
            required_columns = ['part_desc']  # Based on the CSV structure we saw
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV file missing required columns: {missing_columns}")
            
            # Extract unique parts
            parts_series = df['part_desc'].dropna().unique()
            parts_list = sorted([str(part).strip() for part in parts_series if str(part).strip()])
            
            if not parts_list:
                raise ValueError("No valid parts found in CSV file")
            
            # Cache the results
            self._cached_parts = parts_list
            self._cached_data = df
            
            self.logger.info(f"Loaded {len(parts_list)} unique parts from CSV")
            return parts_list
            
        except Exception as e:
            self.logger.error(f"Error loading parts from CSV: {e}")
            raise
    
    def get_cached_data(self) -> Optional[pd.DataFrame]:
        """
        Get cached CSV data.
        
        Returns:
            Cached DataFrame or None if not loaded
        """
        return self._cached_data
    
    async def show_searchable_dropdown(self) -> None:
        """
        Display searchable dropdown interface for part selection.
        
        This method creates Chainlit actions for part selection showing all available parts.
        """
        try:
            # Ensure parts are loaded
            if not self.session_state.available_parts:
                await self.load_available_parts()
            
            parts = self.session_state.available_parts
            
            if not parts:
                await cl.Message(
                    content="❌ No parts available for selection. Please check the CSV data."
                ).send()
                return
            
            # Create dropdown with all parts as clickable actions
            await self._show_parts_dropdown(parts)
            
        except Exception as e:
            error_context = await self.handle_error(e, "showing searchable dropdown")
            await cl.Message(content=error_context.format_user_message()).send()
    
    async def _show_parts_dropdown(self, parts: List[str]) -> None:
        """
        Show all parts in a dropdown format with clickable actions.
        
        Args:
            parts: List of all available parts
        """
        try:
            # Create message with part selection instructions
            content = f"📋 **Select a Part for Analysis**\n\n"
            content += f"Found {len(parts)} available parts. Choose your selection method:\n\n"
            
            # If there are many parts (>20), show alphabetical browse options first
            if len(parts) > 20:
                await self._show_alphabetical_browse(parts)
            else:
                # Show all parts directly if there aren't too many
                await self._show_all_parts_directly(parts)
            
        except Exception as e:
            self.logger.error(f"Error showing parts dropdown: {e}")
            # Fallback to text-based selection
            await self._show_text_based_selection(parts)
    
    async def _show_alphabetical_browse(self, parts: List[str]) -> None:
        """
        Show alphabetical browse options for large part lists.
        
        Args:
            parts: List of all available parts
        """
        # Get first letters of all parts
        first_letters = sorted(set(part[0].upper() for part in parts if part))
        
        # Create browse actions for each letter
        browse_actions = []
        for letter in first_letters[:10]:  # Limit to 10 letters per message
            part_count = len([p for p in parts if p.upper().startswith(letter)])
            browse_actions.append(
                cl.Action(
                    name="browse_letter",
                    value=letter,
                    label=f"{letter} ({part_count} parts)"
                )
            )
        
        browse_content = f"**Browse by First Letter** ({len(parts)} total parts):"
        
        await cl.Message(
            content=browse_content,
            actions=browse_actions
        ).send()
        
        # Show remaining letters if any
        if len(first_letters) > 10:
            remaining_letters = first_letters[10:]
            remaining_actions = []
            for letter in remaining_letters[:10]:
                part_count = len([p for p in parts if p.upper().startswith(letter)])
                remaining_actions.append(
                    cl.Action(
                        name="browse_letter",
                        value=letter,
                        label=f"{letter} ({part_count} parts)"
                    )
                )
            
            await cl.Message(
                content="**More Letters:**",
                actions=remaining_actions
            ).send()
        
        # Also show first 10 parts directly for quick access
        await self._show_quick_access_parts(parts[:10])
    
    async def _show_quick_access_parts(self, parts: List[str]) -> None:
        """
        Show a quick access list of parts.
        
        Args:
            parts: List of parts to show for quick access
        """
        actions = []
        for part in parts:
            actions.append(
                cl.Action(
                    name="select_part",
                    value=part,
                    label=part
                )
            )
        
        await cl.Message(
            content="**Quick Access - First 10 Parts:**",
            actions=actions
        ).send()
        
        # Add search instructions
        search_content = "\n💡 **Alternative:** Type a part name directly in the chat to select it."
        search_content += "\n🔍 **Search Tips:** Type partial names to find parts (e.g., 'cable' to find cable-related parts)"
        
        await cl.Message(content=search_content).send()
    
    async def _show_all_parts_directly(self, parts: List[str]) -> None:
        """
        Show all parts directly when the list is manageable.
        
        Args:
            parts: List of all available parts
        """
        sorted_parts = sorted(parts)
        
        # Create actions for each part (Chainlit supports up to 10 actions per message)
        batch_size = 10
        total_batches = (len(sorted_parts) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(sorted_parts))
            batch_parts = sorted_parts[start_idx:end_idx]
            
            # Create actions for this batch
            actions = []
            for part in batch_parts:
                actions.append(
                    cl.Action(
                        name="select_part",
                        value=part,
                        label=part
                    )
                )
            
            # Create batch message
            if total_batches > 1:
                batch_content = f"**Parts {start_idx + 1}-{end_idx} of {len(sorted_parts)}:**"
            else:
                batch_content = "**All Available Parts:**"
            
            await cl.Message(
                content=batch_content,
                actions=actions
            ).send()
        
        # Add search instructions at the end
        search_content = "\n💡 **Alternative:** You can also type a part name directly in the chat to select it."
        search_content += "\n🔍 **Search Tips:** Type partial names to find parts (e.g., 'cable' to find cable-related parts)"
        
        await cl.Message(content=search_content).send()
    
    async def _show_text_based_selection(self, parts: List[str]) -> None:
        """
        Fallback method to show text-based part selection.
        
        Args:
            parts: List of available parts
        """
        content = f"📋 **Select a Part for Analysis**\n\n"
        content += f"Found {len(parts)} available parts:\n\n"
        
        # Show first 20 parts
        for i, part in enumerate(parts[:20]):
            content += f"{i+1:2d}. {part}\n"
        
        if len(parts) > 20:
            content += f"... and {len(parts) - 20} more parts\n"
        
        content += "\n**To select a part:** Type the exact part name in the chat."
        content += "\n**Example:** Type 'Adapter' to select the Adapter part."
        
        await cl.Message(content=content).send()
    
    async def show_parts_by_letter(self, letter: str) -> None:
        """
        Show parts that start with a specific letter.
        
        Args:
            letter: Letter to filter parts by
        """
        try:
            if not self.session_state.available_parts:
                await self.load_available_parts()
            
            parts = self.session_state.available_parts
            filtered_parts = [part for part in parts if part.upper().startswith(letter.upper())]
            
            if not filtered_parts:
                await cl.Message(
                    content=f"No parts found starting with '{letter.upper()}'. Try a different letter."
                ).send()
                return
            
            # Create actions for filtered parts
            actions = []
            for part in filtered_parts[:10]:  # Limit to 10 per message
                actions.append(
                    cl.Action(
                        name="select_part",
                        value=part,
                        label=part
                    )
                )
            
            content = f"**Parts starting with '{letter.upper()}' ({len(filtered_parts)} found):**"
            
            await cl.Message(
                content=content,
                actions=actions
            ).send()
            
            # If more than 10 parts, show remaining in text format
            if len(filtered_parts) > 10:
                remaining_content = "**Additional parts:**\n"
                for part in filtered_parts[10:]:
                    remaining_content += f"• {part}\n"
                remaining_content += "\n💡 Type the exact part name to select it."
                
                await cl.Message(content=remaining_content).send()
                
        except Exception as e:
            error_context = await self.handle_error(e, "showing parts by letter")
            await cl.Message(content=error_context.format_user_message()).send()
    
    async def show_part_selector(self) -> None:
        """
        Display part selector interface.
        
        This is an alias for show_searchable_dropdown for compatibility.
        """
        await self.show_searchable_dropdown()
    
    async def filter_parts(self, search_term: str) -> List[str]:
        """
        Filter parts based on search term with advanced keyword-based filtering.
        
        Args:
            search_term: Search keyword or phrase
            
        Returns:
            Filtered list of parts matching the search term
        """
        try:
            # Ensure parts are loaded
            if not self.session_state.available_parts:
                await self.load_available_parts()
            
            parts = self.session_state.available_parts
            
            if not search_term or not search_term.strip():
                return parts
            
            search_term = search_term.strip().lower()
            
            # Advanced filtering with multiple strategies
            filtered_parts = []
            
            # Strategy 1: Exact substring matching (highest priority)
            exact_matches = [
                part for part in parts
                if search_term in part.lower()
            ]
            
            # Strategy 2: Word-based matching for multi-word searches
            search_words = search_term.split()
            if len(search_words) > 1:
                word_matches = [
                    part for part in parts
                    if all(word in part.lower() for word in search_words)
                    and part not in exact_matches
                ]
                filtered_parts = exact_matches + word_matches
            else:
                # Strategy 3: Fuzzy matching for single words (starts with, contains)
                starts_with = [
                    part for part in parts
                    if part.lower().startswith(search_term)
                    and part not in exact_matches
                ]
                
                contains = [
                    part for part in parts
                    if search_term in part.lower()
                    and part not in exact_matches
                    and part not in starts_with
                ]
                
                filtered_parts = exact_matches + starts_with + contains
            
            # Remove duplicates while preserving order
            seen = set()
            unique_filtered = []
            for part in filtered_parts:
                if part not in seen:
                    seen.add(part)
                    unique_filtered.append(part)
            
            self.logger.info(f"Filtered {len(unique_filtered)} parts for search term: '{search_term}'")
            return unique_filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering parts: {e}")
            return []
    
    async def handle_part_selection(self, selected_part: str) -> bool:
        """
        Handle part selection and validation with comprehensive session state updates.
        
        This method performs thorough validation of the selected part and updates
        the session state appropriately. It implements requirements 1.3 and 1.5
        for part selection validation and state management.
        
        Args:
            selected_part: Selected part name
            
        Returns:
            True if selection was valid and processed successfully, False otherwise
        """
        try:
            # Enhanced input validation
            if not await self._validate_part_input(selected_part):
                return False
            
            selected_part = selected_part.strip()
            
            # Ensure parts are loaded before validation
            if not await self._ensure_parts_loaded():
                return False
            
            # Validate part exists in available parts
            if not await self._validate_part_exists(selected_part):
                return False
            
            # Check for data consistency
            if not await self._validate_data_consistency():
                return False
            
            # Update session state with comprehensive reset
            await self._update_session_state_for_selection(selected_part)
            
            # Confirm selection to user with enhanced feedback
            await self._send_selection_confirmation(selected_part)
            
            self.logger.info(f"Successfully selected and validated part: {selected_part}")
            return True
            
        except Exception as e:
            error_context = await self.handle_error(e, "handling part selection")
            await cl.Message(content=error_context.format_user_message()).send()
            return False
    
    async def _validate_part_input(self, selected_part: str) -> bool:
        """
        Validate the input part selection.
        
        Args:
            selected_part: Part name to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        # Check for None or empty input
        if selected_part is None:
            await cl.Message(
                content="❌ **Invalid Selection**: No part was selected. Please choose a part from the available options."
            ).send()
            return False
        
        # Check for empty or whitespace-only input
        if not selected_part or not selected_part.strip():
            await cl.Message(
                content="❌ **Invalid Selection**: Empty part name provided. Please select a valid part."
            ).send()
            return False
        
        # Check for excessively long input (potential data corruption)
        if len(selected_part.strip()) > 500:
            await cl.Message(
                content="❌ **Invalid Selection**: Part name is too long. Please select a valid part from the dropdown."
            ).send()
            return False
        
        return True
    
    async def _ensure_parts_loaded(self) -> bool:
        """
        Ensure that parts are loaded before validation.
        
        Returns:
            True if parts are loaded successfully, False otherwise
        """
        try:
            if not self.session_state.available_parts:
                self.logger.info("Parts not loaded, attempting to load from CSV")
                await self.load_available_parts()
            
            if not self.session_state.available_parts:
                await cl.Message(
                    content="❌ **System Error**: No parts are available for selection. "
                           "Please check the data source and try again."
                ).send()
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load parts during validation: {e}")
            await cl.Message(
                content="❌ **System Error**: Failed to load available parts. "
                       "Please try refreshing the part list or contact support."
            ).send()
            return False
    
    async def _validate_part_exists(self, selected_part: str) -> bool:
        """
        Validate that the selected part exists in available parts.
        
        Args:
            selected_part: Part name to validate
            
        Returns:
            True if part exists, False otherwise
        """
        if selected_part not in self.session_state.available_parts:
            # Provide helpful suggestions for similar parts
            similar_parts = await self._find_similar_parts(selected_part)
            
            error_message = f"❌ **Part Not Found**: '{selected_part}' is not available in the current dataset."
            
            if similar_parts:
                error_message += f"\n\n**Did you mean one of these?**"
                for part in similar_parts[:3]:  # Show top 3 suggestions
                    error_message += f"\n• {part}"
                error_message += f"\n\n💡 **Tip**: Use the search functionality to find the correct part name."
            else:
                error_message += f"\n\n💡 **Tip**: Use the search or browse functionality to see all available parts."
            
            await cl.Message(content=error_message).send()
            return False
        
        return True
    
    async def _find_similar_parts(self, selected_part: str) -> List[str]:
        """
        Find parts similar to the selected part for suggestions.
        
        Args:
            selected_part: Part name to find similar parts for
            
        Returns:
            List of similar part names
        """
        try:
            # Use the existing filter functionality to find similar parts
            similar_parts = await self.filter_parts(selected_part)
            
            # If no direct matches, try partial matching
            if not similar_parts:
                words = selected_part.lower().split()
                for word in words:
                    if len(word) > 2:  # Only use words longer than 2 characters
                        word_matches = await self.filter_parts(word)
                        similar_parts.extend(word_matches)
            
            # Remove duplicates and limit results
            seen = set()
            unique_similar = []
            for part in similar_parts:
                if part not in seen and part != selected_part:
                    seen.add(part)
                    unique_similar.append(part)
                    if len(unique_similar) >= 5:  # Limit to 5 suggestions
                        break
            
            return unique_similar
            
        except Exception as e:
            self.logger.error(f"Error finding similar parts: {e}")
            return []
    
    async def _validate_data_consistency(self) -> bool:
        """
        Validate data consistency before part selection.
        
        Returns:
            True if data is consistent, False otherwise
        """
        try:
            # Check if cached data is available and consistent
            cached_data = self.get_cached_data()
            if cached_data is None:
                self.logger.warning("No cached data available during part selection")
                return True  # Not a critical error, processing can continue
            
            # Validate that cached data has the expected structure
            if 'part_desc' not in cached_data.columns:
                self.logger.error("Cached data missing required 'part_desc' column")
                await cl.Message(
                    content="❌ **Data Error**: Data structure is inconsistent. Please reload the part list."
                ).send()
                return False
            
            # Check if available parts match cached data
            cached_parts = set(cached_data['part_desc'].dropna().unique())
            session_parts = set(self.session_state.available_parts)
            
            if cached_parts != session_parts:
                self.logger.warning("Mismatch between cached data and session parts, refreshing")
                # Refresh the parts list
                await self.load_available_parts()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data consistency validation failed: {e}")
            await cl.Message(
                content="❌ **Data Error**: Data consistency check failed. Please try reloading the part list."
            ).send()
            return False
    
    async def _update_session_state_for_selection(self, selected_part: str) -> None:
        """
        Update session state with comprehensive reset for new part selection.
        
        Args:
            selected_part: The selected part name
        """
        # Store previous selection for logging
        previous_part = self.session_state.selected_part
        
        # Update selected part
        self.session_state.selected_part = selected_part
        
        # Reset processing-related state (implements requirement 1.3)
        self.session_state.processing_complete = False
        self.session_state.visualizations_displayed = False
        self.session_state.dfg_images = None
        self.session_state.retrievers = None
        
        # Reset query-related state
        self.session_state.awaiting_context_selection = False
        self.session_state.current_context_mode = None
        
        # Log the state change
        if previous_part and previous_part != selected_part:
            self.logger.info(f"Part selection changed from '{previous_part}' to '{selected_part}', state reset")
        else:
            self.logger.info(f"Part selected: '{selected_part}', processing state initialized")
    
    async def _send_selection_confirmation(self, selected_part: str) -> None:
        """
        Send enhanced confirmation message to user.
        
        Args:
            selected_part: The selected part name
        """
        # Get part statistics if available
        part_info = await self._get_part_statistics(selected_part)
        
        content = f"✅ **Part Selected Successfully**\n\n"
        content += f"**Selected Part:** {selected_part}\n"
        
        if part_info:
            content += f"**Data Records:** {part_info.get('record_count', 'Unknown')} events\n"
            if part_info.get('date_range'):
                content += f"**Date Range:** {part_info['date_range']}\n"
        
        content += f"\n🔄 **Next Steps:**\n"
        content += f"• Data will be filtered for this part\n"
        content += f"• Process mining analysis will be performed\n"
        content += f"• Visualizations will be generated automatically\n"
        content += f"• You'll be able to ask questions once processing is complete\n\n"
        content += f"⏳ **Please wait while the system processes the data...**"
        
        await cl.Message(content=content).send()
    
    async def _get_part_statistics(self, selected_part: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for the selected part from cached data.
        
        Args:
            selected_part: Part name to get statistics for
            
        Returns:
            Dictionary with part statistics or None if not available
        """
        try:
            cached_data = self.get_cached_data()
            if cached_data is None:
                return None
            
            # Filter data for the selected part
            part_data = cached_data[cached_data['part_desc'] == selected_part]
            
            if part_data.empty:
                return None
            
            stats = {
                'record_count': len(part_data)
            }
            
            # Add date range if timestamp column exists
            if 'timestamp' in part_data.columns:
                try:
                    # Try to parse timestamps
                    timestamps = pd.to_datetime(part_data['timestamp'], errors='coerce')
                    valid_timestamps = timestamps.dropna()
                    
                    if not valid_timestamps.empty:
                        min_date = valid_timestamps.min().strftime('%Y-%m-%d')
                        max_date = valid_timestamps.max().strftime('%Y-%m-%d')
                        stats['date_range'] = f"{min_date} to {max_date}"
                except Exception:
                    pass  # Ignore timestamp parsing errors
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting part statistics: {e}")
            return None
    
    async def validate_selection_requirements(self) -> bool:
        """
        Validate that selection requirements are met before allowing process analysis.
        
        This method implements requirement 1.5: ensuring no process analysis proceeds
        without a valid part selection.
        
        Returns:
            True if requirements are met, False otherwise
        """
        try:
            # Check if a part is selected
            if not self.session_state.selected_part:
                await cl.Message(
                    content="❌ **Selection Required**: No part has been selected. "
                           "Please select a part before proceeding with analysis.\n\n"
                           "💡 **To select a part:** Use the part selection interface to choose "
                           "from the available options."
                ).send()
                return False
            
            # Validate the selected part still exists in available parts
            if not self.session_state.available_parts:
                await self.load_available_parts()
            
            if self.session_state.selected_part not in self.session_state.available_parts:
                await cl.Message(
                    content="❌ **Invalid Selection**: The previously selected part is no longer available. "
                           "Please select a new part from the current options."
                ).send()
                # Clear invalid selection
                self.session_state.selected_part = None
                return False
            
            # Check data consistency
            if not await self._validate_data_consistency():
                return False
            
            self.logger.info(f"Selection requirements validated for part: {self.session_state.selected_part}")
            return True
            
        except Exception as e:
            error_context = await self.handle_error(e, "validating selection requirements")
            await cl.Message(content=error_context.format_user_message()).send()
            return False
    
    async def show_realtime_search_interface(self) -> None:
        """
        Display real-time search interface that updates as user types.
        
        This provides an interactive search experience with immediate feedback.
        """
        try:
            content = "🔍 **Real-time Part Search**\n\n"
            content += "Type your search term below and results will appear instantly:\n\n"
            content += "**Search Instructions:**\n"
            content += "• Type any part of the part name\n"
            content += "• Search is case-insensitive\n"
            content += "• Use multiple keywords for better results\n"
            content += "• Results update as you type\n\n"
            content += "💡 **Examples:** 'cable', 'motor housing', 'connector'\n\n"
            content += "**Type your search term and press Enter:**"
            
            # Use text input for real-time search
            await cl.Message(content=content).send()
            
            # Wait for user input
            search_input = await cl.AskUserMessage(
                content="Enter search term:",
                timeout=60
            ).send()
            
            if search_input:
                search_term = search_input.content.strip()
                if search_term:
                    # Perform real-time filtering
                    filtered_parts = await self.filter_parts(search_term)
                    await self.show_filtered_parts(filtered_parts, search_term)
                else:
                    await self.show_searchable_dropdown()
            else:
                await self.show_searchable_dropdown()
                
        except Exception as e:
            error_context = await self.handle_error(e, "showing real-time search interface")
            await cl.Message(content=error_context.format_user_message()).send()

    async def show_filtered_parts(self, filtered_parts: List[str], search_term: str) -> None:
        """
        Display filtered parts as selectable actions with enhanced search feedback.
        
        Args:
            filtered_parts: List of parts matching the search
            search_term: Original search term
        """
        try:
            if not filtered_parts:
                content = f"🔍 **Search Results for '{search_term}'**\n\n"
                content += "❌ No parts found matching your search.\n\n"
                content += "**Suggestions:**\n"
                content += "• Try shorter keywords (e.g., 'cable' instead of 'cable head')\n"
                content += "• Check spelling\n"
                content += "• Use different terms\n"
                content += "• Browse all parts to see available options"
                
                actions = [
                    cl.Action(
                        name="search_parts_realtime",
                        value="",
                        label="🔍 Try Another Search"
                    ),
                    cl.Action(
                        name="show_all_parts",
                        value="show_all",
                        label="📋 Show All Parts"
                    ),
                    cl.Action(
                        name="browse_alphabetically",
                        value="browse",
                        label="📚 Browse Alphabetically"
                    )
                ]
            else:
                content = f"🔍 **Search Results for '{search_term}'**\n\n"
                content += f"✅ Found **{len(filtered_parts)}** matching parts:\n\n"
                
                # Show results with better formatting
                display_parts = filtered_parts[:15]  # Show more results
                for i, part in enumerate(display_parts, 1):
                    # Highlight search term in results
                    highlighted_part = self._highlight_search_term(part, search_term)
                    content += f"{i}. {highlighted_part}\n"
                
                if len(filtered_parts) > 15:
                    content += f"\n... and **{len(filtered_parts) - 15}** more results\n"
                
                content += f"\n💡 **Tip:** Click on any part to select it for analysis"
                
                # Create selection actions
                actions = []
                for part in display_parts:
                    actions.append(
                        cl.Action(
                            name="select_part",
                            value=part,
                            label=f"📦 {part}"
                        )
                    )
                
                # Add utility actions
                actions.extend([
                    cl.Action(
                        name="search_parts_realtime",
                        value="",
                        label="🔍 Refine Search"
                    ),
                    cl.Action(
                        name="show_all_parts",
                        value="show_all",
                        label="📋 Show All Parts"
                    )
                ])
                
                # If there are many results, add pagination
                if len(filtered_parts) > 15:
                    actions.append(
                        cl.Action(
                            name="show_more_results",
                            value=f"{search_term}|15",
                            label=f"➡️ Show More Results ({len(filtered_parts) - 15} remaining)"
                        )
                    )
            
            await cl.Message(
                content=content,
                actions=actions
            ).send()
            
        except Exception as e:
            error_context = await self.handle_error(e, "showing filtered parts")
            await cl.Message(content=error_context.format_user_message()).send()
    
    def _highlight_search_term(self, part_name: str, search_term: str) -> str:
        """
        Highlight search term in part name for better visual feedback.
        
        Args:
            part_name: Original part name
            search_term: Search term to highlight
            
        Returns:
            Part name with highlighted search term
        """
        try:
            if not search_term or not search_term.strip():
                return part_name
            
            search_term = search_term.strip().lower()
            part_lower = part_name.lower()
            
            # Find the position of the search term
            pos = part_lower.find(search_term)
            if pos != -1:
                # Create highlighted version
                before = part_name[:pos]
                match = part_name[pos:pos + len(search_term)]
                after = part_name[pos + len(search_term):]
                return f"{before}**{match}**{after}"
            
            return part_name
            
        except Exception:
            return part_name
    
    async def show_alphabetical_browser(self) -> None:
        """
        Display parts organized alphabetically for easy browsing.
        """
        try:
            parts = self.session_state.available_parts
            if not parts:
                await self.load_available_parts()
                parts = self.session_state.available_parts
            
            if not parts:
                await cl.Message(
                    content="❌ No parts available for selection."
                ).send()
                return
            
            # Group parts by first letter
            grouped_parts = {}
            for part in parts:
                first_letter = part[0].upper()
                if first_letter not in grouped_parts:
                    grouped_parts[first_letter] = []
                grouped_parts[first_letter].append(part)
            
            # Sort letters
            sorted_letters = sorted(grouped_parts.keys())
            
            content = f"📚 **Browse Parts Alphabetically** ({len(parts)} total)\n\n"
            content += "**Available letters:**\n"
            
            # Show letter distribution
            for letter in sorted_letters:
                count = len(grouped_parts[letter])
                content += f"• **{letter}**: {count} parts\n"
            
            content += "\n💡 **Select a letter to see all parts starting with that letter**"
            
            # Create letter selection actions
            actions = []
            for letter in sorted_letters:
                count = len(grouped_parts[letter])
                actions.append(
                    cl.Action(
                        name="browse_letter",
                        value=letter,
                        label=f"📝 {letter} ({count} parts)"
                    )
                )
            
            # Add utility actions
            actions.extend([
                cl.Action(
                    name="search_parts_realtime",
                    value="",
                    label="🔍 Search Instead"
                ),
                cl.Action(
                    name="show_all_parts",
                    value="show_all",
                    label="📋 Show All Parts"
                )
            ])
            
            await cl.Message(
                content=content,
                actions=actions
            ).send()
            
        except Exception as e:
            error_context = await self.handle_error(e, "showing alphabetical browser")
            await cl.Message(content=error_context.format_user_message()).send()

    async def show_parts_by_letter(self, letter: str) -> None:
        """
        Display all parts starting with a specific letter.
        
        Args:
            letter: Letter to filter by
        """
        try:
            parts = self.session_state.available_parts
            if not parts:
                await self.load_available_parts()
                parts = self.session_state.available_parts
            
            # Filter parts by letter
            letter_parts = [part for part in parts if part[0].upper() == letter.upper()]
            
            if not letter_parts:
                await cl.Message(
                    content=f"❌ No parts found starting with '{letter}'"
                ).send()
                return
            
            content = f"📝 **Parts starting with '{letter.upper()}'** ({len(letter_parts)} parts)\n\n"
            
            # Show all parts for this letter
            for i, part in enumerate(letter_parts, 1):
                content += f"{i}. {part}\n"
            
            content += f"\n💡 **Click on any part to select it for analysis**"
            
            # Create selection actions
            actions = []
            for part in letter_parts:
                actions.append(
                    cl.Action(
                        name="select_part",
                        value=part,
                        label=f"📦 {part}"
                    )
                )
            
            # Add navigation actions
            actions.extend([
                cl.Action(
                    name="browse_alphabetically",
                    value="browse",
                    label="📚 Back to Alphabetical Browse"
                ),
                cl.Action(
                    name="search_parts_realtime",
                    value="",
                    label="🔍 Search Parts"
                )
            ])
            
            await cl.Message(
                content=content,
                actions=actions
            ).send()
            
        except Exception as e:
            error_context = await self.handle_error(e, f"showing parts for letter {letter}")
            await cl.Message(content=error_context.format_user_message()).send()

    async def show_all_parts(self) -> None:
        """
        Display all available parts in a paginated format.
        """
        try:
            parts = self.session_state.available_parts
            if not parts:
                await self.load_available_parts()
                parts = self.session_state.available_parts
            
            if not parts:
                await cl.Message(
                    content="❌ No parts available for selection."
                ).send()
                return
            
            # Paginate parts (show 12 at a time for better UI)
            page_size = 12
            total_pages = (len(parts) + page_size - 1) // page_size
            
            content = f"📋 **All Available Parts** ({len(parts)} total)\n\n"
            content += f"**Page 1 of {total_pages}:**\n\n"
            
            # Show first page
            first_page_parts = parts[:page_size]
            for i, part in enumerate(first_page_parts, 1):
                content += f"{i}. {part}\n"
            
            content += f"\n💡 **Click on any part to select it for analysis**"
            
            # Create actions for first page
            actions = []
            for part in first_page_parts:
                actions.append(
                    cl.Action(
                        name="select_part",
                        value=part,
                        label=f"📦 {part}"
                    )
                )
            
            # Add navigation actions if needed
            if total_pages > 1:
                actions.append(
                    cl.Action(
                        name="next_page",
                        value="2",
                        label="➡️ Next Page"
                    )
                )
            
            # Add utility actions
            actions.extend([
                cl.Action(
                    name="search_parts_realtime",
                    value="",
                    label="🔍 Search Parts"
                ),
                cl.Action(
                    name="browse_alphabetically",
                    value="browse",
                    label="📚 Browse Alphabetically"
                )
            ])
            
            await cl.Message(
                content=content,
                actions=actions
            ).send()
            
        except Exception as e:
            error_context = await self.handle_error(e, "showing all parts")
            await cl.Message(content=error_context.format_user_message()).send()
    
    def get_part_count(self) -> int:
        """
        Get the total number of available parts.
        
        Returns:
            Number of available parts
        """
        return len(self.session_state.available_parts) if self.session_state.available_parts else 0
    
    def is_part_selected(self) -> bool:
        """
        Check if a part has been selected.
        
        Returns:
            True if a part is selected
        """
        return self.session_state.selected_part is not None
    
    def get_selected_part(self) -> Optional[str]:
        """
        Get the currently selected part.
        
        Returns:
            Selected part name or None if no part is selected
        """
        return self.session_state.selected_part
    
    async def reset_selection(self) -> None:
        """
        Reset part selection and allow user to select a new part.
        """
        try:
            self.session_state.reset_for_new_part()
            
            await cl.Message(
                content="🔄 **Part selection reset.** Please select a new part for analysis."
            ).send()
            
            await self.show_searchable_dropdown()
            
        except Exception as e:
            error_context = await self.handle_error(e, "resetting part selection")
            await cl.Message(content=error_context.format_user_message()).send()