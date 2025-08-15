"""
Chat History Manager for Chainlit Integration.

This module manages chat history storage and retrieval for the process mining tool.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages chat history storage and retrieval."""
    
    def __init__(self, history_dir: str = "chat_history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.current_session_file = None
    
    def start_new_session(self, session_id: str) -> str:
        """Start a new chat session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_filename = f"session_{timestamp}_{session_id[:8]}.json"
        self.current_session_file = self.history_dir / session_filename
        
        # Initialize session file
        session_data = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "messages": [],
            "metadata": {
                "llm_type": None,
                "selected_part": None,
                "total_messages": 0
            }
        }
        
        self._save_session_data(session_data)
        logger.info(f"Started new chat session: {session_filename}")
        return str(self.current_session_file)
    
    def add_message(self, message_type: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the current session."""
        if not self.current_session_file or not self.current_session_file.exists():
            logger.warning("No active session to add message to")
            return
        
        session_data = self._load_session_data()
        if not session_data:
            return
        
        message = {
            "timestamp": datetime.now().isoformat(),
            "type": message_type,  # "user", "assistant", "system"
            "content": content,
            "metadata": metadata or {}
        }
        
        session_data["messages"].append(message)
        session_data["metadata"]["total_messages"] = len(session_data["messages"])
        
        self._save_session_data(session_data)
    
    def update_session_metadata(self, **kwargs):
        """Update session metadata."""
        if not self.current_session_file or not self.current_session_file.exists():
            return
        
        session_data = self._load_session_data()
        if not session_data:
            return
        
        session_data["metadata"].update(kwargs)
        self._save_session_data(session_data)
    
    def get_session_history(self, session_file: str = None) -> Optional[Dict[str, Any]]:
        """Get history for a specific session."""
        if session_file:
            file_path = Path(session_file)
        else:
            file_path = self.current_session_file
        
        if not file_path or not file_path.exists():
            return None
        
        return self._load_session_data(file_path)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available chat sessions."""
        sessions = []
        
        for session_file in self.history_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sessions.append({
                    "filename": session_file.name,
                    "session_id": data.get("session_id", "unknown"),
                    "start_time": data.get("start_time", "unknown"),
                    "message_count": len(data.get("messages", [])),
                    "llm_type": data.get("metadata", {}).get("llm_type"),
                    "selected_part": data.get("metadata", {}).get("selected_part"),
                    "file_path": str(session_file)
                })
            except Exception as e:
                logger.error(f"Error reading session file {session_file}: {e}")
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x["start_time"], reverse=True)
        return sessions
    
    def delete_session(self, session_file: str) -> bool:
        """Delete a chat session."""
        try:
            file_path = Path(session_file)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted session: {session_file}")
                return True
        except Exception as e:
            logger.error(f"Error deleting session {session_file}: {e}")
        
        return False
    
    def export_session(self, session_file: str = None, format: str = "json") -> Optional[str]:
        """Export session data in specified format."""
        session_data = self.get_session_history(session_file)
        if not session_data:
            return None
        
        if format == "json":
            return json.dumps(session_data, indent=2)
        elif format == "txt":
            lines = [f"Chat Session: {session_data.get('session_id', 'Unknown')}"]
            lines.append(f"Start Time: {session_data.get('start_time', 'Unknown')}")
            lines.append(f"LLM Type: {session_data.get('metadata', {}).get('llm_type', 'Unknown')}")
            lines.append(f"Selected Part: {session_data.get('metadata', {}).get('selected_part', 'None')}")
            lines.append("-" * 50)
            
            for msg in session_data.get("messages", []):
                timestamp = msg.get("timestamp", "Unknown")
                msg_type = msg.get("type", "unknown").upper()
                content = msg.get("content", "")
                lines.append(f"[{timestamp}] {msg_type}: {content}")
                lines.append("")
            
            return "\n".join(lines)
        
        return None
    
    def _load_session_data(self, file_path: Path = None) -> Optional[Dict[str, Any]]:
        """Load session data from file."""
        if file_path is None:
            file_path = self.current_session_file
        
        if not file_path or not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session data from {file_path}: {e}")
            return None
    
    def _save_session_data(self, data: Dict[str, Any], file_path: Path = None):
        """Save session data to file."""
        if file_path is None:
            file_path = self.current_session_file
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session data to {file_path}: {e}")


# Global chat history manager
chat_history = ChatHistoryManager()