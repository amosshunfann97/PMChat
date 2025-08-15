"""
Simple Chat Query Handler for testing imports.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ChatQueryHandler:
    """Simple handler for testing."""
    
    def __init__(self, session_state):
        self.session_state = session_state
        self.logger = logger
    
    def test_method(self):
        return "Test successful"