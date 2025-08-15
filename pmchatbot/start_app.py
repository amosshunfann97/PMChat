#!/usr/bin/env python3
"""
Simple startup script for the Process Mining Analysis Tool.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Chainlit application."""
    print("🚀 Starting Process Mining Analysis Tool...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Check if virtual environment is activated
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("❌ Virtual environment not found. Please ensure .venv exists in the parent directory.")
        return 1
    
    # Start Chainlit application
    try:
        cmd = [
            str(venv_python), 
            "-m", "chainlit", "run", 
            "pmchatbot/src/chainlit_app.py", 
            "--port", "8003"
        ]
        
        print("📡 Starting server on http://localhost:8003")
        print("Press Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())