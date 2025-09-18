#!/usr/bin/env python3
"""
🚁 Vihangam Disaster Management Server Starter
Starts Django server with ASGI support for WebSocket connections
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚁 Starting Vihangam Disaster Management Server")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("disaster_dashboard"):
        print("❌ Error: Please run this script from the Vihangam root directory")
        return 1
    
    # Check virtual environment path  
    venv_path = Path("venv/bin/activate")
    if not venv_path.exists():
        # Try from current directory if running from Django directory
        venv_path = Path("../venv/bin/activate")
    if not venv_path.exists():
        print("❌ Error: Virtual environment not found. Please run:")
        print("  python -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        return 1
    
    try:
        # Change to Django directory
        os.chdir("disaster_dashboard")
        
        print("✅ Starting ASGI server with Daphne...")
        print("📡 WebSocket support: ENABLED")
        print("🤖 YOLO Model: Auto-detecting custom model")
        print("🌐 Server URL: http://localhost:8000/")
        print("🔧 Detection Interface: http://localhost:8000/detection/")
        print("")
        print("💡 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start Daphne ASGI server
        cmd = [
            "../venv/bin/python", "-m", "daphne", 
            "-p", "8000",
            "disaster_dashboard.asgi:application"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())