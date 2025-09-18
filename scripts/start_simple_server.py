#!/usr/bin/env python3
"""
🚁 Vihangam Simple Server Starter
Starts Django server with basic functionality (no YOLO dependencies)
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚁 Starting Vihangam Dashboard Server")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("disaster_dashboard"):
        print("❌ Error: Please run this script from the Vihangam root directory")
        return 1
    
    # Check virtual environment
    venv_path = Path("venv/bin/activate")
    if not venv_path.exists():
        print("❌ Error: Virtual environment not found")
        print("  Run: python -m venv venv && source venv/bin/activate && pip install django")
        return 1
    
    try:
        # Change to Django directory
        os.chdir("disaster_dashboard")
        
        print("✅ Starting Django development server...")
        print("🌐 Server URL: http://localhost:8000/")
        print("📊 Dashboard: http://localhost:8000/dashboard/")
        print("🔍 Detection: http://localhost:8000/detection/")
        print("🧭 Pathfinding: http://localhost:8000/pathfinding/")
        print("")
        print("💡 Press Ctrl+C to stop the server")
        print("-" * 40)
        
        # Start regular Django server (no ASGI needed for simple version)
        cmd = [
            "../venv/bin/python", "manage.py", "runserver"
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