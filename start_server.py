#!/usr/bin/env python3
"""
CraftBuddy Backend Server Startup Script
"""
import os
import sys
from pathlib import Path

def main():
    """Start the CraftBuddy backend server"""
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        import uvicorn
        from main import app
        
        print("🚀 Starting CraftBuddy Backend Server...")
        print("📖 API Documentation will be available at:")
        print("   - Swagger UI: http://localhost:8000/docs")
        print("   - ReDoc: http://localhost:8000/redoc")
        print("🔧 Health check: http://localhost:8000/health")
        print("📋 Routes list: http://localhost:8000/api/routes")
        print("\n" + "="*50)
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload during development
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("📦 Please install requirements first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
