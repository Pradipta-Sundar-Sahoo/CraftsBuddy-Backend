#!/usr/bin/env python3
"""
Create initial database tables for CraftBuddy Backend
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.core.database import get_database_config

# Load environment variables
load_dotenv()

def create_tables():
    """Create all database tables"""
    print("🏗️  Creating CraftBuddy database tables...")
    
    try:
        # Get database configuration
        db_config = get_database_config()
        
        # Create all tables
        db_config.create_tables()
        
        print("✅ All tables created successfully!")
        print("\n📋 Tables created:")
        print("  • users")
        print("  • products") 
        print("  • product_specifications")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🏗️  CraftBuddy Database Schema Creator")
    print("=" * 50)
    
    if create_tables():
        print("\n🎉 Database schema created successfully!")
        print("💡 You can now run migrations with: python run_migration.py")
    else:
        print("\n💥 Schema creation failed. Please check your database configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
