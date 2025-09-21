#!/usr/bin/env python3
"""
Manual database migration script for CraftBuddy Backend
Run this script to apply database migrations manually
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import psycopg2
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

def get_database_connection():
    """Get database connection"""
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    
    if not all([db_host, db_name, db_user, db_password]):
        print("❌ Missing required database environment variables:")
        print("   DB_HOST, DB_NAME, DB_USER, DB_PASSWORD")
        return None
    
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = %s AND column_name = %s
        );
    """, (table_name, column_name))
    return cursor.fetchone()[0]

def run_migration():
    """Run the database migration"""
    print("🔄 Starting CraftBuddy database migration...")
    
    conn = get_database_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check if users table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'users'
            );
        """)
        
        if not cursor.fetchone()[0]:
            print("❌ Users table does not exist. Please create the basic schema first.")
            return False
        
        print("✅ Users table found")
        
        # Add new columns if they don't exist
        migrations = [
            ("email", "ADD COLUMN email VARCHAR(255) UNIQUE"),
            ("password_hash", "ADD COLUMN password_hash VARCHAR(255)"),
            ("phone_verified", "ADD COLUMN phone_verified BOOLEAN DEFAULT FALSE NOT NULL"),
            ("onboarding_completed", "ADD COLUMN onboarding_completed BOOLEAN DEFAULT FALSE NOT NULL"),
            ("first_name", "ADD COLUMN first_name VARCHAR(100)"),
            ("last_name", "ADD COLUMN last_name VARCHAR(100)"),
            ("contact_info", "ADD COLUMN contact_info TEXT")
        ]
        
        for column_name, alter_sql in migrations:
            if not check_column_exists(cursor, 'users', column_name):
                print(f"➕ Adding column: {column_name}")
                cursor.execute(f"ALTER TABLE users {alter_sql}")
            else:
                print(f"⏭️  Column {column_name} already exists")
        
        # Make phone_number unique if not already
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.table_constraints 
            WHERE table_name = 'users' 
            AND constraint_name = 'uq_users_phone_number'
            AND constraint_type = 'UNIQUE'
        """)
        
        if cursor.fetchone()[0] == 0:
            print("➕ Adding unique constraint on phone_number")
            try:
                cursor.execute("ALTER TABLE users ADD CONSTRAINT uq_users_phone_number UNIQUE (phone_number)")
            except psycopg2.errors.UniqueViolation:
                print("⚠️  Cannot add unique constraint on phone_number - duplicate values exist")
        else:
            print("⏭️  Phone number unique constraint already exists")
        
        # Commit changes
        conn.commit()
        print("✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def check_migration_status():
    """Check current migration status"""
    print("🔍 Checking migration status...")
    
    conn = get_database_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Get table structure
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'users'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        if columns:
            print("\n📋 Current users table structure:")
            for col in columns:
                nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {col[3]}" if col[3] else ""
                print(f"  • {col[0]} ({col[1]}) {nullable}{default}")
        else:
            print("❌ Users table not found")
            
    except Exception as e:
        print(f"❌ Status check failed: {e}")
    finally:
        cursor.close()
        conn.close()

def main():
    """Main function"""
    print("🏗️  CraftBuddy Database Migration Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_migration_status()
    else:
        if run_migration():
            print("\n🎉 Database is ready for CraftBuddy Backend!")
            print("💡 You can now start the server with: python main.py")
        else:
            print("\n💥 Migration failed. Please check your database configuration.")
            sys.exit(1)

if __name__ == "__main__":
    main()
