"""Database configuration and setup"""
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from src.apps.db_models import DatabaseConfig

# Load environment variables
load_dotenv()

def get_database_url() -> str:
    """Get database URL from environment variables"""
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_ssl_mode = os.getenv('DB_SSL_MODE', 'require')
    
    if not all([db_host, db_name, db_user, db_password]):
        raise ValueError("Missing required database environment variables")
    
    # URL encode username and password to handle special characters
    encoded_user = quote_plus(db_user)
    encoded_password = quote_plus(db_password)
    
    return f"postgresql://{encoded_user}:{encoded_password}@{db_host}:{db_port}/{db_name}?sslmode={db_ssl_mode}"

def get_database_config() -> DatabaseConfig:
    """Get configured database instance"""
    db_url = get_database_url()
    return DatabaseConfig(db_url)
