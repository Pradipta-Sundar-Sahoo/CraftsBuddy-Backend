"""Configuration settings for CraftBuddy application"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AppConfig:
    """Application configuration class"""
    
    # Gemini AI Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = "gemini-1.5-flash"
    USE_GEMINI: bool = bool(GEMINI_API_KEY)
    
    # Session Configuration
    CONTEXT_TIMEOUT_SECONDS: int = 15 * 60  # 15 minutes
    
    # Google Cloud Storage Configuration (Required)
    GCS_BUCKET_NAME: Optional[str] = os.getenv("GCS_BUCKET_NAME")
    GCS_CREDENTIALS_PATH: Optional[str] = os.getenv("GCS_CREDENTIALS_PATH")
    USE_GCS: bool = bool(GCS_BUCKET_NAME)
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDINGS_MODEL: str = "text-embedding-3-small"
    USE_OPENAI: bool = bool(OPENAI_API_KEY)
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "craftbuddy-products")
    USE_PINECONE: bool = bool(PINECONE_API_KEY and PINECONE_ENVIRONMENT)
    
    # Buyer Query Agent Configuration
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    MAX_RETRIEVED_PRODUCTS: int = int(os.getenv("MAX_RETRIEVED_PRODUCTS", "20"))
    MAX_RERANKED_PRODUCTS: int = int(os.getenv("MAX_RERANKED_PRODUCTS", "10"))
    
    # Validation
    def validate(self) -> bool:
        """Validate required configuration"""
        if not self.GCS_BUCKET_NAME:
            raise ValueError("GCS_BUCKET_NAME is required for cloud storage")
        return True

# Global configuration instance
config = AppConfig()
