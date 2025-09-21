#!/usr/bin/env python3
"""
Initialization script for the Buyer Query System
Run this script to set up the vector database and initialize product embeddings
"""

import os
import sys
import asyncio
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.apps.buyer_query_agent import setup_buyer_query_system
from src.core.settings import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Initialize the buyer query system"""
    try:
        logger.info("🚀 Initializing Buyer Query System with Pinecone and OpenAI...")
        
        # Check required environment variables
        required_vars = {
            "GEMINI_API_KEY": config.GEMINI_API_KEY,
            "OPENAI_API_KEY": config.OPENAI_API_KEY,
            "PINECONE_API_KEY": config.PINECONE_API_KEY,
            "PINECONE_ENVIRONMENT": config.PINECONE_ENVIRONMENT
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            logger.info("Please set the following environment variables:")
            logger.info("  - GEMINI_API_KEY: Your Google Gemini API key")
            logger.info("  - OPENAI_API_KEY: Your OpenAI API key")
            logger.info("  - PINECONE_API_KEY: Your Pinecone API key")
            logger.info("  - PINECONE_ENVIRONMENT: Your Pinecone environment (e.g., 'us-east1-gcp')")
            sys.exit(1)
        
        # Setup the system
        success = await setup_buyer_query_system()
        
        if success:
            logger.info("✅ Buyer Query System initialized successfully!")
            logger.info("🎯 System ready to process buyer queries at /api/catalog/query")
            logger.info("📊 Available features:")
            logger.info("  - Natural language query processing")
            logger.info("  - Intent detection and sentiment analysis")
            logger.info("  - Vector similarity search with Pinecone")
            logger.info("  - Product recommendations and related items")
            logger.info("  - Multi-agent workflow with LangGraph")
        else:
            logger.error("❌ System initialization failed")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
