"""Catalog service for product listing and search functionality"""
from typing import List, Tuple
from src.core.database_service import DatabaseService
from src.apps.db_models import Product
import logging

logger = logging.getLogger(__name__)

class CatalogService:
    """Service for catalog operations"""
    
    def __init__(self):
        self.db_service = DatabaseService()
    
    def get_products_paginated(self, page: int = 1, per_page: int = 20) -> Tuple[List[dict], int]:
        """Get paginated list of products"""
        logger.info(f"📖 Getting products - page: {page}, per_page: {per_page}")
        
        offset = (page - 1) * per_page
        products = self.db_service.get_all_products(limit=per_page, offset=offset)
        
        # Get total count (simplified - in production, use a separate count query)
        total = len(self.db_service.get_all_products(limit=1000))  # TODO: Optimize this
        
        logger.info(f"📊 Found {len(products)} products on page {page}, total: {total}")
        return products, total
    
    def search_products(self, search_term: str, page: int = 1, per_page: int = 20) -> Tuple[List[dict], int]:
        """Search products by name"""
        logger.info(f"🔍 Searching products for: '{search_term}' - page: {page}, per_page: {per_page}")
        
        if not search_term.strip():
            return self.get_products_paginated(page, per_page)
        
        # For now, we'll use the simple search and handle pagination manually
        all_results = self.db_service.search_products(search_term, limit=1000)
        total = len(all_results)
        
        # Manual pagination
        start = (page - 1) * per_page
        end = start + per_page
        products = all_results[start:end]
        
        logger.info(f"🎯 Found {len(products)} products on page {page}, total matches: {total}")
        return products, total
    
    def get_product_details(self, product_id: int) -> dict:
        """Get detailed product information"""
        logger.info(f"🔍 Getting product details for ID: {product_id}")
        return self.db_service.get_product_by_id(product_id)
