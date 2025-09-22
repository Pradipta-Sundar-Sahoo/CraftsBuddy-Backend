"""Seller service for seller profile and product management"""
from typing import List, Optional, Dict, Any
from src.core.database_service import DatabaseService
from src.apps.db_models import User, Product
import logging

class SellerService:
    """Service for seller operations"""
    
    def __init__(self):
        self.db_service = DatabaseService()
    
    def get_seller_profile(self, user_id: int) -> Optional[User]:
        """Get seller profile information"""
        user = self.db_service.get_user_by_id(user_id)
        if user and user.is_seller:
            return user
        return None
    
    def update_seller_profile(self, user_id: int, profile_data: Dict[str, Any]) -> Optional[User]:
        """Update seller profile"""
        user = self.db_service.get_user_by_id(user_id)
        if not user:
            return None
        
        # Set as seller if not already
        if not user.is_seller:
            profile_data['is_seller'] = True
        
        return self.db_service.update_user(user_id, **profile_data)
    
    def get_seller_products(self, user_id: int) -> List[dict]:
        """Get all products for a seller"""
        user = self.db_service.get_user_by_id(user_id)
        if not user or not user.is_seller:
            return []
        
        return self.db_service.get_products_by_user(user_id)
    
    def create_product(self, user_id: int, product_data: Dict[str, Any]) -> Optional[dict]:
        """Create a new product for seller"""
        user = self.db_service.get_user_by_id(user_id)
        if not user or not user.is_seller:
            return None
        
        # Extract specifications if provided
        specifications = product_data.pop('specifications', None)
        
        return self.db_service.create_product(
            user_id=user_id,
            specifications=specifications,
            **product_data
        )
    
    def update_product(self, user_id: int, product_id: int, product_data: Dict[str, Any]) -> Optional[dict]:
        """Update a product (only if owned by seller)"""
        product = self.db_service.get_product_by_id(product_id)
        if not product or product.get('user_id') != user_id:
            return None
        
        # Handle specifications update separately if needed
        specifications = product_data.pop('specifications', None)
        if specifications:
            # TODO: Implement specification updates
            pass
        
        return self.db_service.update_product(product_id, **product_data)
    
    def delete_product(self, user_id: int, product_id: int) -> bool:
        """Soft delete a product (mark as inactive)"""
        # Verify product belongs to seller
        product = self.db_service.get_product_by_id(product_id)
        if not product or product.get('user_id') != user_id:
            return False
        
        # Soft delete by marking as inactive
        updated_product = self.db_service.update_product(product_id, is_active=False)
        return updated_product is not None
