"""Database service for CRUD operations"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
from src.apps.db_models import User, Product, ProductSpecification
from src.core.database import get_database_config

import logging
logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database operations"""
    
    def __init__(self):
        self.db_config = get_database_config()
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.db_config.get_session()
    
    # User operations
    def create_user(self, telegram_id: int = None, email: str = None, password_hash: str = None,
                   name: str = None, brand_name: str = None, phone_number: str = None, 
                   address: str = None, is_seller: bool = False) -> Optional[User]:
        """Create a new user"""
        try:
            with self.get_session() as session:
                stmt = text("""
                    INSERT INTO users (telegram_id, email, password_hash, name, brand_name, 
                                     phone_number, address, is_seller, phone_verified, onboarding_completed)
                    VALUES (:telegram_id, :email, :password_hash, :name, :brand_name, 
                            :phone_number, :address, :is_seller, false, false)
                    RETURNING id
                """)
                
                result = session.execute(stmt, {
                    "telegram_id": telegram_id,
                    "email": email,
                    "password_hash": password_hash,
                    "name": name,
                    "brand_name": brand_name,
                    "phone_number": phone_number,
                    "address": address,
                    "is_seller": is_seller
                }).fetchone()
                
                session.commit()
                user_id = result[0]
                
                # Return the created user
                return self.get_user_by_id(user_id)
        except SQLAlchemyError as e:
            logger.error(f"Failed to create user: {e}")
            return None
    
    def get_user_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        """Get user by telegram ID"""
        try:
            with self.get_session() as session:
                stmt = text("SELECT * FROM users WHERE telegram_id = :telegram_id")
                result = session.execute(stmt, {"telegram_id": telegram_id}).fetchone()
                if result:
                    # Convert to User object
                    user_data = dict(result._mapping)
                    user = User(**{k: v for k, v in user_data.items() if hasattr(User, k)})
                    return user
                return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get user by telegram_id {telegram_id}: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            with self.get_session() as session:
                stmt = text("SELECT * FROM users WHERE id = :user_id")
                result = session.execute(stmt, {"user_id": user_id}).fetchone()
                if result:
                    # Convert to User object
                    user_data = dict(result._mapping)
                    user = User(**{k: v for k, v in user_data.items() if hasattr(User, k)})
                    return user
                return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get user by id {user_id}: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            with self.get_session() as session:
                stmt = text("SELECT * FROM users WHERE email = :email")
                result = session.execute(stmt, {"email": email}).fetchone()
                if result:
                    # Convert to User object
                    user_data = dict(result._mapping)
                    user = User(**{k: v for k, v in user_data.items() if hasattr(User, k)})
                    return user
                return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    def get_user_by_phone(self, phone_number: str) -> Optional[User]:
        """Get user by phone number"""
        try:
            with self.get_session() as session:
                stmt = text("SELECT * FROM users WHERE phone_number = :phone_number")
                result = session.execute(stmt, {"phone_number": phone_number}).fetchone()
                if result:
                    # Convert to User object
                    user_data = dict(result._mapping)
                    user = User(**{k: v for k, v in user_data.items() if hasattr(User, k)})
                    return user
                return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get user by phone {phone_number}: {e}")
            return None
    
    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user information"""
        try:
            with self.get_session() as session:
                # Build dynamic UPDATE query
                if not kwargs:
                    return self.get_user_by_id(user_id)
                
                set_clause = ", ".join([f"{key} = :{key}" for key in kwargs.keys()])
                stmt = text(f"UPDATE users SET {set_clause} WHERE id = :user_id RETURNING *")
                
                params = kwargs.copy()
                params['user_id'] = user_id
                
                result = session.execute(stmt, params).fetchone()
                session.commit()
                
                if result:
                    return self.get_user_by_id(user_id)
                return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to update user: {e}")
            return None
    
    # Product operations
    def create_product(self, user_id: int, product_name: str, price: int, 
                      description: str = None, local_image_path: str = None,
                      cloud_image_url: str = None,
                      specifications: Dict[str, str] = None) -> Optional[dict]:
        """Create a new product with specifications"""
        try:
            with self.get_session() as session:
                # Insert product
                stmt = text("""
                    INSERT INTO products (user_id, product_name, price, description, 
                                        local_image_path, cloud_image_url, is_active)
                    VALUES (:user_id, :product_name, :price, :description, 
                            :local_image_path, :cloud_image_url, true)
                    RETURNING id
                """)
                
                result = session.execute(stmt, {
                    "user_id": user_id,
                    "product_name": product_name,
                    "price": price,
                    "description": description,
                    "local_image_path": local_image_path,
                    "cloud_image_url": cloud_image_url
                }).fetchone()
                
                product_id = result[0]
                
                # Add specifications
                if specifications:
                    for key, value in specifications.items():
                        spec_stmt = text("""
                            INSERT INTO product_specifications (product_id, spec_key, spec_value)
                            VALUES (:product_id, :spec_key, :spec_value)
                        """)
                        session.execute(spec_stmt, {
                            "product_id": product_id,
                            "spec_key": key,
                            "spec_value": value
                        })
                
                session.commit()
                
                # Return the created product as dict
                return self.get_product_by_id(product_id)
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to create product: {e}")
            return None
    
    def get_products_by_user(self, user_id: int) -> List[dict]:
        """Get all products for a user"""
        try:
            with self.get_session() as session:
                stmt = text("""
                    SELECT p.*, 
                           COALESCE(
                               json_object_agg(ps.spec_key, ps.spec_value) 
                               FILTER (WHERE ps.spec_key IS NOT NULL), 
                               '{}'::json
                           ) as specifications
                    FROM products p
                    LEFT JOIN product_specifications ps ON p.id = ps.product_id
                    WHERE p.user_id = :user_id AND p.is_active = true
                    GROUP BY p.id
                    ORDER BY p.created_at DESC
                """)
                
                results = session.execute(stmt, {"user_id": user_id}).fetchall()
                
                return [dict(row._mapping) for row in results]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get products for user_id {user_id}: {e}")
            return []
    
    def get_all_products(self, limit: int = 50, offset: int = 0) -> List[dict]:
        """Get all active products with pagination"""
        try:
            with self.get_session() as session:
                stmt = text("""
                    SELECT p.*, 
                           COALESCE(
                               json_object_agg(ps.spec_key, ps.spec_value) 
                               FILTER (WHERE ps.spec_key IS NOT NULL), 
                               '{}'::json
                           ) as specifications
                    FROM products p
                    LEFT JOIN product_specifications ps ON p.id = ps.product_id
                    WHERE p.is_active = true
                    GROUP BY p.id
                    ORDER BY p.created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                
                results = session.execute(stmt, {"limit": limit, "offset": offset}).fetchall()
                
                return [dict(row._mapping) for row in results]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get all products: {e}")
            return []
    
    def search_products(self, search_term: str, limit: int = 50) -> List[dict]:
        """Search products by name or description"""
        try:
            with self.get_session() as session:
                stmt = text("""
                    SELECT p.*, 
                           COALESCE(
                               json_object_agg(ps.spec_key, ps.spec_value) 
                               FILTER (WHERE ps.spec_key IS NOT NULL), 
                               '{}'::json
                           ) as specifications
                    FROM products p
                    LEFT JOIN product_specifications ps ON p.id = ps.product_id
                    WHERE p.is_active = true 
                    AND p.product_name ILIKE :search_term
                    GROUP BY p.id
                    ORDER BY p.created_at DESC
                    LIMIT :limit
                """)
                
                results = session.execute(stmt, {
                    "search_term": f"%{search_term}%", 
                    "limit": limit
                }).fetchall()
                
                return [dict(row._mapping) for row in results]
        except SQLAlchemyError as e:
            logger.error(f"Failed to search products: {e}")
            return []
    
    def get_product_by_id(self, product_id: int) -> Optional[dict]:
        """Get product by ID"""
        try:
            with self.get_session() as session:
                stmt = text("""
                    SELECT p.*, 
                           COALESCE(
                               json_object_agg(ps.spec_key, ps.spec_value) 
                               FILTER (WHERE ps.spec_key IS NOT NULL), 
                               '{}'::json
                           ) as specifications
                    FROM products p
                    LEFT JOIN product_specifications ps ON p.id = ps.product_id
                    WHERE p.id = :product_id
                    GROUP BY p.id
                """)
                
                result = session.execute(stmt, {"product_id": product_id}).fetchone()
                
                return dict(result._mapping) if result else None
        except SQLAlchemyError as e:
            logger.error(f"Failed to get product by id {product_id}: {e}")
            return None
    
    def update_product(self, product_id: int, **kwargs) -> Optional[dict]:
        """Update product information"""
        try:
            with self.get_session() as session:
                # Build dynamic UPDATE query
                if not kwargs:
                    return self.get_product_by_id(product_id)
                
                set_clause = ", ".join([f"{key} = :{key}" for key in kwargs.keys()])
                stmt = text(f"UPDATE products SET {set_clause} WHERE id = :product_id RETURNING *")
                
                params = kwargs.copy()
                params['product_id'] = product_id
                
                result = session.execute(stmt, params).fetchone()
                session.commit()
                
                if result:
                    return self.get_product_by_id(product_id)
                return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to update product: {e}")
            return None
    
    # Migration helper methods
    def migrate_json_data(self, json_file_path: str, chat_id: int) -> bool:
        """Migrate data from JSON files to database"""
        import json
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get or create user
            user = self.get_user_by_telegram_id(chat_id)
            if not user:
                user = self.create_user(telegram_id=chat_id, name=f"User_{chat_id}", is_seller=True)
                if not user:
                    return False
            
            # Create product
            product = self.create_product(
                user_id=user.id,
                product_name=data.get('product_name', 'Unknown Product'),
                price=data.get('price', 0),
                description=data.get('description', ''),
                local_image_path=data.get('local_image_path'),
                specifications=data.get('specifications', {})
            )
            
            return product is not None
            
        except Exception as e:
            logger.error(f"Failed to migrate JSON data from {json_file_path}: {e}")
            return False
    
    def _product_to_dict_with_session(self, product: Product, session) -> dict:
        """Convert product to dictionary while session is active"""
        try:
            # Force load specifications while session is active
            specs = {spec.spec_key: spec.spec_value for spec in product.specifications}
            
            return {
                "id": product.id,
                "user_id": product.user_id,
                "product_name": product.product_name,
                "price": product.price,
                "description": product.description,
                "local_image_path": product.local_image_path,
                "cloud_image_url": product.cloud_image_url,
                "is_active": product.is_active,
                "created_at": product.created_at,
                "updated_at": product.updated_at,
                "specifications": specs
            }
        except Exception as e:
            logger.error(f"Error converting product to dict: {e}")
            # Return basic product info without specifications
            return {
                "id": product.id,
                "user_id": product.user_id,
                "product_name": product.product_name,
                "price": product.price,
                "description": product.description,
                "local_image_path": product.local_image_path,
                "cloud_image_url": product.cloud_image_url,
                "is_active": product.is_active,
                "created_at": product.created_at,
                "updated_at": product.updated_at,
                "specifications": {}
            }
