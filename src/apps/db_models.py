"""SQLAlchemy database models for CraftsBuddy"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, TIMESTAMP, ForeignKey, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from typing import Dict, Any
import json

Base = declarative_base()

class User(Base):
    """User information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=True)  # Telegram user ID for authentication (nullable)
    email = Column(String(255), unique=True, nullable=True)  # Email for authentication (nullable)
    password_hash = Column(String(255), nullable=True)  # Hashed password for email auth (nullable)
    name = Column(String(255), nullable=False)
    brand_name = Column(String(255))
    phone_number = Column(String(20), unique=True, nullable=True)  # Phone number for authentication
    address = Column(Text)
    is_seller = Column(Boolean, default=False, nullable=False)
    phone_verified = Column(Boolean, default=False, nullable=False)  # Phone number verification status
    onboarding_completed = Column(Boolean, default=False, nullable=False)  # Seller onboarding status
    first_name = Column(String(100), nullable=True)  # Separate first name for onboarding
    last_name = Column(String(100), nullable=True)  # Separate last name for onboarding
    contact_info = Column(Text, nullable=True)  # Additional contact information
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    products = relationship("Product", back_populates="user", cascade="all, delete-orphan")

class Product(Base):
    """Product information"""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    product_name = Column(String(500), nullable=False)
    price = Column(Integer, nullable=False, default=0)  # Price in smallest currency unit
    description = Column(Text)
    local_image_path = Column(String(500))  # For migration compatibility
    cloud_image_url = Column(String(1000))  # Google Cloud Storage URL
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="products")
    specifications = relationship("ProductSpecification", back_populates="product", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "product_name": self.product_name,
            "price": self.price,
            "description": self.description,
            "local_image_path": self.local_image_path,
            "cloud_image_url": self.cloud_image_url,
            "specifications": {spec.spec_key: spec.spec_value for spec in self.specifications},
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ProductSpecification(Base):
    """Product specifications - flexible key-value storage"""
    __tablename__ = 'product_specifications'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    spec_key = Column(String(255), nullable=False)
    spec_value = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    product = relationship("Product", back_populates="specifications")
    
    # Unique constraint
    __table_args__ = (
        {'schema': None},
    )

# Database configuration
class DatabaseConfig:
    """Database configuration and session management"""
    
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
