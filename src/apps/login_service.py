"""Authentication service for user login and token management"""
from typing import Optional
import jwt
import bcrypt
from datetime import datetime, timedelta
from src.core.database_service import DatabaseService
from src.apps.db_models import User
import logging

logger = logging.getLogger(__name__)

class AuthService:
    """Service for handling authentication and authorization"""
    
    def __init__(self):
        self.db_service = DatabaseService()
        # In production, use a secure secret key from environment
        self.secret_key = "your-secret-key-here"  # TODO: Move to environment variables
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
    
    def authenticate_telegram_user(self, telegram_id: int) -> Optional[User]:
        """Authenticate user by telegram ID"""
        logger.info(f"🔐 Authenticating user with telegram_id: {telegram_id}")
        
        # Get user by telegram ID
        user = self.db_service.get_user_by_telegram_id(telegram_id)
        
        if not user:
            # Create new user if doesn't exist
            logger.info(f"👤 Creating new user for telegram_id: {telegram_id}")
            user = self.db_service.create_user(
                telegram_id=telegram_id,
                name=f"User_{telegram_id}",
                is_seller=False
            )
        
        return user
    
    def authenticate_phone_user(self, phone_number: str) -> Optional[User]:
        """Authenticate user by phone number"""
        logger.info(f"📱 Authenticating user with phone: {phone_number}")
        return self.db_service.get_user_by_phone(phone_number)
    
    def authenticate_email_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user by email and password"""
        logger.info(f"📧 Authenticating user with email: {email}")
        
        user = self.db_service.get_user_by_email(email)
        if user and user.password_hash and self.verify_password(password, user.password_hash):
            return user
        return None
    
    def create_phone_user(self, phone_number: str, name: str, is_seller: bool = False) -> Optional[User]:
        """Create new user with phone number"""
        logger.info(f"📱 Creating new user with phone: {phone_number}, name: {name}, is_seller: {is_seller}")
        
        # Check if user already exists
        existing_user = self.db_service.get_user_by_phone(phone_number)
        if existing_user:
            logger.warning(f"⚠️ User already exists with phone: {phone_number}")
            return None
        
        return self.db_service.create_user(
            phone_number=phone_number,
            name=name,
            is_seller=is_seller
        )
    
    def create_email_user(self, email: str, password: str, name: str, is_seller: bool = False) -> Optional[User]:
        """Create new user with email and password"""
        logger.info(f"📧 Creating new user with email: {email}, name: {name}, is_seller: {is_seller}")
        
        # Check if user already exists
        existing_user = self.db_service.get_user_by_email(email)
        if existing_user:
            logger.warning(f"⚠️ User already exists with email: {email}")
            return None
        
        # Hash password
        password_hash = self.hash_password(password)
        
        return self.db_service.create_user(
            email=email,
            password_hash=password_hash,
            name=name,
            is_seller=is_seller
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_access_token(self, user_id: int, remember_me: bool = False) -> str:
        """Create JWT access token"""
        # Extend expiry time if remember_me is True
        expire_minutes = 24 * 60 * 7 if remember_me else self.access_token_expire_minutes  # 7 days vs 30 minutes
        expire = datetime.utcnow() + timedelta(minutes=expire_minutes)
        
        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[int]:
        """Verify JWT token and return user ID"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = int(payload.get("sub"))
            return user_id
        except jwt.PyJWTError:
            return None
    
    def get_current_user(self, token: str) -> Optional[User]:
        """Get current user from token"""
        user_id = self.verify_token(token)
        if user_id:
            return self.db_service.get_user_by_id(user_id)
        return None
