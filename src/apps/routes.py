"""FastAPI routes for CraftBuddy application"""
from fastapi import APIRouter, HTTPException, Depends, status, Header
from typing import Optional, List
import logging
import json
from fastapi.encoders import jsonable_encoder

# Import services
from src.apps.login_service import AuthService
from src.apps.catalog_service import CatalogService
from src.apps.seller_service import SellerService
from src.apps.otp_service import OTPService
from src.apps.buyer_query_agent import process_buyer_query

# Import Pydantic models
from src.apps.interface import (
    UserCreate, UserUpdate, UserResponse,
    ProductCreate, ProductUpdate, ProductResponse, ProductListResponse,
    TelegramLoginRequest, PhoneLoginRequest, PhoneSignupRequest,
    EmailLoginRequest, EmailSignupRequest, LoginResponse, SignupResponse,
    SendOTPRequest, VerifyOTPRequest, OTPResponse,
    SellerOnboardingRequest, OnboardingCompleteResponse,
    ProductSearchRequest, BuyerQueryRequest, BuyerQueryResponse,
    MessageResponse, ErrorResponse
)

logger = logging.getLogger(__name__)

# Initialize services
auth_service = AuthService()
catalog_service = CatalogService()
seller_service = SellerService()
otp_service = OTPService()

# Create routers
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
catalog_router = APIRouter(prefix="/catalog", tags=["Catalog"])
seller_router = APIRouter(prefix="/seller", tags=["Seller"])

# Dependency for authentication
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Get current authenticated user from token"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    try:
        # Extract token from "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )
    
    user = auth_service.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    return user

# Dependency for seller authentication
async def get_current_seller(current_user = Depends(get_current_user)):
    """Get current authenticated seller"""
    if not current_user.is_seller:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seller access required"
        )
    return current_user

# ==================== AUTHENTICATION ROUTES ====================

@auth_router.post("/login/telegram", response_model=LoginResponse)
async def telegram_login(login_data: TelegramLoginRequest):
    """Login with Telegram ID"""
    logger.info(f"🔐 Telegram login attempt for telegram_id: {login_data.telegram_id}")
    
    try:
        user = auth_service.authenticate_telegram_user(login_data.telegram_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
        
        access_token = auth_service.create_access_token(user.id)
        
        return LoginResponse(
            user=UserResponse.from_orm(user),
            access_token=access_token
        )
    
    except Exception as e:
        logger.error(f"❌ Telegram login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@auth_router.post("/login/phone", response_model=LoginResponse)
async def phone_login(login_data: PhoneLoginRequest):
    """Login with phone number"""
    logger.info(f"📱 Phone login attempt for: {login_data.phone_number}")
    
    try:
        user = auth_service.authenticate_phone_user(login_data.phone_number)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Phone number not registered. Please sign up first."
            )
        
        access_token = auth_service.create_access_token(user.id)
        
        return LoginResponse(
            user=UserResponse.from_orm(user),
            access_token=access_token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Phone login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@auth_router.post("/login/email", response_model=LoginResponse)
async def email_login(login_data: EmailLoginRequest):
    """Login with email and password"""
    logger.info(f"📧 Email login attempt for: {login_data.email}")
    
    try:
        user = auth_service.authenticate_email_user(login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        access_token = auth_service.create_access_token(user.id, login_data.remember_me)
        
        return LoginResponse(
            user=UserResponse.from_orm(user),
            access_token=access_token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Email login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@auth_router.post("/signup/phone", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def phone_signup(signup_data: PhoneSignupRequest):
    """Sign up with phone number"""
    logger.info(f"📱 Phone signup attempt for: {signup_data.phone_number}, type: {signup_data.user_type}")
    
    try:
        is_seller = signup_data.user_type == "seller"
        user = auth_service.create_phone_user(
            phone_number=signup_data.phone_number,
            name=signup_data.name,
            is_seller=is_seller
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Phone number already registered. Please login instead."
            )
        
        access_token = auth_service.create_access_token(user.id)
        
        return SignupResponse(
            user=UserResponse.from_orm(user),
            access_token=access_token,
            message=f"Account created successfully as {'seller' if is_seller else 'buyer'}",
            requires_onboarding=is_seller and not user.onboarding_completed
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Phone signup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signup failed"
        )

@auth_router.post("/signup/email", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def email_signup(signup_data: EmailSignupRequest):
    """Sign up with email and password"""
    logger.info(f"📧 Email signup attempt for: {signup_data.email}, type: {signup_data.user_type}")
    
    try:
        is_seller = signup_data.user_type == "seller"
        user = auth_service.create_email_user(
            email=signup_data.email,
            password=signup_data.password,
            name=signup_data.name,
            is_seller=is_seller
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered. Please login instead."
            )
        
        access_token = auth_service.create_access_token(user.id)
        
        return SignupResponse(
            user=UserResponse.from_orm(user),
            access_token=access_token,
            message=f"Account created successfully as {'seller' if is_seller else 'buyer'}",
            requires_onboarding=is_seller and not user.onboarding_completed
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Email signup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signup failed"
        )

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse.from_orm(current_user)

@auth_router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user = Depends(get_current_user)
):
    """Update current user information"""
    try:
        updated_user = seller_service.db_service.update_user(
            current_user.id, 
            **user_data.dict(exclude_unset=True)
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse.from_orm(updated_user)
    
    except Exception as e:
        logger.error(f"❌ User update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

# ==================== OTP VERIFICATION ROUTES ====================

@auth_router.post("/send-otp", response_model=OTPResponse)
async def send_otp(otp_request: SendOTPRequest):
    """Send OTP to phone number"""
    logger.info(f"📱 Sending OTP to: {otp_request.phone_number}")
    
    try:
        success = otp_service.send_otp(otp_request.phone_number)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send OTP"
            )
        
        return OTPResponse(
            message=f"OTP sent to {otp_request.phone_number}. Use hardcoded OTP: {otp_service.get_hardcoded_otp()}",
            otp_sent=True
        )
    
    except Exception as e:
        logger.error(f"❌ Send OTP error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP"
        )

@auth_router.post("/verify-otp", response_model=LoginResponse)
async def verify_otp(verify_request: VerifyOTPRequest):
    """Verify OTP and login user"""
    logger.info(f"🔍 Verifying OTP for: {verify_request.phone_number}")
    
    try:
        # Verify OTP
        if not otp_service.verify_otp(verify_request.phone_number, verify_request.otp):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired OTP"
            )
        
        # Get user by phone number
        user = auth_service.authenticate_phone_user(verify_request.phone_number)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found. Please sign up first."
            )
        
        # Mark phone as verified
        updated_user = seller_service.db_service.update_user(user.id, phone_verified=True)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user verification status"
            )
        
        # Create access token
        access_token = auth_service.create_access_token(updated_user.id)
        
        return LoginResponse(
            user=UserResponse.from_orm(updated_user),
            access_token=access_token
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Verify OTP error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OTP verification failed"
        )

# ==================== SELLER ONBOARDING ROUTES ====================

@auth_router.post("/complete-onboarding", response_model=OnboardingCompleteResponse)
async def complete_seller_onboarding(
    onboarding_data: SellerOnboardingRequest,
    current_user = Depends(get_current_user)
):
    """Complete seller onboarding process"""
    logger.info(f"📋 Completing onboarding for user: {current_user.id}")
    
    try:
        # Check if user is a seller
        if not current_user.is_seller:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only sellers can complete onboarding"
            )
        
        # Check if already completed
        if current_user.onboarding_completed:
            return OnboardingCompleteResponse(
                user=UserResponse.from_orm(current_user),
                message="Onboarding already completed"
            )
        
        # Update user with onboarding data
        update_data = {
            "first_name": onboarding_data.first_name,
            "last_name": onboarding_data.last_name,
            "contact_info": onboarding_data.contact_info,
            "onboarding_completed": True
        }
        
        # Update email if provided and different
        if onboarding_data.email and onboarding_data.email != current_user.email:
            update_data["email"] = onboarding_data.email
        
        # Update brand name if provided
        if onboarding_data.brand_name:
            update_data["brand_name"] = onboarding_data.brand_name
        
        # Update full name combining first and last name
        full_name = f"{onboarding_data.first_name} {onboarding_data.last_name}".strip()
        if full_name:
            update_data["name"] = full_name
        
        updated_user = seller_service.db_service.update_user(current_user.id, **update_data)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to complete onboarding"
            )
        
        return OnboardingCompleteResponse(
            user=UserResponse.from_orm(updated_user),
            message="Onboarding completed successfully! Welcome to CraftBuddy."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Complete onboarding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete onboarding"
        )

@auth_router.get("/onboarding-status")
async def get_onboarding_status(current_user = Depends(get_current_user)):
    """Get user's onboarding status"""
    return {
        "is_seller": current_user.is_seller,
        "phone_verified": current_user.phone_verified,
        "onboarding_completed": current_user.onboarding_completed,
        "requires_onboarding": current_user.is_seller and not current_user.onboarding_completed
    }

# ==================== CATALOG ROUTES ====================

@catalog_router.get("/products", response_model=ProductListResponse)
async def get_products(
    page: int = 1,
    per_page: int = 20,
    search: Optional[str] = None
):
    """Get paginated list of products with optional search"""
    logger.info(f"📖 Getting products - page: {page}, per_page: {per_page}, search: {search}")
    
    try:
        if search:
            products, total = catalog_service.search_products(search, page, per_page)
        else:
            products, total = catalog_service.get_products_paginated(page, per_page)
        
        # Convert to response models
        product_responses = []
        for product in products:
            # product is already a dict from database service
            product_responses.append(ProductResponse(**product))
        
        response = ProductListResponse(
            products=product_responses,
            total=total,
            page=page,
            per_page=per_page
        )
        with open('response.json', 'w') as f:
            json.dump(jsonable_encoder(response.model_dump()), f)
        return response
    
    except Exception as e:
        logger.error(f"❌ Get products error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch products"
        )

@catalog_router.get("/products/{product_id}", response_model=ProductResponse)
async def get_product_details(product_id: int):
    """Get detailed information about a specific product"""
    logger.info(f"🔍 Getting product details for ID: {product_id}")
    
    try:
        product = catalog_service.get_product_details(product_id)
        if not product or not product.get('is_active', False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found"
            )
        
        # product is already a dict from database service
        return ProductResponse(**product)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get product details error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch product details"
        )

@catalog_router.post("/query", response_model=BuyerQueryResponse)
async def process_buyer_query_endpoint(
    query_request: BuyerQueryRequest,
    current_user = Depends(get_current_user)
):
    """Process a buyer query using AI agents"""
    logger.info(f"🤖 Processing buyer query: '{query_request.query}' for user: {current_user.id}")
    
    try:
        # Process the query through the agent system
        result = await process_buyer_query(
            query=query_request.query,
            user_id=current_user.id
        )
        
        # Convert product dictionaries to ProductResponse objects if needed
        if result.get('products'):
            products = []
            for product in result['products']:
                # Handle both dict and ProductResponse objects
                if isinstance(product, dict):
                    products.append(ProductResponse(**product))
                else:
                    products.append(product)
            result['products'] = products
        
        if result.get('related_products'):
            related_products = []
            for product in result['related_products']:
                if isinstance(product, dict):
                    related_products.append(ProductResponse(**product))
                else:
                    related_products.append(product)
            result['related_products'] = related_products
        
        # Return the structured response
        return BuyerQueryResponse(**result)
    
    except Exception as e:
        logger.error(f"❌ Buyer query processing error: {e}")
        return BuyerQueryResponse(
            type="error",
            message="I'm having trouble processing your request right now. Please try again later.",
            processing_time=0.0,
            workflow_path=["error"],
            query_metadata={"error": str(e)}
        )

# ==================== SELLER ROUTES ====================

@seller_router.get("/profile", response_model=UserResponse)
async def get_seller_profile(current_seller = Depends(get_current_seller)):
    """Get seller profile information"""
    return UserResponse.from_orm(current_seller)

@seller_router.put("/profile", response_model=UserResponse)
async def update_seller_profile(
    profile_data: UserUpdate,
    current_seller = Depends(get_current_seller)
):
    """Update seller profile"""
    try:
        updated_user = seller_service.update_seller_profile(
            current_seller.id,
            profile_data.dict(exclude_unset=True)
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Seller not found"
            )
        
        return UserResponse.from_orm(updated_user)
    
    except Exception as e:
        logger.error(f"❌ Seller profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update seller profile"
        )

@seller_router.get("/products", response_model=List[ProductResponse])
async def get_seller_products(current_seller = Depends(get_current_seller)):
    """Get all products for the current seller"""
    try:
        products = seller_service.get_seller_products(current_seller.id)
        
        # Convert to response models
        product_responses = []
        for product in products:
            # Check if it's a Product object or dict
            if hasattr(product, 'to_dict'):
                product_dict = product.to_dict()
                product_responses.append(ProductResponse(**product_dict))
            else:
                product_responses.append(ProductResponse(**product))
        
        return product_responses
    
    except Exception as e:
        logger.error(f"❌ Get seller products error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch seller products"
        )

@seller_router.post("/products", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    product_data: ProductCreate,
    current_seller = Depends(get_current_seller)
):
    """Create a new product"""
    try:
        product = seller_service.create_product(
            current_seller.id,
            product_data.dict()
        )
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create product"
            )
        
        # product is already a dict from database service
        return ProductResponse(**product)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Create product error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create product"
        )

@seller_router.get("/products/{product_id}", response_model=ProductResponse)
async def get_seller_product(
    product_id: int,
    current_seller = Depends(get_current_seller)
):
    """Get specific product details for seller"""
    try:
        product = seller_service.db_service.get_product_by_id(product_id)
        
        if not product or product.get('user_id') != current_seller.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found"
            )
        
        # product is already a dict from database service
        return ProductResponse(**product)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get seller product error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch product"
        )

@seller_router.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    product_data: ProductUpdate,
    current_seller = Depends(get_current_seller)
):
    """Update a product"""
    try:
        updated_product = seller_service.update_product(
            current_seller.id,
            product_id,
            product_data.dict(exclude_unset=True)
        )
        
        if not updated_product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found or access denied"
            )
        
        # updated_product is already a dict from database service
        return ProductResponse(**updated_product)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Update product error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update product"
        )

@seller_router.delete("/products/{product_id}", response_model=MessageResponse)
async def delete_product(
    product_id: int,
    current_seller = Depends(get_current_seller)
):
    """Delete (deactivate) a product"""
    try:
        success = seller_service.delete_product(current_seller.id, product_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found or access denied"
            )
        
        return MessageResponse(message="Product deleted successfully")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete product error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete product"
        )

# ==================== UTILITY ROUTES ====================

@seller_router.post("/become-seller", response_model=UserResponse)
async def become_seller(current_user = Depends(get_current_user)):
    """Convert regular user to seller"""
    try:
        if current_user.is_seller:
            return UserResponse.from_orm(current_user)
        
        updated_user = seller_service.update_seller_profile(
            current_user.id,
            {"is_seller": True}
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to become seller"
            )
        
        return UserResponse.from_orm(updated_user)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Become seller error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to become seller"
        )
