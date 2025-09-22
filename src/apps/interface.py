"""Pydantic models for API request/response validation"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

# User/Authentication Models
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    telegram_id: Optional[int] = None
    brand_name: Optional[str] = Field(None, max_length=255)
    phone_number: Optional[str] = Field(None, max_length=20)
    address: Optional[str] = None
    is_seller: bool = False

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    brand_name: Optional[str] = Field(None, max_length=255)
    phone_number: Optional[str] = Field(None, max_length=20)
    address: Optional[str] = None
    is_seller: Optional[bool] = None

class UserResponse(BaseModel):
    id: int
    name: str
    telegram_id: Optional[int]
    brand_name: Optional[str]
    phone_number: Optional[str]
    address: Optional[str]
    is_seller: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Product Models
class ProductCreate(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=500)
    price: int = Field(..., ge=0)  # Price in smallest currency unit
    description: Optional[str] = None
    cloud_image_url: Optional[str] = Field(None, max_length=1000)
    specifications: Optional[Dict[str, str]] = None

class ProductUpdate(BaseModel):
    product_name: Optional[str] = Field(None, min_length=1, max_length=500)
    price: Optional[int] = Field(None, ge=0)
    description: Optional[str] = None
    cloud_image_url: Optional[str] = Field(None, max_length=1000)
    is_active: Optional[bool] = None
    specifications: Optional[Dict[str, str]] = None

class ProductResponse(BaseModel):
    id: int
    user_id: int
    product_name: str
    price: int
    description: Optional[str]
    local_image_path: Optional[str]
    cloud_image_url: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    specifications: Dict[str, str] = {}
    
    class Config:
        from_attributes = True

class ProductListResponse(BaseModel):
    products: List[ProductResponse]
    total: int
    page: int
    per_page: int

# Authentication Models
class TelegramLoginRequest(BaseModel):
    telegram_id: int

class PhoneLoginRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=15)

class PhoneSignupRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=15)
    name: str = Field(..., min_length=1, max_length=255)
    user_type: str = Field(..., pattern="^(buyer|seller)$")  # "buyer" or "seller"

class EmailLoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    remember_me: bool = False

class EmailSignupRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    name: str = Field(..., min_length=1, max_length=255)
    user_type: str = Field(..., pattern="^(buyer|seller)$")

class LoginResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: str = "bearer"

class SignupResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: str = "bearer"
    message: str = "Account created successfully"
    requires_onboarding: bool = False

# OTP Verification Models
class SendOTPRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=15)

class VerifyOTPRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=15)
    otp: str = Field(..., min_length=4, max_length=6)

class OTPResponse(BaseModel):
    message: str
    otp_sent: bool = True

# Seller Onboarding Models
class SellerOnboardingRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = Field(None, min_length=5, max_length=255)
    brand_name: Optional[str] = Field(None, max_length=255)
    contact_info: Optional[str] = Field(None, max_length=500)  # "How to reach you"

class OnboardingCompleteResponse(BaseModel):
    user: UserResponse
    message: str = "Onboarding completed successfully"
    dashboard_ready: bool = True

# Search Models
class ProductSearchRequest(BaseModel):
    search_term: Optional[str] = None
    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)

# Buyer Query Models
class BuyerQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The buyer's query or question")
    
class BuyerQueryResponse(BaseModel):
    type: str = Field(..., description="Response type: products, suggestions, clarification, support, error")
    message: str = Field(..., description="Main response message")
    products: Optional[List[ProductResponse]] = Field(None, description="Matching products if any")
    related_products: Optional[List[ProductResponse]] = Field(None, description="Related products")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions for unclear queries")
    follow_up_questions: Optional[List[str]] = Field(None, description="Follow-up questions for clarification")
    support_options: Optional[List[str]] = Field(None, description="Support options for negative sentiment")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    workflow_path: List[str] = Field(..., description="Path taken through the workflow")
    query_metadata: Optional[Dict] = Field(None, description="Metadata about query processing")

# Standard Response Models
class MessageResponse(BaseModel):
    message: str
    status: str = "success"

class ErrorResponse(BaseModel):
    message: str
    status: str = "error"
    details: Optional[str] = None
