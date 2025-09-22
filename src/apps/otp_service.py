"""OTP service for phone number verification"""
from typing import Optional, Dict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OTPService:
    """Service for OTP generation and verification"""
    
    def __init__(self):
        # In-memory storage for OTPs (in production, use Redis or database)
        self.otp_storage: Dict[str, Dict] = {}
        self.hardcoded_otp = "123456"  # Hardcoded OTP for development
        self.otp_expiry_minutes = 5
    
    def send_otp(self, phone_number: str) -> bool:
        """Send OTP to phone number (hardcoded for now)"""
        logger.info(f"📱 Sending OTP to phone: {phone_number}")
        
        # Generate OTP (using hardcoded for now)
        otp = self.hardcoded_otp
        expiry = datetime.utcnow() + timedelta(minutes=self.otp_expiry_minutes)
        
        # Store OTP with expiry
        self.otp_storage[phone_number] = {
            "otp": otp,
            "expiry": expiry,
            "attempts": 0
        }
        
        logger.info(f"🔢 OTP sent to {phone_number}: {otp} (hardcoded)")
        return True
    
    def verify_otp(self, phone_number: str, provided_otp: str) -> bool:
        """Verify OTP for phone number"""
        logger.info(f"🔍 Verifying OTP for phone: {phone_number}")
        
        if phone_number not in self.otp_storage:
            logger.warning(f"⚠️ No OTP found for phone: {phone_number}")
            return False
        
        otp_data = self.otp_storage[phone_number]
        
        # Check if OTP expired
        if datetime.utcnow() > otp_data["expiry"]:
            logger.warning(f"⏰ OTP expired for phone: {phone_number}")
            del self.otp_storage[phone_number]
            return False
        
        # Check attempts limit
        if otp_data["attempts"] >= 3:
            logger.warning(f"🚫 Too many attempts for phone: {phone_number}")
            del self.otp_storage[phone_number]
            return False
        
        # Verify OTP
        if otp_data["otp"] == provided_otp:
            logger.info(f"✅ OTP verified successfully for phone: {phone_number}")
            del self.otp_storage[phone_number]  # Remove after successful verification
            return True
        else:
            # Increment attempt counter
            otp_data["attempts"] += 1
            logger.warning(f"❌ Invalid OTP for phone: {phone_number}. Attempts: {otp_data['attempts']}")
            return False
    
    def resend_otp(self, phone_number: str) -> bool:
        """Resend OTP to phone number"""
        logger.info(f"🔄 Resending OTP to phone: {phone_number}")
        
        # Remove existing OTP and send new one
        if phone_number in self.otp_storage:
            del self.otp_storage[phone_number]
        
        return self.send_otp(phone_number)
    
    def get_hardcoded_otp(self) -> str:
        """Get hardcoded OTP for development/testing"""
        return self.hardcoded_otp
