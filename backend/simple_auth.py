"""
Simple authentication module for PANDA Therapy development

This module provides basic user identification without complex JWT/Auth0 systems.
"""

from typing import Optional, Dict
from fastapi import Header, HTTPException, status
from models.user import User
from database import Database

# Database instance
db = Database()

async def get_current_user(x_user_id: Optional[str] = Header(None)) -> User:
    """
    Get the current user based on a simple header
    
    This is a simplified auth system for development only.
    For production, use a proper authentication system.
    
    Args:
        x_user_id: User ID passed in the X-User-ID header
        
    Returns:
        The User object
    """
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide X-User-ID header.",
        )
    
    # Get the user from the database
    user = db.get_user(x_user_id)
    
    # If user doesn't exist, raise exception
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

def create_dev_user(username: str, name: str, role: str) -> User:
    """
    Create a development user without password verification
    
    Args:
        username: User's username
        name: User's display name
        role: User's role (patient, therapist, admin)
        
    Returns:
        The created User object
    """
    # Check if user already exists
    existing_user = db.get_user_by_username(username)
    if existing_user:
        return existing_user
        
    # Create user with minimal validation
    user_data = {
        "username": username,
        "email": f"{username}@example.com",  # Placeholder email
        "name": name,
        "role": role,
        "password": "dev_password"  # No real password needed
    }
    
    # Additional fields for specific roles
    if role == "patient":
        user_data["medical_history"] = "Development account"
        user_data["therapy_goals"] = "Testing the PANDA system"
        
    elif role == "therapist":
        user_data["specialization"] = "Test Therapist"
        user_data["license_number"] = "DEV-12345"
        
    # Create the user
    user_id = db.create_user(user_data)
    return db.get_user(user_id)
