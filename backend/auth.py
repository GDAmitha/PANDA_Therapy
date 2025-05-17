"""
Authentication module for the PANDA Therapy application

This module handles JWT token generation, validation, and user authentication.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from database import Database

# Authentication settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "REPLACE_WITH_SECRET_KEY_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# OAuth2 scheme for token handling
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database instance
db = Database()

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        JWT access token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Validate the access token and return the current user
    
    Args:
        token: JWT token from the request
        
    Returns:
        The authenticated User object
        
    Raises:
        HTTPException: If the token is invalid or the user doesn't exist
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    # Get the user from the database
    user = db.get_user(user_id)
    if user is None:
        raise credentials_exception
    
    return user

async def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user with username and password
    
    Args:
        username: User's username
        password: User's password
        
    Returns:
        User object if authentication successful, None otherwise
    """
    # Verify the password
    if not db.verify_password(username, password):
        return None
    
    # Get the user
    user = db.get_user_by_username(username)
    return user
