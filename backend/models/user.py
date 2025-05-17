"""
User models for the PANDA Therapy system
"""
from pydantic import BaseModel, Field, EmailStr
from enum import Enum
from typing import Optional, Dict, Any, List
import os
import uuid
from datetime import datetime

class UserRole(str, Enum):
    """User roles in the system"""
    PATIENT = "patient"
    THERAPIST = "therapist"
    ADMIN = "admin"

class User(BaseModel):
    """Base user model for all system users"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    name: str
    role: UserRole
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Letta agent information
    letta_agent_id: Optional[str] = None
    
    def get_storage_path(self, base_dir: str = "./user_data") -> str:
        """Get the user-specific storage path"""
        path = os.path.join(base_dir, self.id)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_transcripts_path(self) -> str:
        """Get the user's transcripts directory"""
        path = os.path.join(self.get_storage_path(), "transcripts")
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_audio_path(self) -> str:
        """Get the user's audio files directory"""
        path = os.path.join(self.get_storage_path(), "audio")
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_storage_context_path(self) -> str:
        """Get the path for the user's vector store storage context"""
        path = os.path.join(self.get_storage_path(), "storage")
        os.makedirs(path, exist_ok=True)
        return path

class Patient(User):
    """Patient-specific user model"""
    role: UserRole = UserRole.PATIENT
    therapist_id: Optional[str] = None
    therapy_sessions: List[str] = Field(default_factory=list)
    notes: Dict[str, Any] = Field(default_factory=dict)
    
class Therapist(User):
    """Therapist-specific user model"""
    role: UserRole = UserRole.THERAPIST
    patients: List[str] = Field(default_factory=list)
    specializations: List[str] = Field(default_factory=list)

# User request models for API
class UserCreate(BaseModel):
    """Model for creating a user"""
    username: str
    email: str
    name: str
    password: str  # Will be hashed before storage
    role: UserRole
    
class UserLogin(BaseModel):
    """Model for user login"""
    username: str
    password: str
    
class UserUpdate(BaseModel):
    """Model for updating user information"""
    name: Optional[str] = None
    email: Optional[str] = None
    
class TokenResponse(BaseModel):
    """Model for authentication token response"""
    access_token: str
    token_type: str = "bearer"
