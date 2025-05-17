"""
User management and authentication routes for PANDA Therapy
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import List, Optional
from datetime import timedelta

from models.user import User, Patient, Therapist, UserCreate, UserUpdate, TokenResponse, UserRole
from database import Database
from simple_auth import get_current_user, create_dev_user
from letta_agent import LettaAgentManager

# Initialize router, database and agent manager
router = APIRouter(tags=["users"])
db = Database()
letta_mgr = LettaAgentManager()

@router.post("/register", response_model=User)
async def register_user(user_data: UserCreate):
    """
    Register a new user in the system
    
    Args:
        user_data: User registration information
        
    Returns:
        Created user object
    """
    # Create the user in the database - simplified for dev testing
    user = create_dev_user(
        username=user_data.username,
        name=user_data.name,
        role=user_data.role.value if hasattr(user_data.role, 'value') else user_data.role
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User could not be created"
        )
    
    # Create a Letta agent for the user
    agent_id = letta_mgr.create_agent_for_user(user)
    if agent_id:
        # Update the user with the agent ID
        db.update_user_letta_agent(user.id, agent_id)
        user.letta_agent_id = agent_id
    
    return user

@router.post("/dev-login", response_model=dict)
async def dev_login(username: str, name: str, role: str = "patient"):
    """
    Simplified development login that creates a user if it doesn't exist
    
    Args:
        username: User's username
        name: User's display name
        role: User's role (patient, therapist, admin)
        
    Returns:
        User ID and other user information
    """
    user = create_dev_user(username, name, role)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create development user"
        )
    
    # Create a Letta agent for the user if they don't have one
    if not getattr(user, 'letta_agent_id', None):
        agent_id = letta_mgr.create_agent_for_user(user)
        if agent_id:
            # Update the user with the agent ID
            db.update_user_letta_agent(user.id, agent_id)
            user.letta_agent_id = agent_id
    
    return {
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "agent_id": getattr(user, "letta_agent_id", None),
        "message": "Use this user_id in X-User-ID header for authentication"
    }

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get the current authenticated user"""
    return current_user

@router.put("/users/me", response_model=User)
async def update_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user)
):
    """
    Update the current user's information
    
    Args:
        user_update: Fields to update
        current_user: Current authenticated user
        
    Returns:
        Updated user information
    """
    # Update user implementation would go here
    # For now, this is a placeholder
    return current_user

@router.get("/users", response_model=List[User])
async def get_all_users(
    role: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get all users (admin only)
    
    Args:
        role: Optional role to filter by
        current_user: Current authenticated user
        
    Returns:
        List of users
    """
    # Check if the current user has admin role
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view all users"
        )
    
    # Convert role string to enum if provided
    user_role = None
    if role:
        try:
            user_role = UserRole(role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {role}"
            )
    
    # Get all users
    users = db.get_all_users(role=user_role)
    return users

@router.post("/therapist/{therapist_id}/patients/{patient_id}", response_model=Patient)
async def assign_patient_to_therapist(
    therapist_id: str, 
    patient_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Assign a patient to a therapist
    
    Args:
        therapist_id: ID of the therapist
        patient_id: ID of the patient
        current_user: Current authenticated user
        
    Returns:
        Updated patient information
    """
    # Check if the current user is an admin or the therapist
    if current_user.role != UserRole.ADMIN and current_user.id != therapist_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to assign patients"
        )
    
    # Implementation for assigning patient to therapist would go here
    # For now, this is a placeholder
    return Patient(
        id=patient_id,
        username="patient",
        email="patient@example.com",
        name="Patient Name",
        role=UserRole.PATIENT,
        therapist_id=therapist_id
    )
