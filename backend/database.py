"""
Database module for the PANDA Therapy application

This module provides a simple JSON-based database for storing user information.
For a production application, this would be replaced with a proper database system.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Type, TypeVar
from datetime import datetime
import bcrypt
from models.user import User, Patient, Therapist, UserRole

logger = logging.getLogger(__name__)

# Type variable for generic functions that return a specific model type
T = TypeVar('T', bound=User)

class Database:
    """Simple JSON-based database for the PANDA Therapy application"""
    
    def __init__(self, db_path: str = "./database"):
        """Initialize the database with the given path"""
        self.db_path = db_path
        self.users_file = os.path.join(db_path, "users.json")
        self.patients_file = os.path.join(db_path, "patients.json")
        self.therapists_file = os.path.join(db_path, "therapists.json")
        self.sessions_file = os.path.join(db_path, "therapy_sessions.json")
        self.transcripts_file = os.path.join(db_path, "transcripts.json")
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize database files if they don't exist
        self._ensure_file_exists(self.users_file, {})
        self._ensure_file_exists(self.patients_file, {})
        self._ensure_file_exists(self.therapists_file, {})
        self._ensure_file_exists(self.sessions_file, {})
        self._ensure_file_exists(self.transcripts_file, {})
    
    def _ensure_file_exists(self, file_path: str, default_content: Any) -> None:
        """Ensure that the given file exists, creating it with default content if not"""
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(default_content, f, indent=2)
    
    def _load_data(self, file_path: str) -> Dict[str, Any]:
        """Load data from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return {}
    
    def _save_data(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Save data to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serial)
            return True
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            return False
    
    def _json_serial(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            # This handles Pydantic models
            return obj.model_dump()
        # Handle dict-like objects
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Type {type(obj)} not serializable")
    
    # User management methods
    
    def create_user(self, username: str, email: str, name: str, password: str, 
                   role: UserRole, **kwargs) -> Optional[User]:
        """
        Create a new user with the given information
        
        Args:
            username: Unique username for the new user
            email: User's email address
            name: User's display name
            password: User's password (will be hashed)
            role: User's role in the system
        
        Returns:
            The created user object, or None if creation failed
        """
        # Check if username already exists
        users = self._load_data(self.users_file)
        for user_id, user_data in users.items():
            if user_data.get("username") == username:
                logger.warning(f"Username {username} already exists")
                return None
        
        # Hash the password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode(), salt).decode()
        
        # Create the appropriate user type
        user_kwargs = {
            "username": username,
            "email": email,
            "name": name,
            "role": role,
            **kwargs
        }
        
        if role == UserRole.PATIENT:
            user = Patient(**user_kwargs)
            specific_file = self.patients_file
        elif role == UserRole.THERAPIST:
            user = Therapist(**user_kwargs)
            specific_file = self.therapists_file
        else:
            user = User(**user_kwargs)
            specific_file = None
        
        # Save the user to the users file
        users = self._load_data(self.users_file)
        user_data = user.model_dump()
        user_data["password"] = hashed_password
        users[user.id] = user_data
        success = self._save_data(self.users_file, users)
        
        # If applicable, save to the role-specific file as well
        if specific_file and success:
            specific_data = self._load_data(specific_file)
            specific_data[user.id] = user_data
            success = self._save_data(specific_file, specific_data)
        
        return user if success else None
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID"""
        users = self._load_data(self.users_file)
        user_data = users.get(user_id)
        
        if not user_data:
            return None
        
        # Remove password from the data
        if "password" in user_data:
            del user_data["password"]
        
        # Create the appropriate user type
        role = user_data.get("role")
        if role == UserRole.PATIENT.value:
            return Patient(**user_data)
        elif role == UserRole.THERAPIST.value:
            return Therapist(**user_data)
        else:
            return User(**user_data)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username"""
        users = self._load_data(self.users_file)
        
        for user_id, user_data in users.items():
            if user_data.get("username") == username:
                # Remove password from the data
                if "password" in user_data:
                    del user_data["password"]
                
                # Create the appropriate user type
                role = user_data.get("role")
                if role == UserRole.PATIENT.value:
                    return Patient(**user_data)
                elif role == UserRole.THERAPIST.value:
                    return Therapist(**user_data)
                else:
                    return User(**user_data)
        
        return None
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verify a user's password"""
        users = self._load_data(self.users_file)
        
        for user_id, user_data in users.items():
            if user_data.get("username") == username:
                stored_password = user_data.get("password", "")
                return bcrypt.checkpw(password.encode(), stored_password.encode())
        
        return False
    
    def update_user_letta_agent(self, user_id: str, agent_id: str) -> bool:
        """Update a user's Letta agent ID"""
        users = self._load_data(self.users_file)
        
        if user_id not in users:
            logger.warning(f"Cannot update Letta agent ID: User {user_id} not found")
            return False
        
        # Update in the main users file
        users[user_id]["letta_agent_id"] = agent_id
        users[user_id]["updated_at"] = datetime.now().isoformat()
        success = self._save_data(self.users_file, users)
        
        # Update in the role-specific file if applicable
        role = users[user_id].get("role")
        if role == UserRole.PATIENT.value:
            patients = self._load_data(self.patients_file)
            if user_id in patients:
                patients[user_id]["letta_agent_id"] = agent_id
                patients[user_id]["updated_at"] = datetime.now().isoformat()
                success = success and self._save_data(self.patients_file, patients)
        elif role == UserRole.THERAPIST.value:
            therapists = self._load_data(self.therapists_file)
            if user_id in therapists:
                therapists[user_id]["letta_agent_id"] = agent_id
                therapists[user_id]["updated_at"] = datetime.now().isoformat()
                success = success and self._save_data(self.therapists_file, therapists)
        
        logger.info(f"Updated Letta agent ID for user {user_id}: {agent_id}")
        return success
    
    def get_all_users(self, role: Optional[UserRole] = None) -> List[User]:
        """Get all users, optionally filtered by role"""
        users = self._load_data(self.users_file)
        result = []
        
        for user_id, user_data in users.items():
            if role and user_data.get("role") != role.value:
                continue
            
            # Remove password from the data
            if "password" in user_data:
                user_data = {k: v for k, v in user_data.items() if k != "password"}
            
            # Create the appropriate user type
            user_role = user_data.get("role")
            if user_role == UserRole.PATIENT.value:
                result.append(Patient(**user_data))
            elif user_role == UserRole.THERAPIST.value:
                result.append(Therapist(**user_data))
            else:
                result.append(User(**user_data))
        
        return result
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user by ID"""
        users = self._load_data(self.users_file)
        
        if user_id not in users:
            return False
        
        # Delete the user data
        role = users[user_id].get("role")
        del users[user_id]
        success = self._save_data(self.users_file, users)
        
        # Delete from role-specific file if applicable
        if role == UserRole.PATIENT.value:
            patients = self._load_data(self.patients_file)
            if user_id in patients:
                del patients[user_id]
                success = success and self._save_data(self.patients_file, patients)
        elif role == UserRole.THERAPIST.value:
            therapists = self._load_data(self.therapists_file)
            if user_id in therapists:
                del therapists[user_id]
                success = success and self._save_data(self.therapists_file, therapists)
        
        # Delete user's therapy sessions and transcripts
        sessions = self._load_data(self.sessions_file)
        user_sessions = [s_id for s_id, s in sessions.items() if s.get("user_id") == user_id]
        for session_id in user_sessions:
            del sessions[session_id]
        success = success and self._save_data(self.sessions_file, sessions)
        
        # Delete transcripts
        transcripts = self._load_data(self.transcripts_file)
        user_transcripts = [t_id for t_id, t in transcripts.items() if t.get("user_id") == user_id]
        for transcript_id in user_transcripts:
            del transcripts[transcript_id]
        success = success and self._save_data(self.transcripts_file, transcripts)
        
        return success
    
    def create_therapy_session(self, user_id: str, session_data: Dict[str, Any]) -> Optional[str]:
        """Create a new therapy session"""
        sessions = self._load_data(self.sessions_file)
        session_id = str(len(sessions) + 1)
        session_data["id"] = session_id
        session_data["user_id"] = user_id
        session_data["created_at"] = datetime.now().isoformat()
        sessions[session_id] = session_data
        success = self._save_data(self.sessions_file, sessions)
        return session_id if success else None
    
    def get_therapy_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a therapy session by ID"""
        sessions = self._load_data(self.sessions_file)
        return sessions.get(session_id)
    
    def update_therapy_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Update a therapy session"""
        sessions = self._load_data(self.sessions_file)
        if session_id not in sessions:
            return False
        sessions[session_id].update(session_data)
        sessions[session_id]["updated_at"] = datetime.now().isoformat()
        return self._save_data(self.sessions_file, sessions)
    
    def delete_therapy_session(self, session_id: str) -> bool:
        """Delete a therapy session by ID"""
        sessions = self._load_data(self.sessions_file)
        if session_id not in sessions:
            return False
        del sessions[session_id]
        return self._save_data(self.sessions_file, sessions)
    
    def create_transcript(self, user_id: str, transcript_data: Dict[str, Any]) -> str:
        """Create a new transcript for a therapy session"""
        transcripts = self._load_data(self.transcripts_file)
        
        # Generate a transcript ID if needed
        transcript_id = transcript_data.get("id", str(len(transcripts) + 1))
        
        # Add creation timestamp
        transcript_data["created_at"] = datetime.now().isoformat()
        
        # Save to database
        transcripts[transcript_id] = transcript_data
        self._save_data(self.transcripts_file, transcripts)
        
        logger.info(f"Created transcript {transcript_id} for session {user_id}")
        return transcript_id
        
    def create_session_analysis(self, session_id: str, analysis_data: Dict[str, Any]) -> str:
        """Create a new session analysis record"""
        # Create the analysis file if it doesn't exist
        analysis_file = os.path.join(self.db_path, "session_analysis.json")
        analyses = self._load_data(analysis_file)
        
        # Generate an analysis ID
        analysis_id = analysis_data.get("id", str(len(analyses) + 1))
        
        # Add session ID and timestamp
        analysis_data["session_id"] = session_id
        analysis_data["created_at"] = datetime.now().isoformat()
        
        # Save to database
        analyses[analysis_id] = analysis_data
        self._save_data(analysis_file, analyses)
        
        logger.info(f"Created analysis {analysis_id} for session {session_id}")
        return analysis_id
    
    def get_transcript(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """Get a transcript by ID"""
        transcripts = self._load_data(self.transcripts_file)
        return transcripts.get(transcript_id)
    
    def update_transcript(self, transcript_id: str, transcript_data: Dict[str, Any]) -> bool:
        """Update a transcript"""
        transcripts = self._load_data(self.transcripts_file)
        if transcript_id not in transcripts:
            return False
        transcripts[transcript_id].update(transcript_data)
        transcripts[transcript_id]["updated_at"] = datetime.now().isoformat()
        return self._save_data(self.transcripts_file, transcripts)
    
    def delete_transcript(self, transcript_id: str) -> bool:
        """Delete a transcript by ID"""
        transcripts = self._load_data(self.transcripts_file)
        if transcript_id not in transcripts:
            return False
        del transcripts[transcript_id]
        return self._save_data(self.transcripts_file, transcripts)
