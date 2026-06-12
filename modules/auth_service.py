import sqlite3
import hashlib
import os
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

DB_PATH = "data/users.db"

def initialize_db() -> None:
    """Creates the users table in the SQLite database if it doesn't already exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()
        logger.info("SQLite authentication database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        raise

def _hash_password(password: str, salt: str) -> str:
    """Hashes a password with a given salt using SHA-256."""
    hash_input = (password + salt).encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()

def register_user(username: str, password: str) -> Tuple[bool, str]:
    """
    Registers a new user in the SQLite database.
    
    Args:
        username: Desired username string.
        password: Plaintext password.
        
    Returns:
        Tuple of (Success (bool), Message (str))
    """
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty."
        
    try:
        initialize_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username is already taken."
            
        # Generate salt and hash password
        salt = os.urandom(16).hex()
        password_hash = _hash_password(password, salt)
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, password_hash, salt)
        )
        conn.commit()
        conn.close()
        logger.info(f"User '{username}' registered successfully.")
        return True, "Registration successful."
    except Exception as e:
        logger.error(f"Failed to register user: {str(e)}", exc_info=True)
        return False, f"Database error during registration: {str(e)}"

def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticates a user against stored SQLite records.
    
    Args:
        username: Input username.
        password: Input password.
        
    Returns:
        True if username/password match, False otherwise.
    """
    username = username.strip().lower()
    if not username or not password:
        return False
        
    try:
        initialize_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return False
            
        stored_hash, salt = row
        input_hash = _hash_password(password, salt)
        return input_hash == stored_hash
    except Exception as e:
        logger.error(f"Failed to authenticate user: {str(e)}", exc_info=True)
        return False
