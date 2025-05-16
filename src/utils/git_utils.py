import os
import subprocess
import hashlib
import random
import string
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def clone_repo(repo: str, dest: str | None = None, org: str = None) -> str | None:
    """
    Clone a repository from GitHub.
    
    Args:
        repo: Repository name (e.g. "django" or "django/django")
        dest: Destination directory (if None, uses repo name)
        org: Organization name (if None, assumes repo already contains org)
    
    Returns:
        Path to cloned repository or None if it already exists
    """
    # Handle repo format
    full_repo = repo
    if org is not None:
        full_repo = f"{org}/{repo}"
    
    # Create destination path if not provided
    if dest is None:
        dest = repo.split("/")[-1]
    
    # Check if repo already exists
    if os.path.exists(dest):
        logger.info(f"Repository already exists at {dest}")
        return None
    
    # Clone the repository
    logger.info(f"Cloning {full_repo} to {dest}...")
    try:
        result = subprocess.run(
            f"git clone https://github.com/{full_repo}.git {dest}",
            check=True,
            shell=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully cloned {full_repo} to {dest}")
        return dest
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cloning repository: {e.stderr}")
        return None

def does_repo_exist(repo: str, org: str = None) -> bool:
    """
    Check if a repository exists on GitHub.
    
    Args:
        repo: Repository name
        org: Organization name (if None, assumes repo already contains org)
    
    Returns:
        True if repository exists, False otherwise
    """
    full_repo = repo
    if org is not None:
        full_repo = f"{org}/{repo}"
        
    try:
        result = subprocess.run(
            f"git ls-remote https://github.com/{full_repo}.git",
            shell=True,
            check=False,
            capture_output=True,
            text=True
        )
        exists = result.returncode == 0
        
        if exists:
            logger.info(f"Repository {full_repo} exists")
        else:
            logger.warning(f"Repository {full_repo} does not exist: {result.stderr}")
            
        return exists
    except Exception as e:
        logger.error(f"Error checking repository: {e}")
        return False

def generate_hash(s: str) -> str:
    """Generate a short hash from a string."""
    return "".join(
        random.Random(int(hashlib.sha256(s.encode()).hexdigest(), 16)).choices(
            string.ascii_lowercase + string.digits, k=8
        )
    )