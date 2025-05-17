import argparse
import sys
import os
import subprocess
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.utils.git_utils import clone_repo, does_repo_exist

def main():
    parser = argparse.ArgumentParser(description="Test repository cloning")
    parser.add_argument("--owner", type=str, required=True, help="Repository owner/organization")
    parser.add_argument("--repo", type=str, required=True, help="Repository name")
    parser.add_argument("--dest", type=str, help="Destination directory (optional)")
    parser.add_argument("--force", action="store_true", help="Force clone even if repository exists")
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/repos")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Default destination is in the data directory
    dest = args.dest if args.dest else str(data_dir / f"{args.owner}_{args.repo}")
    
    # Check if repository exists on GitHub
    logger.info(f"Checking if repository {args.owner}/{args.repo} exists on GitHub...")
    exists = does_repo_exist(args.repo, args.owner)
    
    if not exists:
        logger.error(f"Repository {args.owner}/{args.repo} does not exist on GitHub!")
        return
    
    # Check if destination already exists
    if os.path.exists(dest):
        if args.force:
            logger.warning(f"Destination {dest} already exists. Removing...")
            try:
                subprocess.run(f"rm -rf {dest}", shell=True, check=True)
                logger.info(f"Removed existing directory {dest}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to remove directory {dest}: {e}")
                return
        else:
            logger.info(f"Repository already exists at {dest}. Use --force to re-clone.")
            # Even though we're not cloning, print the info of the existing repo
            try:
                get_repo_info(dest)
            except Exception as e:
                logger.error(f"Error getting repository info: {e}")
            return
    
    # Clone repository
    result = clone_repo(args.repo, dest, args.owner)
    
    if result:
        logger.info(f"Successfully cloned to {result}")
        
        # Print repository info
        try:
            get_repo_info(result)
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
    else:
        logger.error("Failed to clone repository")

def get_repo_info(repo_path):
    """Get and display information about a repository."""
    repo_size = subprocess.run(
        f"du -sh {repo_path}",
        shell=True,
        check=True,
        capture_output=True,
        text=True
    ).stdout.strip()
    logger.info(f"Repository size: {repo_size}")
    
    # Move to the repository directory to run git commands
    current_dir = os.getcwd()
    os.chdir(repo_path)
    try:
        commit_count = subprocess.run(
            "git rev-list --count HEAD",
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        ).stdout.strip()
        logger.info(f"Commit count: {commit_count}")
        
        latest_commit = subprocess.run(
            "git log -1 --pretty=format:'%h - %an: %s'",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        ).stdout.strip()
        logger.info(f"Latest commit: {latest_commit}")
        
        branch = subprocess.run(
            "git branch --show-current",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        ).stdout.strip()
        logger.info(f"Current branch: {branch}")
    finally:
        # Return to the original directory
        os.chdir(current_dir)

if __name__ == "__main__":
    main()