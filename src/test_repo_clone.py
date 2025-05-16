import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.git_utils import clone_repo, does_repo_exist

def main():
    parser = argparse.ArgumentParser(description="Test repository cloning")
    parser.add_argument("--owner", type=str, required=True, help="Repository owner/organization")
    parser.add_argument("--repo", type=str, required=True, help="Repository name")
    parser.add_argument("--dest", type=str, help="Destination directory (optional)")
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/repos")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Default destination is in the data directory
    dest = args.dest if args.dest else str(data_dir / f"{args.owner}_{args.repo}")
    
    # Check if repository exists
    print(f"Checking if repository {args.owner}/{args.repo} exists...")
    exists = does_repo_exist(args.repo, args.owner)
    
    if not exists:
        print(f"Repository {args.owner}/{args.repo} does not exist!")
        return
    
    # Clone repository
    result = clone_repo(args.repo, dest, args.owner)
    
    if result:
        print(f"Successfully cloned to {result}")
        
        # Print repository info
        try:
            repo_size = subprocess.run(
                f"du -sh {result}",
                shell=True,
                check=True,
                capture_output=True,
                text=True
            ).stdout.strip()
            print(f"Repository size: {repo_size}")
            
            commit_count = subprocess.run(
                f"cd {result} && git rev-list --count HEAD",
                shell=True, 
                check=True,
                capture_output=True,
                text=True
            ).stdout.strip()
            print(f"Commit count: {commit_count}")
        except Exception as e:
            print(f"Error getting repository info: {e}")
    else:
        print("Failed to clone repository")

if __name__ == "__main__":
    main()