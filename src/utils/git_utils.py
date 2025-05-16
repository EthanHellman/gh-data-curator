import os
import subprocess
import hashlib
import random
import string
from pathlib import Path

def clone_repo(repo: str, dest: str | None = None, org: str = "owner") -> str | None:
    """Clone a repository from GitHub."""
    if not os.path.exists(dest or repo):
        clone_cmd = (
            f"git clone git@github.com:{org}/{repo}.git"
            if dest is None
            else f"git clone git@github.com:{org}/{repo}.git {dest}"
        )
        subprocess.run(
            clone_cmd,
            check=True,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return repo if dest is None else dest
    return None

def does_repo_exist(repo: str, org: str = "owner") -> bool:
    """Check if a repository exists in project organization."""
    try:
        from ghapi.all import GhApi
        import os
        
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        api = GhApi(token=GITHUB_TOKEN)
        org_repos = [
            x["name"]
            for page in range(1, 3)
            for x in api.repos.list_for_org(org, per_page=100, page=page)
        ]
        return repo in org_repos
    except Exception:
        # Fallback: Try to check via clone
        temp_dir = f"temp_check_{generate_hash(repo)}"
        try:
            subprocess.run(
                f"git clone --depth 1 https://github.com/{org}/{repo}.git {temp_dir}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            exists = True
        except subprocess.CalledProcessError:
            exists = False
        finally:
            if os.path.exists(temp_dir):
                subprocess.run(f"rm -rf {temp_dir}", shell=True)
        return exists

def generate_hash(s):
    """Generate a short hash from a string."""
    return "".join(
        random.Random(int(hashlib.sha256(s.encode()).hexdigest(), 16)).choices(
            string.ascii_lowercase + string.digits, k=8
        )
    )