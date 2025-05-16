import json
import logging
import os
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class PRCollector:
    """Collect pull requests from GitHub repositories."""
    
    def __init__(self, token: str, data_dir: Path):
        self.token = token
        self.data_dir = data_dir
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
    
    def get_pull_requests(self, owner: str, repo: str, state: str = "closed", 
                         per_page: int = 100, max_pages: int = 10) -> List[Dict]:
        """Fetch pull requests for a repository."""
        prs = []
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        
        for page in range(1, max_pages + 1):
            logger.info(f"Fetching page {page} of {owner}/{repo} PRs")
            params = {
                "state": state,
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "direction": "desc"
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                page_prs = response.json()
                if not page_prs:
                    break
                prs.extend(page_prs)
                logger.info(f"Retrieved {len(page_prs)} PRs from page {page}")
            else:
                logger.error(f"Failed to fetch PRs: {response.status_code} - {response.text}")
                break
                
            # Check for rate limiting
            if "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                if remaining < 10:
                    reset_time = int(response.headers["X-RateLimit-Reset"])
                    current_time = datetime.now().timestamp()
                    sleep_time = max(0, reset_time - current_time) + 1
                    logger.warning(f"Rate limit almost exceeded. Sleeping for {sleep_time}s")
                    time.sleep(sleep_time)
        
        return prs
    
    def get_pr_details(self, owner: str, repo: str, pr_number: int) -> Optional[Dict]:
        """Get detailed information about a specific PR."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch PR details: {response.status_code} - {response.text}")
            return None
    
    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """Get files changed in a PR."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch PR files: {response.status_code} - {response.text}")
            return []
    
    def get_pr_comments(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """Get comments on a PR."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch PR comments: {response.status_code} - {response.text}")
            return []
    
    def get_linked_issues(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """Get issues linked to a PR."""
        linked_issues = []
        
        # Get PR details to check for linked issues in the body
        pr_details = self.get_pr_details(owner, repo, pr_number)
        if not pr_details:
            return linked_issues
        
        # Check PR body for issue references (e.g., "Fixes #123", "Closes #456")
        body = pr_details.get("body", "")
        if body:
            # Look for common keywords that link PRs to issues
            keywords = ["close", "closes", "closed", "fix", "fixes", "fixed", 
                       "resolve", "resolves", "resolved"]
            
            for keyword in keywords:
                pattern = f"{keyword} #(\\d+)"
                import re
                matches = re.findall(pattern, body, re.IGNORECASE)
                
                for issue_number in matches:
                    issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
                    response = requests.get(issue_url, headers=self.headers)
                    
                    if response.status_code == 200:
                        linked_issues.append(response.json())
        
        return linked_issues
    
    def collect_and_save(self, owner: str, repo: str, limit: int = 30, 
                        filter_merged: bool = True) -> Path:
        """Collect PRs and save them to disk."""
        output_dir = self.data_dir / "raw" / f"{owner}_{repo}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get basic PR information
        all_prs = self.get_pull_requests(owner, repo)
        logger.info(f"Retrieved {len(all_prs)} total PRs for {owner}/{repo}")
        
        # Filter for merged PRs if requested
        if filter_merged:
            filtered_prs = [pr for pr in all_prs if pr.get("merged_at")]
            logger.info(f"Filtered to {len(filtered_prs)} merged PRs")
        else:
            filtered_prs = all_prs
        
        # Apply limit
        prs_to_process = filtered_prs[:limit]
        logger.info(f"Processing {len(prs_to_process)} PRs (limit: {limit})")
        
        processed_prs = []
        
        for pr in prs_to_process:
            pr_number = pr["number"]
            pr_dir = output_dir / f"pr_{pr_number}"
            pr_dir.mkdir(exist_ok=True)
            
            logger.info(f"Processing PR #{pr_number}: {pr['title']}")
            
            # Save basic PR info
            with open(pr_dir / "basic_info.json", "w") as f:
                json.dump(pr, f, indent=2)
            
            # Get and save detailed information
            pr_details = self.get_pr_details(owner, repo, pr_number)
            if pr_details:
                with open(pr_dir / "details.json", "w") as f:
                    json.dump(pr_details, f, indent=2)
            
            # Get and save files
            pr_files = self.get_pr_files(owner, repo, pr_number)
            if pr_files:
                with open(pr_dir / "files.json", "w") as f:
                    json.dump(pr_files, f, indent=2)
            
            # Get and save comments
            pr_comments = self.get_pr_comments(owner, repo, pr_number)
            if pr_comments:
                with open(pr_dir / "comments.json", "w") as f:
                    json.dump(pr_comments, f, indent=2)
            
            # Get and save linked issues
            linked_issues = self.get_linked_issues(owner, repo, pr_number)
            if linked_issues:
                with open(pr_dir / "linked_issues.json", "w") as f:
                    json.dump(linked_issues, f, indent=2)
            
            # Create a summary file with the most important information
            summary = {
                "pr_number": pr_number,
                "title": pr.get("title"),
                "author": pr.get("user", {}).get("login"),
                "created_at": pr.get("created_at"),
                "merged_at": pr.get("merged_at"),
                "body": pr_details.get("body") if pr_details else None,
                "files_changed": len(pr_files),
                "comments": len(pr_comments),
                "linked_issues": [issue.get("number") for issue in linked_issues],
                "file_paths": [f.get("filename") for f in pr_files],
            }
            
            with open(pr_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            processed_prs.append(summary)
        
        # Create an index file for all processed PRs
        with open(output_dir / "index.json", "w") as f:
            json.dump(processed_prs, f, indent=2)
        
        logger.info(f"Saved PR data to {output_dir}")
        return output_dir