import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PRProcessor:
    """Process raw PR data into a structured format for analysis."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.selected_dir = data_dir / "selected"
        self.processed_dir = data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
    
    def process_repository(self, owner: str, repo: str, source_dir: Optional[Path] = None) -> Path:
        """
        Process all PRs for a repository.
        
        Args:
            owner: Repository owner or organization
            repo: Repository name
            source_dir: Optional source directory (defaults to raw data directory)
        
        Returns:
            Path to the processed data directory
        """
        repo_key = f"{owner}_{repo}"
        
        # Use provided source directory or default to raw data
        if source_dir is None:
            # Check selected directory first, fall back to raw
            selected_repo_dir = self.selected_dir / repo_key
            if selected_repo_dir.exists():
                repo_dir = selected_repo_dir
                logger.info(f"Using selected data as source for processing: {repo_dir}")
            else:
                repo_dir = self.raw_dir / repo_key
                logger.info(f"Using raw data as source for processing: {repo_dir}")
        else:
            repo_dir = source_dir
            logger.info(f"Using provided source directory for processing: {repo_dir}")
            
        if not repo_dir.exists():
            logger.error(f"Source data not found: {repo_dir}")
            return None
        
        # Output directory for processed data
        output_dir = self.processed_dir / repo_key
        output_dir.mkdir(exist_ok=True)
        
        # Load PR index
        index_path = repo_dir / "index.json"
        selected_index_path = repo_dir / "selected_index.json"
        
        # Check for either regular or selected index
        if index_path.exists():
            with open(index_path, "r") as f:
                pr_index = json.load(f)
            logger.info(f"Using standard index with {len(pr_index)} PRs")
        elif selected_index_path.exists():
            with open(selected_index_path, "r") as f:
                pr_index = json.load(f)
            logger.info(f"Using selected index with {len(pr_index)} PRs")
        else:
            logger.error(f"No PR index found in {repo_dir}")
            return None
        
        logger.info(f"Processing {len(pr_index)} PRs for {owner}/{repo}")
        
        processed_prs = []
        
        for pr_summary in pr_index:
            pr_number = pr_summary["pr_number"]
            pr_raw_dir = repo_dir / f"pr_{pr_number}"
            
            if not pr_raw_dir.exists():
                logger.warning(f"PR directory not found: {pr_raw_dir}")
                continue
            
            logger.info(f"Processing PR #{pr_number}: {pr_summary['title']}")
            
            # Process PR data
            processed_pr = self._process_single_pr(pr_raw_dir, pr_summary)
            if processed_pr:
                processed_prs.append(processed_pr)
                
                # Save processed PR
                pr_output_dir = output_dir / f"pr_{pr_number}"
                pr_output_dir.mkdir(exist_ok=True)
                
                with open(pr_output_dir / "processed.json", "w") as f:
                    json.dump(processed_pr, f, indent=2)
        
        # Save processed PR index
        with open(output_dir / "processed_index.json", "w") as f:
            json.dump(processed_prs, f, indent=2)
        
        logger.info(f"Saved processed PR data to {output_dir}")
        return output_dir
    
    def _process_single_pr(self, pr_dir: Path, pr_summary: Dict) -> Optional[Dict]:
        """Process a single PR and extract relevant information."""
        try:
            # Load all data files
            with open(pr_dir / "basic_info.json", "r") as f:
                basic_info = json.load(f)
            
            details_path = pr_dir / "details.json"
            details = None
            if details_path.exists():
                with open(details_path, "r") as f:
                    details = json.load(f)
            
            files_path = pr_dir / "files.json"
            files = []
            if files_path.exists():
                with open(files_path, "r") as f:
                    files = json.load(f)
            
            comments_path = pr_dir / "comments.json"
            comments = []
            if comments_path.exists():
                with open(comments_path, "r") as f:
                    comments = json.load(f)
            
            linked_issues_path = pr_dir / "linked_issues.json"
            linked_issues = []
            if linked_issues_path.exists():
                with open(linked_issues_path, "r") as f:
                    linked_issues = json.load(f)
            
            # Extract code context
            code_files = self._extract_code_files(files)
            
            # Extract issue description (if available)
            issue_description = self._extract_issue_description(linked_issues)
            
            # Create processed PR record
            processed_pr = {
                "pr_number": pr_summary["pr_number"],
                "title": pr_summary["title"],
                "author": pr_summary["author"],
                "created_at": pr_summary["created_at"],
                "merged_at": pr_summary["merged_at"],
                "body": pr_summary["body"],
                "code_files": code_files,
                "issue_description": issue_description,
                "has_linked_issue": len(linked_issues) > 0,
                "comment_count": len(comments),
                "is_bot_pr": self._is_bot_pr(basic_info, details),
                "is_bug_fix": self._is_bug_fix(pr_summary["title"], pr_summary["body"], issue_description),
                "file_count": len(files),
                "line_changes": sum(f.get("additions", 0) + f.get("deletions", 0) for f in files),
                "additions": sum(f.get("additions", 0) for f in files),
                "deletions": sum(f.get("deletions", 0) for f in files),
            }
            
            return processed_pr
            
        except Exception as e:
            logger.error(f"Error processing PR {pr_dir}: {e}")
            return None
    
    def _extract_code_files(self, files: List[Dict]) -> List[Dict]:
        """Extract code files from PR files."""
        code_files = []
        for file in files:
            filename = file.get("filename")
            if not filename:
                continue
                
            # Skip non-code files
            if not self._is_code_file(filename):
                continue
            
            code_files.append({
                "filename": filename,
                "additions": file.get("additions", 0),
                "deletions": file.get("deletions", 0),
                "changes": file.get("changes", 0),
                "status": file.get("status"),
                "patch": file.get("patch")
            })
        
        return code_files
    
    def _is_code_file(self, filename: str) -> bool:
        """Check if a file is a code file."""
        code_extensions = [
            ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", 
            ".h", ".hpp", ".cs", ".go", ".rb", ".php", ".rs", ".swift"
        ]
        
        return any(filename.endswith(ext) for ext in code_extensions)
    
    def _extract_issue_description(self, linked_issues: List[Dict]) -> Optional[str]:
        """Extract issue description from linked issues."""
        if not linked_issues:
            return None
        
        # Use the first linked issue
        issue = linked_issues[0]
        return issue.get("body")
    
    def _is_bot_pr(self, basic_info: Dict, details: Optional[Dict]) -> bool:
        """Check if a PR is created by a bot."""
        # Check author login for bot indicators
        author = basic_info.get("user", {}).get("login", "")
        if author and re.search(r"bot|dependabot|renovate|github-actions", author, re.IGNORECASE):
            return True
        
        # Check commit author
        if details and details.get("commits_url"):
            # For detailed bot detection, we would fetch commits here
            # but for simplicity we're just using the PR author
            pass
        
        # Check PR title for automated messages
        title = basic_info.get("title", "")
        if title and re.search(r"^(build|chore|ci|docs|style|refactor|perf|test):", title, re.IGNORECASE):
            return True
        
        return False
    
    def _is_bug_fix(self, title: str, body: str, issue_description: Optional[str]) -> bool:
        """Check if a PR is a bug fix."""
        # Check title for bug fix indicators
        if title and re.search(r"fix|bug|issue|error|crash|problem|fail", title, re.IGNORECASE):
            return True
        
        # Check body for bug fix indicators
        if body and re.search(r"fix|bug|issue|error|crash|problem|fail", body, re.IGNORECASE):
            return True
        
        # Check issue description for bug fix indicators
        if issue_description and re.search(r"fix|bug|issue|error|crash|problem|fail", issue_description, re.IGNORECASE):
            return True
        
        return False