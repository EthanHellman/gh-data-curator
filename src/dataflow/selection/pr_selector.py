# src/dataflow/selection/pr_selector.py
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

logger = logging.getLogger(__name__)

class PRSelector:
    """
    Select high-quality PRs from the collected raw data based on heuristic quality metrics.
    
    This performs a lightweight analysis to identify PRs that are likely to be high quality
    without running the full filtering pipeline. It's used to pre-select PRs for further
    processing.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the selector.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.selected_dir = data_dir / "selected"
        self.selected_dir.mkdir(exist_ok=True)
        
        # Initialize quality scoring weights
        self.weights = {
            "non_bot_author": 0.25,            # Not a bot (by username detection)
            "non_bot_title": 0.15,             # Not a bot (by title pattern)
            "non_trivial_changes": 0.15,       # Non-trivial number of changes
            "has_meaningful_pr_body": 0.1,     # Has a substantial PR description
            "has_linked_issue": 0.1,           # Has linked issues
            "has_substantial_comments": 0.05,  # Has a good number of comments
            "has_code_changes": 0.2,           # Changes code files, not just docs or config
        }
        
        # Patterns for bot detection
        self.bot_username_patterns = [
            r'bot$', r'[_-]bot', r'^bot[_-]', r'\[bot\]',
            r'dependabot', r'renovate', r'github-actions',
            r'semantic-release', r'codecov', r'travis'
        ]
        
        # Bot PR title patterns
        self.bot_title_patterns = [
            r'^(build|chore|ci|docs)(\(.*\))?:',  # Conventional commit prefixes common for automation
            r'bump|upgrade|update.*(version|dependency)',
            r'update.*\.(lock|json|yml|yaml|xml)$',
            r'dependency|version',
            r'auto(mate(d)?)?[_-]'
        ]
    
    def select_quality_prs(self, org: str, repo: str, count: int = 30, 
                         initial_sample: int = 100) -> Tuple[Path, List[Dict]]:
        """
        Select high-quality PRs for further processing.
        
        Args:
            org: Repository owner/organization
            repo: Repository name
            count: Number of PRs to select
            initial_sample: Initial number of PRs to analyze
            
        Returns:
            Tuple of (output directory path, list of selected PRs)
        """
        repo_key = f"{org}_{repo}"
        raw_repo_dir = self.raw_dir / repo_key
        
        if not raw_repo_dir.exists():
            logger.error(f"Raw data not found for {org}/{repo}. Run collection first.")
            return None, []
        
        # Load PR index
        index_path = raw_repo_dir / "index.json"
        if not index_path.exists():
            logger.error(f"PR index not found: {index_path}")
            return None, []
        
        with open(index_path, "r") as f:
            all_prs = json.load(f)
        
        logger.info(f"Found {len(all_prs)} PRs for {org}/{repo}")
        
        # Use either all PRs or the initial sample, whichever is smaller
        prs_to_score = all_prs[:min(len(all_prs), initial_sample)]
        logger.info(f"Scoring {len(prs_to_score)} PRs for quality assessment")
        
        # Score PRs by quality
        scored_prs = []
        for pr in prs_to_score:
            quality_score, quality_details = self._score_pr_quality(pr, raw_repo_dir)
            scored_prs.append((pr, quality_score, quality_details))
        
        # Sort by quality score
        scored_prs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top PRs
        selected_count = min(count, len(scored_prs))
        top_prs = [pr for pr, _, _ in scored_prs[:selected_count]]
        
        logger.info(f"Selected {len(top_prs)} highest quality PRs")
        
        # Create output directory
        output_dir = self.selected_dir / repo_key
        output_dir.mkdir(exist_ok=True)
        
        # Save selected PR index
        with open(output_dir / "selected_index.json", "w") as f:
            json.dump(top_prs, f, indent=2)
        
        # Save selection metadata with quality scores
        selection_metadata = {
            "total_prs_found": len(all_prs),
            "prs_scored": len(prs_to_score),
            "prs_selected": len(top_prs),
            "average_quality_score": sum(score for _, score, _ in scored_prs) / len(scored_prs) if scored_prs else 0,
            "average_selected_quality": sum(scored_prs[i][1] for i in range(selected_count)) / selected_count if selected_count > 0 else 0,
            "selection_details": [
                {
                    "pr_number": pr["pr_number"],
                    "title": pr["title"],
                    "quality_score": score,
                    "quality_details": details
                }
                for pr, score, details in scored_prs[:selected_count]
            ]
        }
        
        with open(output_dir / "selection_metadata.json", "w") as f:
            json.dump(selection_metadata, f, indent=2)
        
        # Copy the PR directories for selected PRs
        for pr in top_prs:
            pr_number = pr["pr_number"]
            source_dir = raw_repo_dir / f"pr_{pr_number}"
            target_dir = output_dir / f"pr_{pr_number}"
            
            if source_dir.exists() and not target_dir.exists():
                # Create target directory
                target_dir.mkdir(exist_ok=True)
                
                # Copy all files
                for source_file in source_dir.glob("*"):
                    target_file = target_dir / source_file.name
                    with open(source_file, "r") as src, open(target_file, "w") as dst:
                        dst.write(src.read())
        
        logger.info(f"Saved selected PRs to {output_dir}")
        return output_dir, top_prs
    
    def _score_pr_quality(self, pr: Dict, raw_repo_dir: Path) -> Tuple[float, Dict]:
        """
        Score a PR's quality using lightweight heuristics.
        
        Args:
            pr: PR summary data
            raw_repo_dir: Directory containing raw PR data
            
        Returns:
            Tuple of (quality score, details dictionary)
        """
        quality_details = {}
        
        # 1. Check if author looks like a bot
        author = pr.get("author", "")
        is_bot_author = any(re.search(pattern, author, re.IGNORECASE) for pattern in self.bot_username_patterns)
        quality_details["non_bot_author"] = not is_bot_author
        
        # 2. Check if title looks like a bot PR
        title = pr.get("title", "")
        is_bot_title = any(re.search(pattern, title, re.IGNORECASE) for pattern in self.bot_title_patterns)
        quality_details["non_bot_title"] = not is_bot_title
        
        # 3. Check if changes are non-trivial
        file_count = len(pr.get("file_paths", []))
        quality_details["non_trivial_changes"] = 3 <= file_count <= 20  # Not too small, not too big
        
        # 4. Check if there's a meaningful PR body
        body = pr.get("body", "")
        has_body = bool(body and len(body.split()) > 10)  # At least 10 words
        quality_details["has_meaningful_pr_body"] = has_body
        
        # 5. Check if there are linked issues
        linked_issues = pr.get("linked_issues", [])
        quality_details["has_linked_issue"] = len(linked_issues) > 0
        
        # 6. Check for substantial comments
        comments = pr.get("comments", 0)
        quality_details["has_substantial_comments"] = comments >= 2
        
        # 7. Check if code files were changed (not just docs/config)
        file_paths = pr.get("file_paths", [])
        has_code_changes = False
        
        code_extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rb", ".php", ".rs", ".swift"]
        doc_extensions = [".md", ".rst", ".txt", ".pdf", ".doc", ".docx"]
        config_extensions = [".json", ".yml", ".yaml", ".xml", ".ini", ".toml"]
        
        for file_path in file_paths:
            if any(file_path.endswith(ext) for ext in code_extensions):
                has_code_changes = True
                break
        
        quality_details["has_code_changes"] = has_code_changes
        
        # Calculate weighted score
        quality_score = sum(
            self.weights[factor] * (1.0 if value else 0.0)
            for factor, value in quality_details.items()
        )
        
        return quality_score, quality_details