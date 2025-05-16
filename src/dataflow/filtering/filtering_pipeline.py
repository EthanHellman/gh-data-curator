# # src/dataflow/filtering/filter_pipeline.py
# import logging
# import re
# from pathlib import Path
# from typing import Dict, List, Any, Optional, Set, Tuple

# logger = logging.getLogger(__name__)

# src/dataflow/filtering/filtering_pipeline.py
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

# Import the filter components
from dataflow.filtering.bot_filter import BotFilter
from dataflow.filtering.size_complexity_filter import SizeComplexityFilter
from dataflow.filtering.content_relevance_filter import ContentRelevanceFilter
from dataflow.filtering.file_relationship import RelatedFilePredictor

logger = logging.getLogger(__name__)

class FilterPipeline:
    """
    Advanced multi-stage filtering pipeline for PR data.
    
    This pipeline applies a series of increasingly refined filters to PR data,
    capturing metadata at each stage to enable quality assessment and analysis.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = data_dir / "processed"
        self.filtered_dir = data_dir / "filtered"
        self.filtered_dir.mkdir(exist_ok=True)
        
        # Initialize filters
        self.bot_filter = BotFilter()
        self.size_filter = SizeComplexityFilter()
        self.content_filter = ContentRelevanceFilter()
        
        # Filtering statistics
        self.stats = {
            "total_prs": 0,
            "bot_filtered": 0,
            "size_filtered": 0,
            "content_filtered": 0,
            "passed_all_filters": 0
        }
    
    def filter_repository(self, owner: str, repo: str) -> Path:
        """Filter all processed PRs for a repository."""
        repo_dir = self.processed_dir / f"{owner}_{repo}"
        if not repo_dir.exists():
            logger.error(f"Processed repository data not found: {repo_dir}")
            return None
        
        # Output directory for filtered data
        output_dir = self.filtered_dir / f"{owner}_{repo}"
        output_dir.mkdir(exist_ok=True)
        
        # Load processed PR index
        index_path = repo_dir / "processed_index.json"
        if not index_path.exists():
            logger.error(f"Processed PR index not found: {index_path}")
            return None
        
        with open(index_path, "r") as f:
            processed_prs = json.load(f)
        
        logger.info(f"Filtering {len(processed_prs)} PRs for {owner}/{repo}")
        self.stats["total_prs"] += len(processed_prs)
        
        filtered_prs = []
        filter_metadata = []
        
        for pr in processed_prs:
            pr_number = pr["pr_number"]
            pr_dir = repo_dir / f"pr_{pr_number}"
            
            if not pr_dir.exists():
                logger.warning(f"PR directory not found: {pr_dir}")
                continue
            
            logger.info(f"Filtering PR #{pr_number}: {pr.get('title', '')}")
            
            # Load detailed processed PR data
            processed_file = pr_dir / "processed.json"
            if not processed_file.exists():
                logger.warning(f"Processed PR file not found: {processed_file}")
                continue
                
            with open(processed_file, "r") as f:
                pr_data = json.load(f)
            
            # Apply filters and collect metadata
            filter_result, metadata = self._apply_filters(pr_data)
            
            # If PR passes all filters, add to filtered list
            if filter_result:
                filtered_prs.append(pr_data)
                
                # Save filtered PR
                pr_output_dir = output_dir / f"pr_{pr_number}"
                pr_output_dir.mkdir(exist_ok=True)
                
                with open(pr_output_dir / "filtered.json", "w") as f:
                    json.dump(pr_data, f, indent=2)
                
                # Save filter metadata
                with open(pr_output_dir / "filter_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
            
            # Always save filter metadata for analysis
            filter_metadata.append({
                "pr_number": pr_number,
                "passed_filter": filter_result,
                **metadata
            })
        
        # Save filtered PR index
        with open(output_dir / "filtered_index.json", "w") as f:
            json.dump(filtered_prs, f, indent=2)
        
        # Save filter metadata index
        with open(output_dir / "filter_metadata.json", "w") as f:
            json.dump(filter_metadata, f, indent=2)
        
        logger.info(f"Saved filtered PR data to {output_dir}")
        logger.info(f"Filtering stats: {self.stats}")
        
        return output_dir

    # This is an addition to the FilterPipeline class in filtering_pipeline.py

    def find_related_files(self, pr_data: Dict, repo_path: Optional[Path] = None) -> Dict:
        """Find files related to the changes in a PR."""
        if not repo_path:
            # Try to construct repo path from data directory
            owner_repo = pr_data.get("repository", {}).get("full_name", "")
            if owner_repo:
                owner, repo = owner_repo.split("/")
                repo_path = self.data_dir / "repos" / f"{owner}_{repo}"
            else:
                logger.warning("Unable to determine repository path for related file prediction")
                return pr_data
        
        # Skip if repo path doesn't exist
        if not repo_path.exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return pr_data
        
        # Get changed files
        changed_files = []
        for file in pr_data.get("code_files", []):
            filename = file.get("filename", "")
            if filename:
                changed_files.append(filename)
        
        # Find related files
        related_file_predictor = RelatedFilePredictor(repo_path)
        related_files = related_file_predictor.find_related_files(changed_files)
        
        # Add related files to PR data
        pr_data["related_files"] = related_files
        
        return pr_data
    
    def _apply_filters(self, pr_data: Dict, repo_path: Optional[Path] = None) -> Tuple[bool, Dict]:
        """Apply all filters and generate metadata."""
        metadata = {
            "bot_filter": {
                "passed": False,
                "details": {}
            },
            "size_filter": {
                "passed": False,
                "details": {}
            },
            "content_filter": {
                "passed": False,
                "details": {}
            },
            "related_files": [],
            "quality_score": 0.0
        }
        
        # Apply bot filter
        bot_result, bot_metadata = self.bot_filter.apply(pr_data)
        metadata["bot_filter"]["passed"] = bot_result
        metadata["bot_filter"]["details"] = bot_metadata
        
        if not bot_result:
            self.stats["bot_filtered"] += 1
            return False, metadata
        
        # Apply size filter
        size_result, size_metadata = self.size_filter.apply(pr_data)
        metadata["size_filter"]["passed"] = size_result
        metadata["size_filter"]["details"] = size_metadata
        
        if not size_result:
            self.stats["size_filtered"] += 1
            return False, metadata
        
        # Apply content filter
        content_result, content_metadata = self.content_filter.apply(pr_data)
        metadata["content_filter"]["passed"] = content_result
        metadata["content_filter"]["details"] = content_metadata
        
        if not content_result:
            self.stats["content_filtered"] += 1
            return False, metadata
        
        # Find related files if PR passes filters
        if repo_path:
            related_file_predictor = RelatedFilePredictor(repo_path)
            changed_files = [file.get("filename", "") for file in pr_data.get("code_files", [])]
            related_files = related_file_predictor.find_related_files(changed_files)
            metadata["related_files"] = related_files
            
            # Add related files to PR data for further processing
            pr_data["related_files"] = related_files
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(bot_metadata, size_metadata, content_metadata)
        metadata["quality_score"] = quality_score
        
        self.stats["passed_all_filters"] += 1
        return True, metadata
    
    def _calculate_quality_score(self, bot_metadata: Dict, size_metadata: Dict, content_metadata: Dict) -> float:
        """Calculate an overall quality score for the PR."""
        # Weights for different components
        bot_weight = 0.2
        size_weight = 0.3
        content_weight = 0.5
        
        # Component scores
        bot_score = bot_metadata.get("confidence", 1.0)
        size_score = size_metadata.get("normalized_score", 0.0)
        content_score = content_metadata.get("relevance_score", 0.0)
        
        # Weighted average
        return (bot_score * bot_weight + size_score * size_weight + content_score * content_weight)