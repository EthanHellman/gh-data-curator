# src/dataflow/filtering/filter_pipeline.py
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

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
    
    def _apply_filters(self, pr_data: Dict) -> Tuple[bool, Dict]:
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