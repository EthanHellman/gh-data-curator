# src/dataflow/filtering/size_complexity_filter.py
from typing import Dict, Tuple, List, Any

class SizeComplexityFilter:
    """
    Filter PRs based on size and complexity measures.
    
    This filter evaluates PRs based on their size (files changed, lines added/removed)
    and complexity (type of changes, distribution across files, etc.) to identify
    meaningful contributions while filtering out both trivial and excessively large changes.
    """
    
    def __init__(self):
        # Thresholds for size filtering
        self.min_lines_changed = 3  # Minimum lines changed to be considered non-trivial
        self.max_lines_changed = 1000  # Maximum lines changed to be manageable
        self.max_files_changed = 20  # Maximum files changed to be focused
        
        # File types to prioritize or exclude
        self.code_extensions = [
            ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", 
            ".h", ".hpp", ".cs", ".go", ".rb", ".php", ".rs", ".swift"
        ]
        self.doc_extensions = [".md", ".rst", ".txt", ".pdf", ".doc", ".docx"]
        self.config_extensions = [".json", ".yml", ".yaml", ".xml", ".ini", ".toml"]
        self.generated_extensions = [".lock", ".min.js", ".min.css"]
    
    def apply(self, pr_data: Dict) -> Tuple[bool, Dict]:
        """
        Apply size and complexity filtering to a PR.
        
        Returns:
            Tuple[bool, Dict]: (passed_filter, metadata)
        """
        # Extract size metrics
        total_files = pr_data.get("file_count", 0)
        total_changes = pr_data.get("line_changes", 0)
        additions = pr_data.get("additions", 0)
        deletions = pr_data.get("deletions", 0)
        
        # Get code files
        code_files = pr_data.get("code_files", [])
        
        # Categorize files by type
        code_file_count = 0
        doc_file_count = 0
        config_file_count = 0
        generated_file_count = 0
        other_file_count = 0
        
        for file in code_files:
            filename = file.get("filename", "")
            if any(filename.endswith(ext) for ext in self.code_extensions):
                code_file_count += 1
            elif any(filename.endswith(ext) for ext in self.doc_extensions):
                doc_file_count += 1
            elif any(filename.endswith(ext) for ext in self.config_extensions):
                config_file_count += 1
            elif any(filename.endswith(ext) for ext in self.generated_extensions):
                generated_file_count += 1
            else:
                other_file_count += 1
        
        # Calculate code-focused metrics
        code_changes_ratio = 0.0
        if total_files > 0:
            code_changes_ratio = code_file_count / total_files
        
        # Apply size filters
        too_small = total_changes < self.min_lines_changed
        too_large = total_changes > self.max_lines_changed
        too_many_files = total_files > self.max_files_changed
        no_code_changes = code_file_count == 0
        only_generated_changes = code_file_count == 0 and generated_file_count > 0
        
        # Size score (0.0 to 1.0)
        # Higher score = better size (not too small, not too large)
        size_score = 0.0
        if total_changes >= self.min_lines_changed and total_changes <= self.max_lines_changed:
            # Score peaks at around 20-50 lines of changes
            optimal_size = 35  # Lines changed
            
            if total_changes <= optimal_size:
                size_score = total_changes / optimal_size
            else:
                # Gradually decrease score as size increases beyond optimal
                size_score = 1.0 - ((total_changes - optimal_size) / (self.max_lines_changed - optimal_size))
                size_score = max(0.0, size_score)  # Ensure non-negative
        
        # Complexity score based on file distribution
        complexity_score = 0.0
        if total_files > 0:
            # Higher score for focused changes (not spreading across too many files)
            files_factor = min(1.0, 5.0 / total_files)
            
            # Higher score for code-focused changes
            code_factor = min(1.0, code_changes_ratio * 1.5)
            
            complexity_score = (files_factor + code_factor) / 2.0
        
        # Combined normalized score
        normalized_score = (size_score * 0.6) + (complexity_score * 0.4)
        
        # Generate metadata
        metadata = {
            "total_files": total_files,
            "total_changes": total_changes,
            "additions": additions,
            "deletions": deletions,
            "code_file_count": code_file_count,
            "doc_file_count": doc_file_count,
            "config_file_count": config_file_count,
            "generated_file_count": generated_file_count,
            "other_file_count": other_file_count,
            "code_changes_ratio": code_changes_ratio,
            "too_small": too_small,
            "too_large": too_large,
            "too_many_files": too_many_files,
            "no_code_changes": no_code_changes,
            "only_generated_changes": only_generated_changes,
            "size_score": size_score,
            "complexity_score": complexity_score,
            "normalized_score": normalized_score
        }
        
        # Determine if PR passes the filter
        passed = (
            not too_small and 
            not too_large and 
            not too_many_files and 
            not no_code_changes and 
            not only_generated_changes
        )
        
        return passed, metadata