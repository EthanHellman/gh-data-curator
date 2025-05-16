# src/dataflow/filtering/file_relationship.py
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

class RelatedFilePredictor:
    """
    Predict files that are semantically related to changed files in a PR.
    
    This class implements heuristics to identify files that are related to
    the files changed in a PR but weren't modified, to provide a more complete
    context for the changes.
    """
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    def find_related_files(self, changed_files: List[str], limit: int = 5) -> List[str]:
        """
        Find files related to the changed files.
        
        Args:
            changed_files: List of file paths that were changed in the PR
            limit: Maximum number of related files to return
            
        Returns:
            List of file paths that are related but weren't changed
        """
        if not changed_files:
            return []
        
        # Normalize paths to make sure they're all relative to repo_path
        normalized_changed_files = []
        for file_path in changed_files:
            try:
                # If it's an absolute path, make it relative to repo_path
                if os.path.isabs(file_path):
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    normalized_changed_files.append(rel_path)
                else:
                    # Keep relative paths as they are
                    normalized_changed_files.append(file_path)
            except ValueError:
                # If we can't relativize (e.g., different drives), just use the basename
                normalized_changed_files.append(os.path.basename(file_path))
        
        # Collect candidate files using various heuristics
        candidates = set()
        
        # 1. Files in the same directories
        candidates.update(self._find_files_in_same_directories(normalized_changed_files))
        
        # 2. Files with import relationships
        candidates.update(self._find_import_relationships(normalized_changed_files))
        
        # 3. Files with naming similarities
        candidates.update(self._find_naming_similarities(normalized_changed_files))
        
        # Remove files that were already changed
        candidates = candidates - set(normalized_changed_files)
        
        # Sort by relevance score and limit
        scored_candidates = [(f, self._calculate_relevance(f, normalized_changed_files)) for f in candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [f for f, score in scored_candidates[:limit]]
    
    def _find_files_in_same_directories(self, changed_files: List[str]) -> Set[str]:
        """Find files in the same directories as the changed files."""
        related_files = set()
        
        # Get unique directories from changed files
        directories = set()
        for file_path in changed_files:
            # Construct the full path correctly
            full_path = self.repo_path / file_path
            directory = full_path.parent
            if directory.exists():
                directories.add(directory)
        
        # Find Python files in those directories
        for directory in directories:
            for file_path in directory.glob("*.py"):
                if file_path.is_file():
                    # Convert back to relative path for consistency
                    rel_path = file_path.relative_to(self.repo_path)
                    related_files.add(str(rel_path))
        
        return related_files
    
    def _find_import_relationships(self, changed_files: List[str]) -> Set[str]:
        """Find files that import from or are imported by the changed files."""
        related_files = set()
        
        # Extract module names from changed files
        modules = []
        for file_path in changed_files:
            module_name = self._get_module_name(file_path)
            if module_name:
                modules.append(module_name)
        
        # Search for imports of these modules in repo
        for py_file in self.repo_path.glob("**/*.py"):
            # Convert to relative path for comparison
            rel_py_file = py_file.relative_to(self.repo_path)
            rel_path_str = str(rel_py_file)
            
            if rel_path_str in changed_files:
                continue
                
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    content = f.read()
                    
                    # Check if file imports any of the modules
                    for module in modules:
                        if re.search(f"import\\s+{module}|from\\s+{module}\\s+import", content):
                            related_files.add(rel_path_str)
                            break
                except Exception:
                    # Skip files that can't be read
                    continue
        
        return related_files
    
    def _find_naming_similarities(self, changed_files: List[str]) -> Set[str]:
        """Find files with naming similarities to the changed files."""
        related_files = set()
        
        # Extract base names without extensions
        base_names = []
        for file_path in changed_files:
            base_name = Path(file_path).stem
            if base_name:
                base_names.append(base_name)
        
        # Search for files with similar names
        for py_file in self.repo_path.glob("**/*.py"):
            # Convert to relative path for comparison
            rel_py_file = py_file.relative_to(self.repo_path)
            rel_path_str = str(rel_py_file)
            
            if rel_path_str in changed_files:
                continue
                
            file_name = py_file.stem
            
            # Check name similarities
            for base_name in base_names:
                # Exact matches without extension
                if file_name == base_name:
                    related_files.add(rel_path_str)
                    continue
                    
                # Test prefix is often associated with implementation
                if file_name == f"test_{base_name}" or base_name == f"test_{file_name}":
                    related_files.add(rel_path_str)
                    continue
                    
                # Common patterns like util/utils, model/models
                if file_name.startswith(f"{base_name}_") or base_name.startswith(f"{file_name}_"):
                    related_files.add(rel_path_str)
                    continue
        
        return related_files
    
    def _get_module_name(self, file_path: str) -> str:
        """Convert a file path to a Python module name."""
        path = Path(file_path)
        
        if not path.suffix == ".py":
            return ""
            
        # Convert path to module name (e.g., src/dataflow/acquisition/pr_collector.py -> src.dataflow.acquisition.pr_collector)
        parts = list(path.parts)
        
        # Remove .py extension
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
            
        # Join with dots
        return ".".join(parts)
    
    def _calculate_relevance(self, file_path: str, changed_files: List[str]) -> float:
        """Calculate relevance score for a candidate file."""
        # Default score
        score = 0.5
        
        # Directory proximity
        for changed_file in changed_files:
            try:
                # Make sure both paths are in the same format before comparing
                path1 = str(Path(file_path))
                path2 = str(Path(changed_file))
                
                # Get common path components
                common_path = os.path.commonpath([path1, path2])
                
                if common_path:
                    # More shared path components = higher score
                    common_parts = len(Path(common_path).parts)
                    file_parts = len(Path(file_path).parts)
                    proximity = common_parts / file_parts
                    score = max(score, 0.3 + (proximity * 0.7))  # Scale to 0.3-1.0
            except ValueError:
                # If paths can't be compared (e.g., different drives), skip
                continue
        
        # Name similarity
        file_name = Path(file_path).stem
        for changed_file in changed_files:
            changed_name = Path(changed_file).stem
            
            if file_name == changed_name:
                score = max(score, 0.9)  # Same name, different directory
            elif file_name.startswith(changed_name) or changed_name.startswith(file_name):
                score = max(score, 0.7)  # Name prefix match
        
        return score