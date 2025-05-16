# src/dataflow/filtering/import_analyzer.py
import os
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class ImportAnalyzer:
    """
    Analyze import relationships between Python files in a repository.
    
    This class builds an import dependency graph for Python files in a repository,
    which can be used to find related files based on import relationships.
    """
    
    def __init__(self, repo_path: Path):
        """
        Initialize the analyzer.
        
        Args:
            repo_path: Path to the repository
        """
        self.repo_path = repo_path
        self.import_graph = defaultdict(set)  # file -> set of imported files
        self.reverse_import_graph = defaultdict(set)  # file -> set of files that import it
        self.module_to_file = {}  # module name -> file path
        self.file_to_module = {}  # file path -> module name
        
        # Cache for file content
        self._file_content_cache = {}
    
    def build_import_graph(self, max_files: int = 10000):
        """
        Build the import dependency graph for the repository.
        
        Args:
            max_files: Maximum number of files to process
        """
        logger.info(f"Building import graph for repository: {self.repo_path}")
        
        # First, build a mapping of module names to file paths
        self._build_module_mapping()
        
        # Then, analyze imports in each Python file
        py_files = list(self.repo_path.glob("**/*.py"))[:max_files]
        logger.info(f"Analyzing imports in {len(py_files)} Python files")
        
        for file_path in py_files:
            relative_path = file_path.relative_to(self.repo_path)
            self._analyze_file_imports(relative_path)
        
        logger.info(f"Import graph built with {len(self.import_graph)} files and {len(self.module_to_file)} modules")
    
    def find_related_files(self, file_path: str, depth: int = 1) -> Set[str]:
        """
        Find files related to the given file based on import relationships.
        
        Args:
            file_path: Path to the file (relative to repo root)
            depth: Maximum dependency depth to traverse
            
        Returns:
            Set of related file paths
        """
        # Normalize the file path
        file_path = str(Path(file_path))
        
        related_files = set()
        
        # Find files that this file imports
        self._find_imports_recursive(file_path, related_files, depth, True)
        
        # Find files that import this file
        self._find_imports_recursive(file_path, related_files, depth, False)
        
        # Remove the original file from the results
        if file_path in related_files:
            related_files.remove(file_path)
        
        return related_files
    
    def _build_module_mapping(self):
        """Build mapping between module names and file paths."""
        # Find all Python package directories (containing __init__.py)
        packages = set()
        for init_file in self.repo_path.glob("**/__init__.py"):
            package_dir = init_file.parent
            package_relpath = package_dir.relative_to(self.repo_path)
            packages.add(str(package_relpath))
            
            # Add the module name for this package
            module_name = str(package_relpath).replace("/", ".")
            self.module_to_file[module_name] = str(package_relpath / "__init__.py")
            self.file_to_module[str(package_relpath / "__init__.py")] = module_name
        
        # For each Python file, determine its module name
        for py_file in self.repo_path.glob("**/*.py"):
            if py_file.name == "__init__.py":
                continue  # Already handled above
                
            relative_path = py_file.relative_to(self.repo_path)
            parent_dir = str(relative_path.parent)
            
            # Check if this file is part of a package
            if parent_dir in packages:
                # This file is in a package
                module_name = f"{parent_dir.replace('/', '.')}.{py_file.stem}"
            else:
                # This file is a standalone module
                module_name = py_file.stem
                
            self.module_to_file[module_name] = str(relative_path)
            self.file_to_module[str(relative_path)] = module_name
    
    def _analyze_file_imports(self, file_path: Path):
        """
        Analyze imports in a Python file and add to the import graph.
        
        Args:
            file_path: Path to the file relative to repo root
        """
        abs_path = self.repo_path / file_path
        if not abs_path.exists():
            return
            
        try:
            # Read file content
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()
                self._file_content_cache[str(file_path)] = file_content
            
            # Parse the file with AST
            tree = ast.parse(file_content)
            
            # Find all import statements
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.level > 0:  # Relative import
                            # Calculate the absolute module path
                            parent_parts = str(file_path.parent).split("/")
                            module_parts = parent_parts[:-node.level] if node.level <= len(parent_parts) else []
                            if node.module != ".":
                                module_parts.append(node.module)
                            absolute_module = ".".join(module_parts)
                            imports.append(absolute_module)
                        else:
                            imports.append(node.module)
            
            # Filter imports to only include modules from this repository
            repo_imports = [imp for imp in imports if imp in self.module_to_file]
            
            # Add edges to the import graph
            for import_module in repo_imports:
                import_file = self.module_to_file.get(import_module)
                if import_file:
                    self.import_graph[str(file_path)].add(import_file)
                    self.reverse_import_graph[import_file].add(str(file_path))
        
        except Exception as e:
            logger.error(f"Error analyzing imports in {file_path}: {e}")
    
    def _find_imports_recursive(self, file_path: str, related_files: Set[str], depth: int, forward: bool = True):
        """
        Recursively find imported files or files that import this file.
        
        Args:
            file_path: Path to the file
            related_files: Set to populate with related files
            depth: Maximum depth to traverse
            forward: If True, find imports; if False, find importers
        """
        if depth <= 0:
            return
            
        # Add the current file to the result set
        related_files.add(file_path)
        
        # Get the next level of files based on direction
        next_files = self.import_graph[file_path] if forward else self.reverse_import_graph[file_path]
        
        # Recursively process the next level
        for next_file in next_files:
            if next_file not in related_files:  # Avoid cycles
                self._find_imports_recursive(next_file, related_files, depth - 1, forward)
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get the content of a file from cache or disk."""
        if file_path in self._file_content_cache:
            return self._file_content_cache[file_path]
            
        abs_path = self.repo_path / file_path
        if not abs_path.exists():
            return None
            
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                self._file_content_cache[file_path] = content
                return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None


# Simple test function
def test_import_analyzer():
    """Test the ImportAnalyzer with a sample repository."""
    repo_path = Path("./example_repo")  # Replace with actual repo path
    
    if not repo_path.exists():
        print(f"Repository path does not exist: {repo_path}")
        return
    
    analyzer = ImportAnalyzer(repo_path)
    analyzer.build_import_graph()
    
    # Test finding related files for a sample file
    test_file = "example_module/main.py"  # Replace with an actual file path
    related_files = analyzer.find_related_files(test_file, depth=2)
    
    print(f"Files related to {test_file}:")
    for file in related_files:
        print(f"  {file}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    test_import_analyzer()