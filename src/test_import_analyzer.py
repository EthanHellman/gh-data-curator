#!/usr/bin/env python3
"""
Test script for the Import Analyzer.
This script demonstrates how to use the import analyzer to find related files.
"""

import logging
import os
import sys
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the import analyzer
from dataflow.filtering.import_analyzer import ImportAnalyzer

def test_with_repo(repo_path: Path):
    """Test the import analyzer with a repository."""
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return
    
    logger.info(f"Testing import analyzer with repository: {repo_path}")
    
    # Initialize the analyzer
    analyzer = ImportAnalyzer(repo_path)
    
    # Build the import graph
    analyzer.build_import_graph()
    
    # Choose a few random Python files to analyze
    py_files = list(repo_path.glob("**/*.py"))[:10]  # Limit to 10 files for testing
    
    if not py_files:
        logger.warning("No Python files found in the repository")
        return
    
    logger.info(f"Testing with {len(py_files)} Python files")
    
    for file_path in py_files:
        relative_path = file_path.relative_to(repo_path)
        logger.info(f"\nAnalyzing related files for: {relative_path}")
        
        # Find related files
        related_files = analyzer.find_related_files(str(relative_path), depth=1)
        
        if related_files:
            logger.info(f"Found {len(related_files)} related files:")
            for related_file in sorted(related_files)[:5]:  # Limit output to 5 files
                logger.info(f"  - {related_file}")
            
            if len(related_files) > 5:
                logger.info(f"  ... and {len(related_files) - 5} more")
        else:
            logger.info("No related files found")

def analyze_specific_file(repo_path: Path, file_path: str, depth: int = 1):
    """Analyze imports for a specific file."""
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return
    
    full_path = repo_path / file_path
    if not full_path.exists():
        logger.error(f"File does not exist: {full_path}")
        return
    
    logger.info(f"Analyzing imports for file: {file_path}")
    
    # Initialize the analyzer
    analyzer = ImportAnalyzer(repo_path)
    
    # Build the import graph
    analyzer.build_import_graph()
    
    # Find related files
    related_files = analyzer.find_related_files(file_path, depth=depth)
    
    if related_files:
        logger.info(f"Found {len(related_files)} related files:")
        for related_file in sorted(related_files):
            logger.info(f"  - {related_file}")
    else:
        logger.info("No related files found")

def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description="Test the import analyzer")
    parser.add_argument("--repo", type=str, default="", help="Path to repository")
    parser.add_argument("--file", type=str, help="Specific file to analyze (relative to repo root)")
    parser.add_argument("--depth", type=int, default=1, help="Depth of import analysis (default: 1)")
    args = parser.parse_args()
    
    # If repo is not specified, use the standard data directory
    if not args.repo:
        base_dir = Path("~/gh-data-curator/data").expanduser()
        repo_path = base_dir / "repos" / "django_django"
        if not repo_path.exists():
            logger.error(f"Default repository not found at {repo_path}")
            logger.error("Please specify a repository path with --repo")
            return
    else:
        repo_path = Path(args.repo)
    
    if args.file:
        # Analyze a specific file
        analyze_specific_file(repo_path, args.file, args.depth)
    else:
        # Test with random files in the repo
        test_with_repo(repo_path)

if __name__ == "__main__":
    main()