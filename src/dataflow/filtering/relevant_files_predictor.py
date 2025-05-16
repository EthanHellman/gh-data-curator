# src/dataflow/filtering/relevant_files_predictor.py
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class RelevantFilesPredictor:
    """
    Predict files that are relevant but not modified in a PR.
    
    This class implements the approach described in the SWE-RL paper, using
    an LLM to find files that are semantically connected to the PR but weren't
    modified. This provides better context for understanding the changes.
    """
    
    def __init__(self, repo_path: Optional[Path] = None, openai_api_key: Optional[str] = None, 
                 use_openai: bool = False, max_workers: int = 5, 
                 use_import_analysis: bool = True):
        """
        Initialize the predictor.
        
        Args:
            repo_path: Path to the cloned repository (optional for heuristic methods)
            openai_api_key: OpenAI API key (required if use_openai=True)
            use_openai: Whether to use OpenAI API for prediction
            max_workers: Maximum number of concurrent API requests
            use_import_analysis: Whether to use import analysis for repository-based prediction
        """
        self.repo_path = repo_path
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.use_openai = use_openai
        self.max_workers = max_workers
        self.use_import_analysis = use_import_analysis
        
        # Initialize heuristic-based file relationship predictor
        from dataflow.filtering.file_relationship import RelatedFilePredictor
        self.heuristic_predictor = RelatedFilePredictor(repo_path) if repo_path else None
        
        # Initialize import analyzer if requested and repository is available
        self.import_analyzer = None
        if use_import_analysis and repo_path and repo_path.exists():
            try:
                from dataflow.filtering.import_analyzer import ImportAnalyzer
                self.import_analyzer = ImportAnalyzer(repo_path)
                # Build the import graph lazily when first needed
            except ImportError:
                logger.warning("Import analyzer not available. Install required dependencies.")
        
        if use_openai and not self.openai_api_key:
            logger.warning("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass it to the constructor.")
            self.use_openai = False
    
    def predict_relevant_files(self, pr_data: Dict, limit: int = 5) -> List[str]:
        """
        Predict relevant but unmodified files for a PR.
        
        Args:
            pr_data: PR data including title, body, and changed files
            limit: Maximum number of files to return
            
        Returns:
            List of file paths that are related but weren't changed
        """
        changed_files = self._get_changed_files(pr_data)
        
        if not changed_files:
            return []
        
        # Try OpenAI method first if enabled
        if self.use_openai and self.openai_api_key:
            try:
                logger.info(f"Using OpenAI to predict relevant files for PR #{pr_data.get('pr_number', 'unknown')}")
                files = self._predict_with_openai(pr_data, changed_files, limit)
                if files:
                    return files
                logger.info("OpenAI prediction returned no results, falling back to heuristic methods")
            except Exception as e:
                logger.error(f"Error using OpenAI for prediction: {e}")
                logger.info("Falling back to heuristic methods")
        
        # Use import analysis if available
        if self.use_import_analysis and self.import_analyzer:
            try:
                if not hasattr(self.import_analyzer, '_graph_built') or not self.import_analyzer._graph_built:
                    logger.info("Building import graph for repository...")
                    self.import_analyzer.build_import_graph()
                    self.import_analyzer._graph_built = True
                
                logger.info(f"Using import analysis to predict relevant files for PR #{pr_data.get('pr_number', 'unknown')}")
                import_related_files = self._predict_with_import_analysis(changed_files, limit)
                
                if import_related_files:
                    return import_related_files
                logger.info("Import analysis returned no results, falling back to basic heuristics")
            except Exception as e:
                logger.error(f"Error using import analysis for prediction: {e}")
                logger.info("Falling back to basic heuristics")
        
        # Fall back to basic heuristic method
        if self.heuristic_predictor:
            logger.info(f"Using basic heuristics to predict relevant files for PR #{pr_data.get('pr_number', 'unknown')}")
            return self.heuristic_predictor.find_related_files(changed_files, limit)
        else:
            logger.warning("No prediction method available. Either provide a repo_path or set use_openai=True with an API key.")
            return []
            
    def _predict_with_import_analysis(self, changed_files: List[str], limit: int = 5) -> List[str]:
        """
        Use import analysis to predict relevant files.
        
        Args:
            changed_files: List of files changed in the PR
            limit: Maximum number of files to return
            
        Returns:
            List of predicted relevant file paths
        """
        if not self.import_analyzer:
            return []
        
        # Collect related files from import analysis for each changed file
        all_related_files = set()
        for file in changed_files:
            if file.endswith(".py"):  # Only analyze Python files
                try:
                    related = self.import_analyzer.find_related_files(file, depth=1)
                    all_related_files.update(related)
                except Exception as e:
                    logger.error(f"Error finding related files for {file}: {e}")
        
        # Remove files that were already changed
        all_related_files = all_related_files - set(changed_files)
        
        # Sort by relevance (we could implement a more sophisticated ranking here)
        # For now, we'll just prioritize files with shorter paths as they're often more core
        related_files_list = sorted(all_related_files, key=lambda x: len(Path(x).parts))
        
        return related_files_list[:limit]
    
    def predict_batch(self, pr_data_list: List[Dict], limit: int = 5) -> Dict[int, List[str]]:
        """
        Predict relevant files for multiple PRs in parallel.
        
        Args:
            pr_data_list: List of PR data dictionaries
            limit: Maximum number of files to return per PR
            
        Returns:
            Dictionary mapping PR numbers to lists of relevant files
        """
        results = {}
        
        if self.use_openai and self.openai_api_key:
            # Use ThreadPoolExecutor for parallel API requests
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pr = {
                    executor.submit(self._predict_with_openai, pr_data, self._get_changed_files(pr_data), limit): 
                    pr_data["pr_number"] 
                    for pr_data in pr_data_list
                }
                
                for future in as_completed(future_to_pr):
                    pr_number = future_to_pr[future]
                    try:
                        results[pr_number] = future.result()
                    except Exception as e:
                        logger.error(f"Error predicting relevant files for PR #{pr_number}: {e}")
                        results[pr_number] = []
        else:
            # Use heuristic method sequentially
            for pr_data in pr_data_list:
                pr_number = pr_data["pr_number"]
                try:
                    results[pr_number] = self.predict_relevant_files(pr_data, limit)
                except Exception as e:
                    logger.error(f"Error predicting relevant files for PR #{pr_number}: {e}")
                    results[pr_number] = []
        
        return results
    
    def _get_changed_files(self, pr_data: Dict) -> List[str]:
        """Extract the list of changed file paths from PR data."""
        changed_files = []
        
        for file in pr_data.get("code_files", []):
            filename = file.get("filename", "")
            if filename:
                changed_files.append(filename)
        
        return changed_files
    
    def _predict_with_openai(self, pr_data: Dict, changed_files: List[str], limit: int = 5) -> List[str]:
        """
        Use OpenAI API to predict relevant files.
        
        Args:
            pr_data: PR data including title, body
            changed_files: List of files changed in the PR
            limit: Maximum number of files to predict
            
        Returns:
            List of predicted relevant file paths
        """
        if not self.openai_api_key:
            logger.error("OpenAI API key is required for LLM-based prediction")
            return []
        
        # Construct prompt for the API
        title = pr_data.get("title", "")
        body = pr_data.get("body", "")
        
        prompt = f"""Given a GitHub pull request, predict which files are relevant to understanding the changes but were NOT modified.

PR Title: {title}

PR Description:
{body}

Files that were modified in this PR:
{', '.join(changed_files)}

Based on the PR title, description, and modified files, list up to {limit} files that are likely relevant to understanding these changes but were not modified.
Return your answer as a JSON array of file paths, like this: ["path/to/file1.py", "path/to/file2.py"]
"""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that predicts which source code files are relevant to understanding a pull request."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,  # Low temperature for more deterministic outputs
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Try to extract a JSON array from the content
                matches = re.search(r'\[.*?\]', content, re.DOTALL)
                if matches:
                    json_str = matches.group(0)
                    try:
                        file_list = json.loads(json_str)
                        if isinstance(file_list, list):
                            return file_list[:limit]
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from API response: {json_str}")
                
                logger.warning(f"Couldn't extract file list from API response: {content}")
                return []
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return []

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    # Example PR data
    example_pr = {
        "pr_number": 123,
        "title": "Fix bug in data processing",
        "body": "This PR fixes a critical issue in the data processing pipeline that was causing crashes when handling null inputs.",
        "code_files": [
            {"filename": "src/data_processor.py"},
            {"filename": "tests/test_processor.py"}
        ]
    }
    
    # Test with heuristic method
    print("Testing heuristic method...")
    repo_path = Path("./example_repo")  # Replace with actual repo path
    if repo_path.exists():
        predictor = RelevantFilesPredictor(repo_path=repo_path)
        relevant_files = predictor.predict_relevant_files(example_pr)
        print(f"Predicted relevant files: {relevant_files}")
    
    # Test with OpenAI (if API key is available)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print("Testing OpenAI method...")
        predictor = RelevantFilesPredictor(use_openai=True, openai_api_key=api_key)
        relevant_files = predictor.predict_relevant_files(example_pr)
        print(f"Predicted relevant files: {relevant_files}")