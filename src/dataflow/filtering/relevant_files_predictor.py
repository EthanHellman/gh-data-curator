# src/dataflow/filtering/relevant_files_predictor.py
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
import json
import requests
import traceback
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
        
        # Enhanced logging for initialization
        logger.info(f"Initializing RelevantFilesPredictor with:")
        logger.info(f"  - repo_path: {repo_path}")
        logger.info(f"  - use_openai: {use_openai}")
        logger.info(f"  - has_api_key: {bool(self.openai_api_key)}")
        logger.info(f"  - use_import_analysis: {use_import_analysis}")
        
        # Initialize heuristic-based file relationship predictor
        from dataflow.filtering.file_relationship import RelatedFilePredictor
        self.heuristic_predictor = RelatedFilePredictor(repo_path) if repo_path else None
        logger.info(f"  - heuristic_predictor: {'initialized' if self.heuristic_predictor else 'None'}")
        
        # Initialize import analyzer if requested and repository is available
        self.import_analyzer = None
        if use_import_analysis and repo_path and repo_path.exists():
            try:
                from dataflow.filtering.import_analyzer import ImportAnalyzer
                self.import_analyzer = ImportAnalyzer(repo_path)
                logger.info(f"  - import_analyzer: initialized (graph will be built on demand)")
                # Build the import graph lazily when first needed
            except ImportError as e:
                logger.warning(f"Import analyzer not available: {e}")
                logger.warning("Install required dependencies.")
        else:
            reason = "disabled" if not use_import_analysis else "no valid repo_path"
            logger.info(f"  - import_analyzer: not initialized ({reason})")
        
        if use_openai and not self.openai_api_key:
            logger.warning("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass it to the constructor.")
            self.use_openai = False
            logger.info("  - use_openai has been disabled due to missing API key")
    
    def predict_relevant_files(self, pr_data: Dict, limit: int = 5) -> List[str]:
        """
        Predict relevant but unmodified files for a PR.
        
        Args:
            pr_data: PR data including title, body, and changed files
            limit: Maximum number of files to return
            
        Returns:
            List of file paths that are related but weren't changed
        """
        pr_number = pr_data.get("pr_number", "unknown")
        logger.info(f"[PR #{pr_number}] Starting prediction of relevant files")
        logger.info(f"[PR #{pr_number}] Settings: use_openai={self.use_openai}, has_api_key={bool(self.openai_api_key)}, use_import_analysis={self.use_import_analysis}")

        changed_files = self._get_changed_files(pr_data)
        logger.info(f"[PR #{pr_number}] Found {len(changed_files)} changed files")
        
        if not changed_files:
            logger.warning(f"[PR #{pr_number}] No changed files found, returning empty list")
            return []
        
        # Try OpenAI method first if enabled
        if self.use_openai and self.openai_api_key:
            try:
                logger.info(f"[PR #{pr_number}] Attempting prediction with OpenAI")
                files = self._predict_with_openai(pr_data, changed_files, limit)
                if files:
                    logger.info(f"[PR #{pr_number}] OpenAI prediction successful: found {len(files)} relevant files")
                    return files
                logger.info(f"[PR #{pr_number}] OpenAI prediction returned no results, falling back to other methods")
            except Exception as e:
                logger.error(f"[PR #{pr_number}] Error using OpenAI for prediction: {e}")
                logger.info(f"[PR #{pr_number}] Falling back to other methods")
                logger.debug(f"[PR #{pr_number}] Exception details: {traceback.format_exc()}")
        else:
            logger.info(f"[PR #{pr_number}] Skipping OpenAI prediction (not enabled or no API key)")
        
        # Use import analysis if available
        if self.use_import_analysis and self.import_analyzer:
            try:
                # Build the import graph if not already built
                if not hasattr(self.import_analyzer, '_graph_built') or not self.import_analyzer._graph_built:
                    logger.info(f"[PR #{pr_number}] Building import graph for repository...")
                    self.import_analyzer.build_import_graph()
                    self.import_analyzer._graph_built = True
                    logger.info(f"[PR #{pr_number}] Import graph built successfully")
                
                logger.info(f"[PR #{pr_number}] Attempting prediction with import analysis")
                import_related_files = self._predict_with_import_analysis(pr_data, changed_files, limit)
                
                if import_related_files:
                    logger.info(f"[PR #{pr_number}] Import analysis prediction successful: found {len(import_related_files)} relevant files")
                    return import_related_files
                logger.info(f"[PR #{pr_number}] Import analysis returned no results, falling back to basic heuristics")
            except Exception as e:
                logger.error(f"[PR #{pr_number}] Error using import analysis for prediction: {e}")
                logger.info(f"[PR #{pr_number}] Falling back to basic heuristics")
                logger.debug(f"[PR #{pr_number}] Exception details: {traceback.format_exc()}")
        else:
            reason = "disabled" if not self.use_import_analysis else "import analyzer not initialized"
            logger.info(f"[PR #{pr_number}] Skipping import analysis prediction ({reason})")
        
        # Fall back to basic heuristic method
        if self.heuristic_predictor:
            logger.info(f"[PR #{pr_number}] Attempting prediction with basic heuristics")
            heuristic_files = self.heuristic_predictor.find_related_files(changed_files, limit)
            logger.info(f"[PR #{pr_number}] Heuristic prediction found {len(heuristic_files)} relevant files")
            return heuristic_files
        else:
            logger.warning(f"[PR #{pr_number}] No prediction method available. Either provide a repo_path or set use_openai=True with an API key.")
            return []
            
    def _predict_with_import_analysis(self, pr_data: Dict, changed_files: List[str], limit: int = 5) -> List[str]:
        """
        Use import analysis to predict relevant files.
        
        Args:
            pr_data: PR data including additional context
            changed_files: List of files changed in the PR
            limit: Maximum number of files to return
            
        Returns:
            List of predicted relevant file paths
        """
        pr_number = pr_data.get("pr_number", "unknown")
        if not self.import_analyzer:
            logger.warning(f"[PR #{pr_number}] Import analyzer not available")
            return []
        
        # Collect related files from import analysis for each changed file
        all_related_files = set()
        for file in changed_files:
            if file.endswith(".py"):  # Only analyze Python files
                try:
                    logger.info(f"[PR #{pr_number}] Finding related files for {file} using import analysis")
                    related = self.import_analyzer.find_related_files(file, depth=1)
                    logger.info(f"[PR #{pr_number}] Found {len(related)} related files for {file}")
                    
                    # Log a few examples of related files
                    if related:
                        examples = list(related)[:3]
                        logger.info(f"[PR #{pr_number}] Examples of related files for {file}: {examples}")
                    
                    all_related_files.update(related)
                except Exception as e:
                    logger.error(f"[PR #{pr_number}] Error finding related files for {file}: {e}")
        
        # Remove files that were already changed
        before_filtering = len(all_related_files)
        all_related_files = all_related_files - set(changed_files)
        logger.info(f"[PR #{pr_number}] After removing changed files: {before_filtering} -> {len(all_related_files)}")
        
        # Sort by relevance (we could implement a more sophisticated ranking here)
        # For now, we'll just prioritize files with shorter paths as they're often more core
        related_files_list = sorted(all_related_files, key=lambda x: len(Path(x).parts))
        
        result = related_files_list[:limit]
        logger.info(f"[PR #{pr_number}] Final import analysis prediction: {result}")
        return result
    
    def predict_batch(self, pr_data_list: List[Dict], limit: int = 5) -> Dict[int, List[str]]:
        """
        Predict relevant files for multiple PRs in parallel.
        
        Args:
            pr_data_list: List of PR data dictionaries
            limit: Maximum number of files to return per PR
            
        Returns:
            Dictionary mapping PR numbers to lists of relevant files
        """
        logger.info(f"Starting batch prediction for {len(pr_data_list)} PRs")
        results = {}
        
        # Pre-build the import graph if we're using import analysis and have a repository
        if self.use_import_analysis and self.import_analyzer and self.repo_path and self.repo_path.exists():
            try:
                if not hasattr(self.import_analyzer, '_graph_built') or not self.import_analyzer._graph_built:
                    logger.info(f"Building import graph for repository before batch prediction...")
                    self.import_analyzer.build_import_graph()
                    self.import_analyzer._graph_built = True
                    logger.info(f"Import graph built successfully")
            except Exception as e:
                logger.error(f"Error building import graph: {e}")
        
        if self.use_openai and self.openai_api_key:
            logger.info(f"Using ThreadPoolExecutor with {self.max_workers} workers for OpenAI batch prediction")
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
                        logger.info(f"[PR #{pr_number}] Batch prediction completed successfully")
                    except Exception as e:
                        logger.error(f"[PR #{pr_number}] Error in batch prediction: {e}")
                        logger.debug(f"[PR #{pr_number}] Exception details: {traceback.format_exc()}")
                        results[pr_number] = []
        else:
            logger.info("Using sequential processing for batch prediction")
            # Use heuristic method sequentially
            for pr_data in pr_data_list:
                pr_number = pr_data["pr_number"]
                try:
                    results[pr_number] = self.predict_relevant_files(pr_data, limit)
                    logger.info(f"[PR #{pr_number}] Sequential batch prediction completed")
                except Exception as e:
                    logger.error(f"[PR #{pr_number}] Error in sequential batch prediction: {e}")
                    results[pr_number] = []
        
        logger.info(f"Batch prediction completed for {len(results)} PRs")
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
        pr_number = pr_data.get("pr_number", "unknown")
        if not self.openai_api_key:
            logger.error(f"[PR #{pr_number}] OpenAI API key is required for LLM-based prediction")
            return []
        
        logger.info(f"[PR #{pr_number}] Preparing OpenAI API request")

        # Construct prompt for the API
        title = pr_data.get("title", "")
        body = pr_data.get("body", "")
        
        # Generate context from various sources
        import_context = self._generate_import_context(pr_number, changed_files)
        ast_context = self._generate_ast_summary(pr_number, changed_files)
        repo_context = self._generate_repo_structure(pr_number)
        
        # Combine all context (but be mindful of token limits)
        all_context = []
        if import_context:
            all_context.append(import_context)
        if ast_context:
            all_context.append(ast_context)
        if repo_context:
            all_context.append(repo_context)
        
        # Join all context with empty lines between sections
        combined_context = "\n\n".join(all_context)
        
        prompt = f"""Given a GitHub pull request, predict which files are relevant to understanding the changes but were NOT modified.

PR Title: {title}

PR Description:
{body}

Files that were modified in this PR:
{', '.join(changed_files)}

{combined_context}

Based on the PR title, description, modified files, and code relationships described above, list up to {limit} files that are likely relevant to understanding these changes but were not modified.
Return your answer as a JSON array of file paths, like this: ["path/to/file1.py", "path/to/file2.py"]
"""
        
        logger.info(f"[PR #{pr_number}] Sending request to OpenAI API")
        logger.debug(f"[PR #{pr_number}] Prompt: {prompt}")
        
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

                logger.info(f"[PR #{pr_number}] OpenAI API call successful")
                logger.debug(f"[PR #{pr_number}] Raw response: {content}")
                
                # Try to extract a JSON array from the content
                matches = re.search(r'\[.*?\]', content, re.DOTALL)
                if matches:
                    json_str = matches.group(0)
                    try:
                        file_list = json.loads(json_str)
                        if isinstance(file_list, list):
                            logger.info(f"[PR #{pr_number}] Successfully parsed relevant files from API response: {file_list[:limit]}")
                            return file_list[:limit]
                    except json.JSONDecodeError as e:
                        logger.error(f"[PR #{pr_number}] Failed to parse JSON from API response: {e}")
                        logger.error(f"[PR #{pr_number}] JSON string that couldn't be parsed: {json_str}")
                
                logger.warning(f"[PR #{pr_number}] Couldn't extract file list from API response")
                logger.debug(f"[PR #{pr_number}] Full response content: {content}")
                return []
            else:
                logger.error(f"[PR #{pr_number}] OpenAI API error: {response.status_code}")
                logger.error(f"[PR #{pr_number}] Error details: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"[PR #{pr_number}] Error calling OpenAI API: {e}")
            logger.debug(f"[PR #{pr_number}] Exception details: {traceback.format_exc()}")
            return []
    
    def _generate_import_context(self, pr_number: str, changed_files: List[str]) -> str:
        """Generate import relationship context for changed files."""
        if not self.import_analyzer or not hasattr(self.import_analyzer, '_graph_built') or not self.import_analyzer._graph_built:
            logger.info(f"[PR #{pr_number}] Import analyzer not available for context generation")
            return ""
        
        import_relationships = []
        for file in changed_files:
            if file.endswith('.py'):
                try:
                    # Files that this file imports
                    imports = list(self.import_analyzer.import_graph.get(file, []))[:10]  # Limit to 10 imports
                    
                    # Files that import this file
                    imported_by = list(self.import_analyzer.reverse_import_graph.get(file, []))[:10]  # Limit to 10
                    
                    if imports or imported_by:
                        file_info = f"File: {file}\n"
                        if imports:
                            file_info += f"  Imports: {', '.join(imports)}\n"
                        if imported_by:
                            file_info += f"  Imported by: {', '.join(imported_by)}\n"
                        import_relationships.append(file_info)
                except Exception as e:
                    logger.error(f"[PR #{pr_number}] Error generating import context for {file}: {e}")
        
        if import_relationships:
            logger.info(f"[PR #{pr_number}] Generated import context for {len(import_relationships)} files")
            return "Import relationships for modified files:\n" + "\n".join(import_relationships)
        
        logger.info(f"[PR #{pr_number}] No import relationships found for context")
        return ""
    
    def _generate_ast_summary(self, pr_number: str, changed_files: List[str], max_files: int = 3) -> str:
        """
        Generate an AST-based summary of the changed files.
        
        Args:
            pr_number: PR number for logging
            changed_files: List of files that were changed in the PR
            max_files: Maximum number of files to analyze
            
        Returns:
            A string with the AST summary or empty string if not available
        """
        if not self.repo_path or not changed_files:
            logger.info(f"[PR #{pr_number}] Cannot generate AST summary: no repo path or no changed files")
            return ""
        
        try:
            import ast
            
            logger.info(f"[PR #{pr_number}] Generating AST summary for up to {max_files} changed files")
            
            # Only process Python files
            py_files = [f for f in changed_files if f.endswith('.py')][:max_files]
            if not py_files:
                logger.info(f"[PR #{pr_number}] No Python files to analyze in changed files")
                return ""
            
            summaries = []
            for file_path in py_files:
                full_path = self.repo_path / file_path
                if not full_path.exists():
                    logger.warning(f"[PR #{pr_number}] File not found for AST analysis: {file_path}")
                    continue
                    
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                        
                    tree = ast.parse(code)
                    
                    # Extract classes and functions
                    classes = []
                    functions = []
                    imports = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Collect methods for this class
                            methods = []
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    methods.append(item.name)
                            
                            if methods:
                                classes.append(f"{node.name} (methods: {', '.join(methods[:5])}{'...' if len(methods) > 5 else ''})")
                            else:
                                classes.append(node.name)
                        elif isinstance(node, ast.FunctionDef):
                            # Check if this is a top-level function (not a method)
                            # This is a bit tricky with ast, so we'll use a simpler approach
                            if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) if hasattr(parent, 'body') and node in parent.body):
                                functions.append(node.name)
                        elif isinstance(node, ast.Import):
                            for name in node.names:
                                imports.append(name.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                # Handle 'from x import y'
                                module = node.module
                                for name in node.names:
                                    imports.append(f"{module}.{name.name}")
                    
                    summary = f"File: {file_path}\n"
                    if imports:
                        summary += f"  Imports: {', '.join(imports[:10])}{'...' if len(imports) > 10 else ''}\n"
                    if classes:
                        summary += f"  Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}\n"
                    if functions:
                        summary += f"  Functions: {', '.join(functions[:10])}{'...' if len(functions) > 10 else ''}\n"
                        
                    summaries.append(summary)
                    logger.info(f"[PR #{pr_number}] Generated AST summary for {file_path}")
                
                except Exception as e:
                    logger.error(f"[PR #{pr_number}] Error parsing {file_path}: {e}")
                    logger.debug(f"[PR #{pr_number}] Traceback: {traceback.format_exc()}")
            
            if summaries:
                ast_summary = "AST analysis of changed files:\n" + "\n".join(summaries)
                logger.info(f"[PR #{pr_number}] Successfully generated AST summaries for {len(summaries)} files")
                return ast_summary
            
            logger.info(f"[PR #{pr_number}] No AST summaries generated")
            return ""
        
        except ImportError:
            logger.warning(f"[PR #{pr_number}] AST module not available")
            return ""
        except Exception as e:
            logger.error(f"[PR #{pr_number}] Error in AST analysis: {e}")
            logger.debug(f"[PR #{pr_number}] Traceback: {traceback.format_exc()}")
            return ""
    
    def _generate_repo_structure(self, pr_number: str) -> str:
        """
        Generate a summary of the repository structure.
        
        Args:
            pr_number: PR number for logging
            
        Returns:
            A string with the repository structure or empty string if not available
        """
        if not self.repo_path or not self.repo_path.exists():
            logger.info(f"[PR #{pr_number}] Cannot generate repository structure: no valid repo path")
            return ""
        
        try:
            logger.info(f"[PR #{pr_number}] Generating repository structure summary")
            
            # Find Python packages (directories with __init__.py)
            packages = set()
            for init_file in self.repo_path.glob("**/__init__.py"):
                package_dir = init_file.parent
                try:
                    # Get relative path to repository root
                    package_path = package_dir.relative_to(self.repo_path)
                    packages.add(str(package_path))
                except ValueError:
                    # Skip if not relative to repo_path
                    continue
            
            # Get directory structure with file counts
            dir_counts = {}
            for pkg in sorted(packages):
                pkg_path = self.repo_path / pkg
                python_files = list(pkg_path.glob("*.py"))
                dir_counts[pkg] = len(python_files)
            
            # Format the structure
            lines = ["Repository structure:"]
            
            # Add top-level packages with file counts
            for pkg, count in sorted(dir_counts.items(), key=lambda x: x[0]):
                pkg_parts = pkg.split('/')
                
                # Only add top-level packages and their immediate children
                if len(pkg_parts) <= 2:
                    indent = "  " * (len(pkg_parts) - 1)
                    lines.append(f"{indent}- {pkg_parts[-1]}: {count} Python files")
            
            structure = "\n".join(lines)
            logger.info(f"[PR #{pr_number}] Generated repository structure summary with {len(dir_counts)} packages")
            return structure
        
        except Exception as e:
            logger.error(f"[PR #{pr_number}] Error generating repository structure: {e}")
            logger.debug(f"[PR #{pr_number}] Traceback: {traceback.format_exc()}")
            return ""

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