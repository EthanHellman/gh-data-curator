# src/dataflow/filtering/content_relevance_filter.py
import re
from typing import Dict, Tuple, List, Any, Set

class ContentRelevanceFilter:
    """
    Filter PRs based on content relevance and quality.
    
    This filter assesses whether a PR contains meaningful software engineering
    content and problem-solving, distinguishing substantive changes from
    trivial ones.
    """
    
    def __init__(self):
        # Patterns that indicate meaningful problem solving
        self.problem_solving_patterns = [
            r'fix(ed|es|ing)?(\s+a)?\s+(bug|issue|crash|problem|error)',
            r'solv(ed|es|ing)(\s+a)?\s+(bug|issue|crash|problem|error)',
            r'improv(ed|es|ing)\s+(performance|quality|functionality)',
            r'implement(ed|s|ing)\s+(feature|functionality)',
            r'refactor(ed|s|ing)',
            r'optimize(d|s|ing)'
        ]
        
        # Patterns indicating significant code changes
        self.code_change_indicators = [
            r'if\s*\(', r'for\s*\(', r'while\s*\(', r'switch\s*\(',  # Control structures
            r'function\s+\w+', r'def\s+\w+', r'class\s+\w+',  # Definitions
            r'return\s+', r'yield\s+',  # Returns
            r'new\s+\w+', r'this\.', r'self\.',  # Object-oriented
            r'try\s*{', r'catch\s*\(', r'except',  # Error handling
            r'import\s+', r'require\(',  # Imports
        ]
        
        # File types to exclude from relevance analysis
        self.ignored_file_types = [
            ".md", ".rst", ".txt", ".gitignore", ".lock", 
            ".min.js", ".min.css", ".svg", ".png", ".jpg", ".ico"
        ]
        
        # Low-relevance change patterns
        self.low_relevance_patterns = [
            r'fix(ed|es|ing)?\s+typo',
            r'updat(ed|es|ing)\s+version',
            r'bump(ed|ing)?\s+version',
            r'updat(ed|es|ing)\s+(readme|documentation)',
            r'add(ed|s|ing)\s+comment',
            r'remov(ed|es|ing)\s+comment'
        ]
    
    def apply(self, pr_data: Dict) -> Tuple[bool, Dict]:
        """
        Apply content relevance filtering to a PR.
        
        Returns:
            Tuple[bool, Dict]: (passed_filter, metadata)
        """
        # Extract content data
        title = pr_data.get("title", "")
        body = pr_data.get("body", "")
        is_bug_fix = pr_data.get("is_bug_fix", False)
        code_files = pr_data.get("code_files", [])
        
        # Check for problem-solving indicators in title and body
        problem_solving_indicators = []
        for pattern in self.problem_solving_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                problem_solving_indicators.append(f"Title: {pattern}")
            if body and re.search(pattern, body, re.IGNORECASE):
                problem_solving_indicators.append(f"Body: {pattern}")
        
        # Check for low-relevance indicators
        low_relevance_indicators = []
        for pattern in self.low_relevance_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                low_relevance_indicators.append(f"Title: {pattern}")
            if body and re.search(pattern, body, re.IGNORECASE):
                low_relevance_indicators.append(f"Body: {pattern}")
        
        # Analyze code changes in patches
        code_change_indicators = []
        relevant_file_count = 0
        total_patch_lines = 0
        
        for file in code_files:
            filename = file.get("filename", "")
            patch = file.get("patch", "")
            
            # Skip ignored file types
            if any(filename.endswith(ext) for ext in self.ignored_file_types):
                continue
            
            relevant_file_count += 1
            
            # Analyze patch for code change indicators
            if patch:
                total_patch_lines += len(patch.splitlines())
                
                for pattern in self.code_change_indicators:
                    if re.search(pattern, patch, re.IGNORECASE):
                        code_change_indicators.append(f"{filename}: {pattern}")
        
        # Calculate relevance scores
        problem_solving_score = min(1.0, len(problem_solving_indicators) * 0.2)
        code_quality_score = min(1.0, len(code_change_indicators) * 0.1)
        low_relevance_penalty = min(1.0, len(low_relevance_indicators) * 0.3)
        
        # Bonus for bug fixes
        bug_fix_bonus = 0.3 if is_bug_fix else 0.0
        
        # Combined relevance score
        relevance_score = (
            problem_solving_score + 
            code_quality_score + 
            bug_fix_bonus - 
            low_relevance_penalty
        )
        relevance_score = max(0.0, min(1.0, relevance_score))  # Clamp to [0, 1]
        
        # Generate metadata
        metadata = {
            "problem_solving_indicators": problem_solving_indicators,
            "code_change_indicators": code_change_indicators,
            "low_relevance_indicators": low_relevance_indicators,
            "relevant_file_count": relevant_file_count,
            "total_patch_lines": total_patch_lines,
            "problem_solving_score": problem_solving_score,
            "code_quality_score": code_quality_score,
            "bug_fix_bonus": bug_fix_bonus,
            "low_relevance_penalty": low_relevance_penalty,
            "relevance_score": relevance_score
        }
        
        # Determine if PR passes the filter
        passed = relevance_score >= 0.5 and relevant_file_count > 0
        
        return passed, metadata