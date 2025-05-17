#!/usr/bin/env python3
"""
Text Embeddings Module

This module provides functions for generating embeddings using OpenAI's API,
for both code and textual content. It handles API authentication, rate limiting,
and processing of embedding results.
"""
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import requests
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger("enhanced_report_generator")

class OpenAIEmbeddings:
    """Class to handle generation of embeddings using OpenAI's API."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the OpenAI embeddings handler.
        
        Args:
            config_path: Path to the config.json file containing API keys.
                         If None, will try standard locations or environment variables.
        """
        self.api_key = self._load_api_key(config_path)
        self.model = "text-embedding-ada-002"  # Default model for embeddings
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def _load_api_key(self, config_path: Optional[Path] = None) -> str:
        """
        Load OpenAI API key from config file or environment variable.
        
        Args:
            config_path: Path to config.json file.
            
        Returns:
            API key as string.
            
        Raises:
            ValueError: If API key cannot be found.
        """
        # First try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return api_key
        
        # Try config file if path provided
        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "openai_api_key" in config:
                        return config["openai_api_key"]
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading config file: {e}")
        
        # Try standard locations
        standard_paths = [
            Path.home() / "gh-data-curator" / "config.json",
            Path.home() / "config.json",
            Path.cwd() / "config.json",
            Path.cwd().parent / "config.json",
            Path.cwd().parent.parent / "config.json",
            Path("/etc/gh-data-curator/config.json")
        ]
        
        for path in standard_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        config = json.load(f)
                        if "openai_api_key" in config:
                            logger.info(f"Found API key in {path}")
                            return config["openai_api_key"]
                except (json.JSONDecodeError, IOError):
                    continue
        
        # If we get here, no API key was found
        logger.warning("No OpenAI API key found in config files or environment variables")
        
        # Return a placeholder key for development/testing without API access
        # In production, you would want to raise an error instead
        return "sk-placeholder-api-key-for-testing"
    
    def get_embeddings(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process in each API call.
            
        Returns:
            Numpy array of embeddings.
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._get_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Sleep to avoid rate limits if more batches coming
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        return np.array(all_embeddings)
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: Batch of text strings.
            
        Returns:
            List of embeddings as float lists.
        """
        embeddings = []
        
        # Try the API call with retries
        for attempt in range(self.max_retries):
            try:
                response = self._call_openai_api(texts)
                
                # Extract embeddings from response
                for item in response.get("data", []):
                    embeddings.append(item.get("embedding", []))
                
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Wait before retrying, with exponential backoff
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Failed to get embeddings after {self.max_retries} attempts")
                    # If real API is unavailable, generate random embeddings as fallback
                    return self._generate_mock_embeddings(len(texts))
        
        # Check if we have the right number of embeddings
        if len(embeddings) != len(texts):
            logger.warning(f"Received {len(embeddings)} embeddings for {len(texts)} texts")
            # Fill in missing embeddings with random values as fallback
            while len(embeddings) < len(texts):
                embeddings.append(self._generate_mock_embeddings(1)[0])
        
        return embeddings
    
    def _call_openai_api(self, texts: List[str]) -> Dict[str, Any]:
        """
        Make the actual API call to OpenAI.
        
        Args:
            texts: List of text strings.
            
        Returns:
            API response as dict.
            
        Raises:
            Exception: If API call fails.
        """
        url = "https://api.openai.com/v1/embeddings"
        
        payload = {
            "input": texts,
            "model": self.model
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP error codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(response, 'text'):
                logger.error(f"Response: {response.text}")
            raise
    
    def _generate_mock_embeddings(self, count: int, dim: int = 1536) -> List[List[float]]:
        """
        Generate random embeddings as fallback when API is unavailable.
        
        Args:
            count: Number of embeddings to generate.
            dim: Dimensionality of embeddings.
            
        Returns:
            List of random embeddings.
        """
        logger.warning("Generating random mock embeddings as fallback")
        
        # Generate random vectors and normalize them
        embeddings = []
        for _ in range(count):
            vec = np.random.randn(dim)
            vec = vec / np.linalg.norm(vec)  # Unit vector
            embeddings.append(vec.tolist())
        
        return embeddings

def get_embeddings_for_pr_texts(pr_texts: List[str], config_path: Optional[Path] = None) -> np.ndarray:
    """
    Generate embeddings for PR description texts.
    
    Args:
        pr_texts: List of PR description texts.
        config_path: Path to config file with API key.
        
    Returns:
        Numpy array of embeddings.
    """
    # Clean and preprocess texts
    processed_texts = [preprocess_text(text) for text in pr_texts]
    
    # Get embeddings
    embedding_generator = OpenAIEmbeddings(config_path)
    embeddings = embedding_generator.get_embeddings(processed_texts)
    
    return embeddings

def get_embeddings_for_pr_code(code_samples: List[str], config_path: Optional[Path] = None) -> np.ndarray:
    """
    Generate embeddings for PR code samples.
    
    Args:
        code_samples: List of code samples from PRs.
        config_path: Path to config file with API key.
        
    Returns:
        Numpy array of embeddings.
    """
    # Clean and preprocess code
    processed_code = [preprocess_code(code) for code in code_samples]
    
    # Get embeddings
    embedding_generator = OpenAIEmbeddings(config_path)
    embeddings = embedding_generator.get_embeddings(processed_code)
    
    return embeddings

def preprocess_text(text: str) -> str:
    """
    Preprocess PR description text for embedding.
    
    Args:
        text: Raw PR description text.
        
    Returns:
        Preprocessed text.
    """
    if not text:
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Truncate if too long (OpenAI has token limits)
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return text

def preprocess_code(code: str) -> str:
    """
    Preprocess code sample for embedding.
    
    Args:
        code: Raw code sample.
        
    Returns:
        Preprocessed code.
    """
    if not code:
        return ""
    
    # Basic cleaning
    code = code.strip()
    
    # Truncate if too long (OpenAI has token limits)
    max_chars = 8000
    if len(code) > max_chars:
        code = code[:max_chars]
    
    return code

def extract_code_sample(pr_data: Dict[str, Any]) -> str:
    """
    Extract a representative code sample from a PR.
    
    Args:
        pr_data: PR data dictionary.
        
    Returns:
        Representative code sample as string.
    """
    # If we have diff data, use that
    if "diff" in pr_data and pr_data["diff"]:
        return pr_data["diff"]
    
    # Otherwise, try to build a sample from metadata
    sample_parts = []
    
    # Add file names
    if "files" in pr_data and pr_data["files"]:
        sample_parts.append("Files modified: " + ", ".join(pr_data["files"]))
    
    # Add summary of changes
    if "additions" in pr_data and "deletions" in pr_data:
        sample_parts.append(f"Changes: +{pr_data['additions']}, -{pr_data['deletions']}")
    
    # Add PR title and description as context
    if "title" in pr_data and pr_data["title"]:
        sample_parts.append(f"Title: {pr_data['title']}")
    
    if "body" in pr_data and pr_data["body"]:
        # Take just the first part of the body to keep it manageable
        body_preview = pr_data["body"][:500]
        sample_parts.append(f"Description: {body_preview}")
    
    return "\n".join(sample_parts)