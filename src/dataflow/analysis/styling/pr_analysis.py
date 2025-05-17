#!/usr/bin/env python3
"""
PR Analysis Module

This module provides functions for analyzing PRs using clustering and embeddings.
It includes code for generating embeddings from PR content, dimensionality reduction,
and visualizing these embeddings to identify patterns.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pathlib import Path

from dataflow.analysis.functionality.visualization_utils import (
    add_gradient_background, 
    add_gradient_line, 
    adjust_color_brightness
)
from dataflow.analysis.styling.text_embeddings import (
    get_embeddings_for_pr_texts, 
    get_embeddings_for_pr_code, 
    extract_code_sample
)

logger = logging.getLogger("enhanced_report_generator")

def add_pr_clustering(generator, pdf):
    """Add PR clustering analysis to the report with improved visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Add section title with styling
    fig.text(0.5, 0.95, "PR Clustering Analysis", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
    # Add introduction text with better spacing
    intro_text = [
        "This section presents an analysis of PRs using clustering and dimensionality reduction techniques.",
        "The following visualizations show how PRs are distributed in feature space, revealing patterns",
        "and similarities across repositories. We utilize OpenAI embeddings to analyze both code content",
        "and PR descriptions, providing deeper insights into the semantic relationships between PRs."
    ]
    
    fig.text(0.1, 0.87, "\n".join(intro_text), 
            fontsize=12, va='top', ha='left', linespacing=1.5, 
            color='#333333')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Add code embedding visualization
    add_code_embedding_visualization(generator, pdf)
    
    # Add text embedding visualization
    add_text_embedding_visualization(generator, pdf)

def add_code_embedding_visualization(generator, pdf):
    """Add a visualization of code embeddings using OpenAI and dimensionality reduction."""
    # Filter data to PRs that have been processed by at least the bot filter
    cluster_data = generator.pr_data[generator.pr_data['passed_bot_filter'] == True].copy()
    
    # Check if we have enough data
    if len(cluster_data) < 10:
        logger.warning("Not enough data for code embedding visualization")
        # Create a placeholder visualization with error message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        fig.text(0.5, 0.5, "Not enough data for code embedding visualization", 
                fontsize=14, ha='center', weight='bold', color='#7f8c8d')
        fig.text(0.5, 0.45, "Need at least 10 PRs that passed bot filter", 
                fontsize=12, ha='center', color='#7f8c8d')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return
    
    # Prepare data for embedding
    try:
        # Try to find the config file
        config_path = Path.home() / "gh-data-curator" / "config.json"
        if not config_path.exists():
            config_path = Path.cwd().parent / "config.json"
        if not config_path.exists():
            config_path = None
            
        # Generate code samples for embedding
        code_samples = []
        for _, row in cluster_data.iterrows():
            # Extract code sample from PR data
            pr_data = {
                "title": row["title"],
                "body": row["body"],
                "additions": row["additions"],
                "deletions": row["deletions"],
                "files": row.get("relevant_files", [])
            }
            code_sample = extract_code_sample(pr_data)
            code_samples.append(code_sample)
        
        # Get embeddings using OpenAI
        logger.info(f"Generating code embeddings for {len(code_samples)} PRs using OpenAI")
        embeddings = get_embeddings_for_pr_code(code_samples, config_path)
        
        if len(embeddings) < len(cluster_data):
            logger.warning(f"Received {len(embeddings)} embeddings for {len(cluster_data)} PRs")
            # If embeddings are missing, switch to feature-based embedding
            use_feature_embedding = True
        else:
            use_feature_embedding = False
    except Exception as e:
        logger.error(f"Error generating code embeddings: {e}")
        # Fallback to feature-based embedding
        use_feature_embedding = True
    
    # Fallback to feature-based embedding if OpenAI embedding failed
    if use_feature_embedding:
        logger.info("Falling back to feature-based embedding")
        # Select features for code embedding
        features = [
            'file_count', 'code_file_count', 'total_changes',
            'additions', 'deletions', 'size_score', 
            'complexity_score', 'relevance_score', 
            'problem_solving_score', 'code_quality_score'
        ]
        
        # Exclude features with missing values
        valid_features = []
        for feature in features:
            if feature in cluster_data.columns and not cluster_data[feature].isnull().any():
                valid_features.append(feature)
        
        # Prepare data for PCA
        X = cluster_data[valid_features]
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        embeddings = X_scaled
    
    # Dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)
    
    # Create a DataFrame with PCA results and metadata
    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Repository': cluster_data['repository'].values,
        'PR': cluster_data['pr_number'].values,
        'Passed': cluster_data['passed_filter'].values,
        'Quality': cluster_data['quality_score'].values,
        'repo_key': cluster_data['repo_key'].values
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add subtle grid lines
    ax.grid(linestyle='--', alpha=0.3)
    
    # Create scatter plot with custom styling, coloring by repository instead of cluster
    for repo_key in cluster_data['repo_key'].unique():
        repo_points = pca_df[pca_df['repo_key'] == repo_key]
        
        # Draw points with proper styling
        ax.scatter(
            repo_points['PCA1'], 
            repo_points['PCA2'],
            s=repo_points['Quality'] * 100 + 30,  # Size based on quality score
            color=generator.repository_colors.get(repo_key, 'gray'),
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5,
            label=repo_key.replace('_', '/')
        )
    
    # Create a repository legend with better positioning
    repo_legend = plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        title="Repositories",
        frameon=True,
        fontsize=9
    )
    
    # Enhance chart appearance
    title = 'PR Code Semantic Embedding Visualization'
    ax.set_xlabel('Dimension 1', fontsize=12, weight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    
    # Add simplified explanation
    explanation_text = "Using OpenAI code embeddings to capture semantic relationships between PRs. Point size indicates quality score."
    
    fig.text(0.5, 0.05, explanation_text, ha='center', fontsize=10, style='italic',
            color='#333333')
    
    plt.tight_layout(rect=[0, 0.08, 0.85, 0.95])  # Adjust for title and legend
    
    # Save to figures directory for reference
    plt.savefig(generator.figures_dir / "pr_code_embedding.png", dpi=300)
    
    # Add to PDF
    plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_text_embedding_visualization(generator, pdf):
    """Add a visualization of PR description embeddings using OpenAI and dimensionality reduction."""
    # Step 1: Filter data to PRs that passed the bot filter and combine title and body text
    text_data = generator.pr_data[generator.pr_data['passed_bot_filter'] == True].copy()
    
    # Combine title and body for each PR to increase available text
    text_data['combined_text'] = text_data.apply(
        lambda row: f"{row['title'] or ''} {row['body'] or ''}", 
        axis=1
    )
    
    # Filter PRs with some meaningful text
    text_data = text_data[text_data['combined_text'].str.len() > 10]
    
    # Check if we have enough data
    if len(text_data) < 10:
        logger.warning("Not enough text data for embedding visualization, attempting to retrieve more from raw files")
        
        # Try to retrieve more text data from raw files
        for idx, row in generator.pr_data[generator.pr_data['passed_bot_filter'] == True].iterrows():
            if pd.isna(row['body']) or len(row['body'] or '') < 20:
                # Try to get body from raw files
                pr_id = row['pr_number']
                repo_key = row['repo_key']
                
                # Construct path to raw PR data
                raw_path = generator.data_dir / "raw" / repo_key / f"pr_{pr_id}" / "details.json"
                summary_path = generator.data_dir / "raw" / repo_key / f"pr_{pr_id}" / "summary.json"
                
                try:
                    # Try details.json first
                    if raw_path.exists():
                        with open(raw_path, 'r') as f:
                            import json
                            details = json.load(f)
                            if 'body' in details and details['body']:
                                generator.pr_data.at[idx, 'body'] = details['body']
                    
                    # Try summary.json if details didn't work
                    if (pd.isna(generator.pr_data.at[idx, 'body']) or generator.pr_data.at[idx, 'body'] == '') and summary_path.exists():
                        with open(summary_path, 'r') as f:
                            import json
                            summary = json.load(f)
                            if 'body' in summary and summary['body']:
                                generator.pr_data.at[idx, 'body'] = summary['body']
                except Exception as e:
                    logger.warning(f"Error reading raw PR data for {repo_key}/pr_{pr_id}: {e}")
        
        # Retry with updated data
        text_data = generator.pr_data[generator.pr_data['passed_bot_filter'] == True].copy()
        text_data['combined_text'] = text_data.apply(
            lambda row: f"{row['title'] or ''} {row['body'] or ''}", 
            axis=1
        )
        text_data = text_data[text_data['combined_text'].str.len() > 10]
    
    # Check again if we have enough data
    if len(text_data) < 10:
        logger.warning("Still not enough text data for embedding visualization")
        # Create a placeholder visualization with error message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        fig.text(0.5, 0.5, "Not enough text data for embedding visualization", 
                fontsize=14, ha='center', weight='bold', color='#7f8c8d')
        fig.text(0.5, 0.45, "Need at least 10 PRs with sufficient text content", 
                fontsize=12, ha='center', color='#7f8c8d')
        fig.text(0.5, 0.4, "Try including more PR description data in the dataset", 
                fontsize=12, ha='center', color='#7f8c8d')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return
    
    # Step 3: Generate embeddings
    try:
        # Prepare the texts
        texts = text_data['combined_text'].tolist()
        
        # Try to find the config file
        config_path = Path.home() / "gh-data-curator" / "config.json"
        if not config_path.exists():
            config_path = Path.cwd().parent / "config.json"
        if not config_path.exists():
            config_path = None
        
        # Get embeddings using OpenAI
        logger.info(f"Generating text embeddings for {len(texts)} PR descriptions using OpenAI")
        embeddings = get_embeddings_for_pr_texts(texts, config_path)
        
        if len(embeddings) == 0 or len(embeddings) < len(texts):
            logger.warning(f"Received {len(embeddings)} embeddings for {len(texts)} texts")
            raise ValueError("Incomplete embeddings received")
            
    except Exception as e:
        logger.error(f"Error generating text embeddings: {e}")
        # If OpenAI fails, use TF-IDF as fallback
        logger.info("Falling back to TF-IDF for text embedding")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF embeddings
        vectorizer = TfidfVectorizer(
            max_features=100,  # Limit to top features
            stop_words='english',
            min_df=1,  # Allow terms that appear in at least 1 document
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        # Transform texts to TF-IDF features
        try:
            # Replace None or empty values with placeholder
            processed_texts = [t if t else "no description" for t in texts]
            embeddings = vectorizer.fit_transform(processed_texts).toarray()
        except Exception as e:
            logger.error(f"Failed to create TF-IDF embeddings: {e}")
            # Create a placeholder visualization with error message
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            fig.text(0.5, 0.5, "Failed to generate text embeddings", 
                    fontsize=14, ha='center', weight='bold', color='#7f8c8d')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
    
    # Step 4: Apply dimensionality reduction
    try:
        # Try t-SNE first for better clustering
        tsne = TSNE(n_components=2, perplexity=min(30, len(texts)-1), random_state=42)
        X_reduced = tsne.fit_transform(embeddings)
        method = "t-SNE"
    except Exception as e:
        logger.error(f"Failed to apply t-SNE: {e}")
        try:
            # Fall back to PCA if t-SNE fails
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(embeddings)
            method = "PCA"
        except Exception as e:
            logger.error(f"Failed to apply PCA: {e}")
            # Create a placeholder visualization with error message
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            fig.text(0.5, 0.5, "Failed to reduce dimensionality for visualization", 
                    fontsize=14, ha='center', weight='bold', color='#7f8c8d')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
    
    # Step 5: Create a DataFrame with the results
    embedding_df = pd.DataFrame({
        'X': X_reduced[:, 0],
        'Y': X_reduced[:, 1],
        'Repository': text_data['repository'].values,
        'PR': text_data['pr_number'].values,
        'Passed': text_data['passed_filter'].values,
        'Quality': text_data['quality_score'].values,
        'repo_key': text_data['repo_key'].values,
        'Title': text_data['title'].values
    })
    
    # Step 6: Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add subtle grid
    ax.grid(linestyle='--', alpha=0.3)
    
    # Plot points colored by repository
    for repo_key in text_data['repo_key'].unique():
        repo_points = embedding_df[embedding_df['repo_key'] == repo_key]
        
        ax.scatter(
            repo_points['X'],
            repo_points['Y'],
            s=repo_points['Quality'] * 80 + 40,  # Size based on quality
            color=generator.repository_colors.get(repo_key, 'gray'),
            alpha=0.7,
            edgecolor='white',
            linewidth=1,
            label=repo_key.replace('_', '/')
        )
    
    # Add repository legend with better positioning
    plt.legend(
        title="Repositories", 
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5), 
        frameon=True, 
        fontsize=9
    )
    
    # Enhance chart appearance
    ax.set_xlabel('Dimension 1', fontsize=12, weight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, weight='bold')
    
    if method == "t-SNE":
        title = 'PR Description Semantic Embedding (t-SNE)'
    else:
        title = 'PR Description Semantic Embedding (PCA)'
        
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    
    # Add simplified explanation
    explanation_text = "Using text embeddings to capture semantic relationships between PR descriptions. Point size indicates quality score."
    
    fig.text(0.5, 0.05, explanation_text, ha='center', fontsize=10, style='italic',
            color='#333333')
    
    plt.tight_layout(rect=[0, 0.08, 0.85, 0.95])  # Adjust for title and legend
    
    # Save to figures directory for reference
    plt.savefig(generator.figures_dir / "pr_text_embedding.png", dpi=300)
    
    # Add to PDF
    plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)