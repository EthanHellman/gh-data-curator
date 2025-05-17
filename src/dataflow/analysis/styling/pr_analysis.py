
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
from sklearn.feature_extraction.text import TfidfVectorizer

from visualization_utils import add_gradient_background, add_gradient_line

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
    
    # Add introduction text
    intro_text = [
        "This section presents an analysis of PRs using clustering and dimensionality reduction techniques.",
        "The following visualizations show how PRs are distributed in feature space, revealing patterns",
        "and similarities across repositories. Code embedding and text embedding techniques have been",
        "used to analyze both the code changes and PR descriptions."
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
    """Add a visualization of code embeddings using PCA."""
    # Filter data to PRs that have been processed by at least the bot filter
    cluster_data = generator.pr_data[generator.pr_data['passed_bot_filter'] == True].copy()
    
    # Check if we have enough data
    if len(cluster_data) < 10:
        logger.warning("Not enough data for clustering visualization")
        return
    
    # Select code-specific features for embedding
    features = [
        'file_count', 'code_file_count', 'total_changes',
        'additions', 'deletions', 'size_score', 
        'complexity_score', 'relevance_score', 
        'problem_solving_score', 'code_quality_score'
    ]
    
    # Exclude features with missing values
    valid_features = []
    for feature in features:
        if not cluster_data[feature].isnull().any():
            valid_features.append(feature)
    
    # Prepare data for PCA
    X = cluster_data[valid_features]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply K-means clustering with silhouette analysis for optimal cluster number
    best_score = -1
    best_n_clusters = 3  # Default
    
    for n_clusters in range(2, min(6, len(X) // 5 + 1)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Simple evaluation based on inertia
        score = -kmeans.inertia_
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    # Apply best clustering
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create a DataFrame with PCA results and cluster labels
    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': cluster_labels,
        'Repository': cluster_data['repository'],
        'PR': cluster_data['pr_number'],
        'Passed': cluster_data['passed_filter'],
        'Quality': cluster_data['quality_score'],
        'repo_key': cluster_data['repo_key']
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for clusters
    cluster_colors = sns.color_palette("Set2", best_n_clusters)
    
    # Add subtle grid lines
    ax.grid(linestyle='--', alpha=0.3)
    
    # Create scatter plot with custom styling
    for i in range(best_n_clusters):
        cluster_points = pca_df[pca_df['Cluster'] == i]
        
        # Draw points with proper styling
        ax.scatter(
            cluster_points['PCA1'], 
            cluster_points['PCA2'],
            s=cluster_points['Quality'] * 100 + 30,  # Size based on quality score
            color=cluster_colors[i],
            alpha=0.7,
            edgecolor=[generator.repository_colors.get(repo, 'gray') for repo in cluster_points['repo_key']],
            linewidth=1.5,
            label=f'Cluster {i+1}'
        )
    
    # Add markers for passed PRs
    passed_prs = pca_df[pca_df['Passed'] == True]
    if not passed_prs.empty:
        ax.scatter(
            passed_prs['PCA1'],
            passed_prs['PCA2'],
            s=passed_prs['Quality'] * 100 + 30,
            facecolors='none',
            edgecolors='black',
            linewidth=1.5,
            alpha=0.7,
            label='Passed All Filters'
        )
    
    # Find top contributing features for each principal component
    feature_loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=valid_features
    )
    
    # Find top contributing features for each principal component
    pc1_features = feature_loadings.sort_values('PC1', key=abs, ascending=False)['PC1'].head(3)
    pc2_features = feature_loadings.sort_values('PC2', key=abs, ascending=False)['PC2'].head(3)
    
    # Format nice feature names
    feature_names = {
        'file_count': 'File Count',
        'code_file_count': 'Code Files',
        'total_changes': 'Total Changes',
        'additions': 'Additions',
        'deletions': 'Deletions',
        'size_score': 'Size Score',
        'complexity_score': 'Complexity Score',
        'relevance_score': 'Relevance Score',
        'problem_solving_score': 'Problem Solving',
        'code_quality_score': 'Code Quality'
    }
    
    # Create a legend for repositories
    repo_legend_elements = []
    for repo_key, color in generator.repository_colors.items():
        if repo_key in pca_df['repo_key'].values:
            repo_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor='gray',
                                           markeredgecolor=color,
                                           markersize=10, markeredgewidth=2,
                                           label=repo_key.replace('_', '/')))
    
    # Add the repository legend on the right side of the plot
    if repo_legend_elements:
        repo_legend = plt.legend(handles=repo_legend_elements, 
                               loc='center left', 
                               bbox_to_anchor=(1.02, 0.5),
                               title="Repositories",
                               frameon=True,
                               fontsize=9)
        plt.gca().add_artist(repo_legend)
    
    # Add cluster legend at the bottom
    cluster_legend_elements = []
    for i in range(best_n_clusters):
        cluster_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                               markerfacecolor=cluster_colors[i],
                                               markersize=10,
                                               label=f'Cluster {i+1}'))
    
    cluster_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor='none',
                                           markeredgecolor='black',
                                           markersize=10, markeredgewidth=1.5,
                                           label='Passed All Filters'))
    
    plt.legend(handles=cluster_legend_elements, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.12),
              title="Clusters",
              frameon=True,
              fontsize=9,
              ncol=len(cluster_legend_elements))
    
    # Enhance chart appearance
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                fontsize=12, weight='bold')
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                fontsize=12, weight='bold')
    ax.set_title('PR Code Embedding Visualization', fontsize=16, weight='bold', pad=20)
    
    # Add text explaining the principal components
    pc1_text = "PC1 reflects: " + ", ".join([f"{feature_names.get(feat, feat)} ({val:.2f})" 
                                         for feat, val in pc1_features.items()])
    pc2_text = "PC2 reflects: " + ", ".join([f"{feature_names.get(feat, feat)} ({val:.2f})" 
                                         for feat, val in pc2_features.items()])
    
    # Add box with component explanations
    explanation_text = (
        f"{pc1_text}\n"
        f"{pc2_text}\n\n"
        "This visualization shows PRs positioned according to their code metrics.\n"
        "Similar PRs are grouped into clusters, with point size indicating quality score.\n"
        "Edge color represents repository, while black outlines indicate PRs that passed all filters."
    )
    
    box = FancyBboxPatch((0.05, 0.01), 0.9, 0.12, fill=True, 
                      facecolor='#f8f9fa', alpha=0.9, transform=fig.transFigure, 
                      boxstyle="round,pad=0.02", zorder=1000)
    ax.add_artist(box)
    
    fig.text(0.5, 0.07, explanation_text, ha='center', fontsize=9, style='italic',
            color='#333333', zorder=1001)
    
    plt.tight_layout(rect=[0, 0.14, 0.85, 0.95])  # Adjust for title and legend
    
    # Save to figures directory for reference
    plt.savefig(generator.figures_dir / "pr_code_embedding.png", dpi=300)
    
    # Add to PDF
    plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_text_embedding_visualization(generator, pdf):
    """Add a visualization of PR description embeddings using TF-IDF and dimensionality reduction."""
    # Filter data to PRs with non-empty descriptions
    text_data = generator.pr_data[
        (generator.pr_data['passed_bot_filter'] == True) & 
        (generator.pr_data['body'].notna()) & 
        (generator.pr_data['body'] != '')
    ].copy()
    
    # Check if we have enough data
    if len(text_data) < 10:
        logger.warning("Not enough text data for embedding visualization")
        return
    
    # Prepare texts for embedding
    texts = text_data['body'].fillna('').tolist()
    
    # Create TF-IDF embeddings
    vectorizer = TfidfVectorizer(
        max_features=100,  # Limit to top features
        stop_words='english',
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.9  # Ignore terms that appear in more than 90% of documents
    )
    
    # Transform texts to TF-IDF features
    try:
        X_tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        logger.warning("Could not create text embeddings - not enough textual content")
        return
    
    # Apply dimensionality reduction - use t-SNE for better clustering in 2D
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, len(texts)-1), random_state=42)
        X_tsne = tsne.fit_transform(X_tfidf.toarray())
    except ValueError:
        # Fallback to PCA if t-SNE fails
        pca = PCA(n_components=2)
        X_tsne = pca.fit_transform(X_tfidf.toarray())
    
    # Create a DataFrame with the results
    embedding_df = pd.DataFrame({
        'X': X_tsne[:, 0],
        'Y': X_tsne[:, 1],
        'Repository': text_data['repository'].values,
        'PR': text_data['pr_number'].values,
        'Passed': text_data['passed_filter'].values,
        'Quality': text_data['quality_score'].values,
        'repo_key': text_data['repo_key'].values,
        'Title': text_data['title'].values
    })
    
    # Create the plot
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
    
    # Add markers for passed PRs
    passed_prs = embedding_df[embedding_df['Passed'] == True]
    if not passed_prs.empty:
        ax.scatter(
            passed_prs['X'],
            passed_prs['Y'],
            s=passed_prs['Quality'] * 80 + 40,
            facecolors='none',
            edgecolors='black',
            linewidth=1.5,
            alpha=0.7,
            label='Passed All Filters'
        )
    
    # Add repository legend
    plt.legend(title="Repositories", loc='center left', 
              bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
    
    # Enhance chart appearance
    ax.set_xlabel('Dimension 1', fontsize=12, weight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, weight='bold')
    ax.set_title('PR Description Text Embedding', fontsize=16, weight='bold', pad=20)
    
    # Add explanation
    # Extract most important terms
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top features from TF-IDF
    if hasattr(vectorizer, 'idf_'):
        top_indices = np.argsort(vectorizer.idf_)[:10]
        top_terms = [feature_names[i] for i in top_indices]
    else:
        top_terms = []
    
    explanation_text = (
        "This visualization shows PRs positioned according to their textual description content.\n"
        "PRs with similar descriptions are positioned closer together in the space.\n"
        "Point size indicates quality score, color indicates repository, and black outlines show PRs passing all filters."
    )
    
    if top_terms:
        explanation_text += f"\n\nCommon terms across PR descriptions: {', '.join(top_terms)}"
    
    box = FancyBboxPatch((0.05, 0.01), 0.9, 0.1, fill=True, 
                      facecolor='#f8f9fa', alpha=0.9, transform=fig.transFigure, 
                      boxstyle="round,pad=0.02", zorder=1000)
    ax.add_artist(box)
    
    fig.text(0.5, 0.06, explanation_text, ha='center', fontsize=9, style='italic',
            color='#333333', zorder=1001)
    
    plt.tight_layout(rect=[0, 0.12, 0.85, 0.95])  # Adjust for title and legend
    
    # Save to figures directory for reference
    plt.savefig(generator.figures_dir / "pr_text_embedding.png", dpi=300)
    
    # Add to PDF
    plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)