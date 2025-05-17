#!/usr/bin/env python3
"""
PR Analysis Module - Simplified version to fix imports
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
    """Add PR clustering analysis to the report."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    fig.text(0.5, 0.95, "PR Clustering Analysis", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
    intro_text = [
        "This section presents an analysis of PRs using clustering techniques.",
        "The visualizations show patterns across repositories."
    ]
    
    fig.text(0.1, 0.87, "\n".join(intro_text), 
            fontsize=12, va='top', ha='left', linespacing=1.5, 
            color='#333333')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_code_embedding_visualization(generator, pdf):
    """Add a visualization of code embeddings."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    fig.text(0.5, 0.5, "Code Embedding Visualization", 
            fontsize=18, ha='center', weight='bold', color='#2c3e50')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_text_embedding_visualization(generator, pdf):
    """Add a visualization of PR description embeddings."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    fig.text(0.5, 0.5, "Text Embedding Visualization", 
            fontsize=18, ha='center', weight='bold', color='#2c3e50')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)