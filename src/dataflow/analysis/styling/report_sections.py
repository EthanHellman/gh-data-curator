#!/usr/bin/env python3
"""
Report Sections Module - Simplified version to fix imports
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import logging

from dataflow.analysis.functionality.visualization_utils import (
    add_gradient_background, 
    add_gradient_line, 
    get_quality_color,
    add_styled_box, 
    adjust_color_brightness
)

logger = logging.getLogger("enhanced_report_generator")

def add_cover_page(generator, pdf):
    """Add an attractive cover page to the report."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add title
    fig.text(0.5, 0.7, "Enhanced Data Curation Report", 
            fontsize=32, ha='center', va='center', weight='bold', color='#2c3e50')
    
    # Add timestamp
    fig.text(0.5, 0.3, f"Generated: {generator.timestamp}", 
            fontsize=12, ha='center', va='center', color='#7f8c8d')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_executive_summary(generator, pdf):
    """Add executive summary with key metrics and charts."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    fig.text(0.5, 0.95, "Executive Summary", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_cross_repo_comparison(generator, pdf):
    """Add cross-repository comparison section."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    fig.text(0.5, 0.95, "Cross-Repository Comparison", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_repo_section(generator, pdf, repo_key):
    """Add detailed analysis section for a single repository."""
    repo_name = repo_key.replace('_', '/')
    
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    fig.text(0.5, 0.95, f"Repository Analysis: {repo_name}", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_quality_profiles(generator, pdf):
    """Add exemplary PR profiles showing different quality characteristics."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    fig.text(0.5, 0.95, "Quality Profile Analysis", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_pr_scorecard(generator, pdf, pr):
    """Add an enhanced quality scorecard for a PR."""
    pr_number = pr.get("pr_number")
    repo_name = pr.get("repository")
    
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    fig.text(0.5, 0.95, f"PR #{pr_number} Quality Scorecard - {repo_name}", 
            fontsize=16, ha='center', weight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_methodology_section(generator, pdf):
    """Add an enhanced methodology section to the report."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    fig.text(0.5, 0.95, "Methodology", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)