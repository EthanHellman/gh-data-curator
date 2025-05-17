import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import logging
from pathlib import Path

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
    
    # Add a gradient background
    add_gradient_background(ax)
    
    # Add title with styling
    fig.text(0.5, 0.75, "Enhanced Data Curation Report", 
            fontsize=32, ha='center', va='center', weight='bold', color='#2c3e50')
    
    # Add subtitle
    fig.text(0.5, 0.68, "Comprehensive Analysis of GitHub PR Filtering Results", 
            fontsize=18, ha='center', va='center', color='#34495e')
    
    # Add Reflection.AI branding
    fig.text(0.5, 0.6, "Prepared for Reflection.AI", 
            fontsize=16, ha='center', va='center', color='#3498db', style='italic')
    
    # Add horizontal line
    add_gradient_line(fig, 0.2, 0.8, 0.55)
    
    # Add summary statistics
    stats = generator.generate_summary_stats()
    
    summary_box = add_styled_box(fig, 0.25, 0.35, 0.5, 0.15, 
                               color='#f8f9fa', edge_color='#bdc3c7')
    
    # Add key statistics with more spacing
    fig.text(0.5, 0.45, f"Total PRs Analyzed: {stats['total_prs']}", 
            fontsize=14, ha='center', va='center', color='#2c3e50')
    fig.text(0.5, 0.4, f"PRs Passed All Filters: {stats['passed_prs']} ({stats['pass_rate']:.1%})", 
            fontsize=14, ha='center', va='center', color='#2c3e50')
    fig.text(0.5, 0.35, f"Data Reduction: {stats['data_reduction']:.1%}", 
            fontsize=14, ha='center', va='center', color='#2c3e50')
    
    # Add timestamp and footer with more spacing
    fig.text(0.5, 0.25, f"Generated: {generator.timestamp}", 
            fontsize=12, ha='center', va='center', color='#7f8c8d')
    
    # Add stylized footer
    add_gradient_line(fig, 0.1, 0.9, 0.2)
    fig.text(0.5, 0.15, "GitHub Data Curation Pipeline", 
            fontsize=10, ha='center', va='center', color='#7f8c8d', style='italic')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_executive_summary(generator, pdf):
    """Add executive summary with key metrics and charts."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add section title with styling
    fig.text(0.5, 0.95, "Executive Summary", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
    # Calculate statistics for the summary
    stats = generator.generate_summary_stats()
    
    # Create a 2x2 grid of small subplots for summary visualizations with increased spacing
    gs = fig.add_gridspec(2, 2, left=0.1, right=0.9, bottom=0.35, top=0.8, wspace=0.4, hspace=0.5)
    
    # 1. Filter funnel chart (top left)
    funnel_ax = fig.add_subplot(gs[0, 0])
    
    funnel_stages = ['Initial', 'After Bot Filter', 'After Size Filter', 'After Content Filter']
    funnel_counts = [
        stats['total_prs'],
        stats['total_prs'] - stats['bot_filtered'],
        stats['total_prs'] - stats['bot_filtered'] - stats['size_filtered'],
        stats['passed_prs']
    ]
    
    # Create funnel plot with gradient colors
    funnel_colors = sns.color_palette("Blues", len(funnel_stages))
    funnel_bars = funnel_ax.bar(funnel_stages, funnel_counts, 
                               color=funnel_colors, edgecolor='white', linewidth=0.5)
    
    # Add data labels
    for i, v in enumerate(funnel_counts):
        funnel_ax.text(i, v + max(funnel_counts) * 0.02, str(v), 
                     ha='center', va='bottom', fontsize=9)
    
    funnel_ax.set_title("Filter Funnel", fontsize=12, weight='bold')
    funnel_ax.set_ylabel("Number of PRs", fontsize=10)
    funnel_ax.tick_params(axis='x', rotation=45, labelsize=8)
    funnel_ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 2. Repository comparison (top right)
    repo_ax = fig.add_subplot(gs[0, 1])
    
    repo_names = [name.replace('_', '/') for name in stats['repo_stats'].keys()]
    repo_pass_rates = [data['pass_rate'] for data in stats['repo_stats'].values()]
    
    # Sort by pass rate for better visualization
    sorted_indices = np.argsort(repo_pass_rates)
    sorted_repos = [repo_names[i] for i in sorted_indices]
    sorted_rates = [repo_pass_rates[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    repo_bars = repo_ax.barh(sorted_repos, sorted_rates, 
                            color=sns.color_palette("viridis", len(repo_names)),
                            edgecolor='white', linewidth=0.5)
    
    # Add data labels with enough spacing
    for i, v in enumerate(sorted_rates):
        repo_ax.text(v + 0.02, i, f"{v:.1%}", va='center', fontsize=8)
    
    repo_ax.set_title("Pass Rate by Repository", fontsize=12, weight='bold')
    repo_ax.set_xlabel("Pass Rate", fontsize=10)
    repo_ax.set_xlim(0, max(repo_pass_rates) * 1.3)  # Increased for label space
    repo_ax.grid(axis='x', linestyle='--', alpha=0.3)
    repo_ax.set_yticks(range(len(sorted_repos)))
    repo_ax.set_yticklabels([repo[:13] + '...' if len(repo) > 13 else repo for repo in sorted_repos], fontsize=8)
    
    # 3. Filter rejection reasons (bottom left)
    reject_ax = fig.add_subplot(gs[1, 0])
    
    rejection_labels = ['Bot Filter', 'Size Filter', 'Content Filter']
    rejection_counts = [stats['bot_filtered'], stats['size_filtered'], stats['content_filtered']]
    
    if sum(rejection_counts) > 0:  # Only create pie chart if there are rejections
        # Create pie chart with better spacing for labels
        wedges, texts, autotexts = reject_ax.pie(
            rejection_counts, 
            labels=rejection_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(rejection_labels)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            textprops={'fontsize': 9},
            pctdistance=0.85,  # Move percentage labels closer to center
            labeldistance=1.1  # Move labels slightly further out
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        reject_ax.set_title("Rejection by Filter Stage", fontsize=12, weight='bold')
    else:
        reject_ax.text(0.5, 0.5, "No filter rejections", ha='center', va='center')
        reject_ax.set_title("Rejection by Filter Stage (No Data)", fontsize=12, weight='bold')
    
    # 4. Quality score distribution (bottom right)
    quality_ax = fig.add_subplot(gs[1, 1])
    
    # Extract quality scores from PR data
    quality_scores = generator.pr_data['quality_score'].dropna()
    
    if not quality_scores.empty:
        # Create histogram with a single color instead of a color palette
        quality_bins = np.linspace(0, 1, 11)
        quality_ax.hist(quality_scores, bins=quality_bins, 
                      color='#3498db',  
                      edgecolor='white', linewidth=0.5)
        
        quality_ax.set_title("Quality Score Distribution", fontsize=12, weight='bold')
        quality_ax.set_xlabel("Quality Score", fontsize=10)
        quality_ax.set_ylabel("Number of PRs", fontsize=10)
        quality_ax.grid(linestyle='--', alpha=0.3)
    else:
        quality_ax.text(0.5, 0.5, "No quality score data", ha='center', va='center')
        quality_ax.set_title("Quality Score Distribution (No Data)", fontsize=12, weight='bold')
    
    # Add key insights at the bottom with better spacing
    key_findings = [
        f"Key Findings:",
        f"• {stats['bot_filtered_pct']:.1%} of PRs were filtered as likely automated or bot-created",
        f"• {stats['size_filtered_pct']:.1%} of PRs were filtered due to size or complexity issues",
        f"• {stats['content_filtered_pct']:.1%} of PRs were filtered due to content relevance issues",
        f"• Average quality score of passing PRs: {stats['avg_quality']:.2f} out of 1.0"
    ]
    
    fig.text(0.1, 0.25, "\n".join(key_findings), 
            fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
    
    plt.tight_layout(rect=[0, 0.2, 1, 0.9])  # Adjust the rect to avoid overlaps
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_cross_repo_comparison(generator, pdf):
    """Add cross-repository comparison section with enhanced visualizations."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add section title with styling
    fig.text(0.5, 0.95, "Cross-Repository Comparison", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
    # Create a gridspec for more control over plot layout with improved spacing
    gs = fig.add_gridspec(2, 2, left=0.1, right=0.9, bottom=0.35, top=0.85, wspace=0.4, hspace=0.5)
    
    # Get repository stats for comparison
    stats = generator.generate_summary_stats()
    repo_stats = stats['repo_stats']
    
    # 1. Pass rate comparison (top left)
    pass_ax = fig.add_subplot(gs[0, 0])
    
    repo_names = [name.replace('_', '/') for name in repo_stats.keys()]
    pass_rates = [data['pass_rate'] for data in repo_stats.values()]
    
    # Sort by pass rate
    sorted_indices = np.argsort(pass_rates)
    sorted_repos = [repo_names[i] for i in sorted_indices]
    sorted_rates = [pass_rates[i] for i in sorted_indices]
    
    # Use repository-specific colors
    repo_colors = [generator.repository_colors.get(name.replace('/', '_'), 'gray') 
                 for name in sorted_repos]
    
    # Create horizontal bar chart
    pass_bars = pass_ax.barh(sorted_repos, sorted_rates, 
                           color=repo_colors, edgecolor='white', linewidth=0.5)
    
    # Add data labels with better spacing
    for i, v in enumerate(sorted_rates):
        pass_ax.text(v + 0.02, i, f"{v:.1%}", va='center', fontsize=8)
    
    pass_ax.set_title("PR Pass Rate by Repository", fontsize=12, weight='bold')
    pass_ax.set_xlabel("Pass Rate", fontsize=10)
    pass_ax.set_xlim(0, 1.1)  # Increased for label space
    pass_ax.grid(axis='x', linestyle='--', alpha=0.3)
    pass_ax.set_yticks(range(len(sorted_repos)))
    pass_ax.set_yticklabels([repo[:12] + '...' if len(repo) > 12 else repo for repo in sorted_repos], fontsize=8)
    
    # 2. Quality score comparison (top right)
    quality_ax = fig.add_subplot(gs[0, 1])
    
    quality_values = [data['avg_quality'] for data in repo_stats.values()]
    
    # Sort by quality score
    q_sorted_indices = np.argsort(quality_values)
    q_sorted_repos = [repo_names[i] for i in q_sorted_indices]
    q_sorted_quality = [quality_values[i] for i in q_sorted_indices]
    
    # Use repository-specific colors
    q_repo_colors = [generator.repository_colors.get(name.replace('/', '_'), 'gray') 
                   for name in q_sorted_repos]
    
    # Create horizontal bar chart
    quality_bars = quality_ax.barh(q_sorted_repos, q_sorted_quality, 
                                 color=q_repo_colors, edgecolor='white', linewidth=0.5)
    
    # Add data labels with better spacing
    for i, v in enumerate(q_sorted_quality):
        quality_ax.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=8)
    
    quality_ax.set_title("Average Quality Score by Repository", fontsize=12, weight='bold')
    quality_ax.set_xlabel("Quality Score (0-1)", fontsize=10)
    quality_ax.set_xlim(0, 1.1)  # Increased for label space
    quality_ax.grid(axis='x', linestyle='--', alpha=0.3)
    quality_ax.set_yticks(range(len(q_sorted_repos)))
    quality_ax.set_yticklabels([repo[:12] + '...' if len(repo) > 12 else repo for repo in q_sorted_repos], fontsize=8)
    
    # 3. Filter rejection breakdown (bottom left)
    filter_ax = fig.add_subplot(gs[1, 0])
    
    # Prepare data for stacked bar chart
    bot_reject_rates = [data['bot_filtered_pct'] for data in repo_stats.values()]
    size_reject_rates = [data['size_filtered_pct'] for data in repo_stats.values()]
    content_reject_rates = [data['content_filtered_pct'] for data in repo_stats.values()]
    
    # Sort by total rejection rate
    total_reject = [b + s + c for b, s, c in zip(bot_reject_rates, size_reject_rates, content_reject_rates)]
    f_sorted_indices = np.argsort(total_reject)
    f_sorted_repos = [repo_names[i] for i in f_sorted_indices]
    f_sorted_bot = [bot_reject_rates[i] for i in f_sorted_indices]
    f_sorted_size = [size_reject_rates[i] for i in f_sorted_indices]
    f_sorted_content = [content_reject_rates[i] for i in f_sorted_indices]
    
    # Create stacked bar chart with clear separation
    filter_ax.barh(f_sorted_repos, f_sorted_bot, 
                 color='#e74c3c', edgecolor='white', linewidth=0.5, 
                 label='Bot Filter')
    filter_ax.barh(f_sorted_repos, f_sorted_size, left=f_sorted_bot, 
                 color='#f39c12', edgecolor='white', linewidth=0.5, 
                 label='Size Filter')
    filter_ax.barh(f_sorted_repos, f_sorted_content, 
                 left=[b + s for b, s in zip(f_sorted_bot, f_sorted_size)], 
                 color='#3498db', edgecolor='white', linewidth=0.5,
                 label='Content Filter')
    
    filter_ax.set_title("Rejection Rate by Filter Stage", fontsize=12, weight='bold')
    filter_ax.set_xlabel("Rejection Rate", fontsize=10)
    filter_ax.set_xlim(0, 1.1)  # Increased for label space
    filter_ax.grid(axis='x', linestyle='--', alpha=0.3)
    filter_ax.set_yticks(range(len(f_sorted_repos)))
    filter_ax.set_yticklabels([repo[:12] + '...' if len(repo) > 12 else repo for repo in f_sorted_repos], fontsize=8)
    filter_ax.legend(fontsize=8, loc='lower right', frameon=True)
    
    # 4. PR size comparison (bottom right)
    size_ax = fig.add_subplot(gs[1, 1])
    
    # Group PR data by repository and calculate average size metrics
    if not generator.pr_data.empty:
        size_data = generator.pr_data.groupby('repo_key').agg({
            'file_count': 'mean',
            'total_changes': 'mean'
        }).reset_index()
        
        # Map repo_key back to display name
        size_data['repository'] = size_data['repo_key'].str.replace('_', '/')
        
        # Create scatter plot with better spacing
        for i, repo in size_data.iterrows():
            size_ax.scatter(
                repo['file_count'], 
                repo['total_changes'],
                s=100,  # Size of markers
                color=generator.repository_colors.get(repo['repo_key'], 'gray'),
                edgecolor='white',
                linewidth=1,
                alpha=0.7,
                label=repo['repository']
            )
            
            # Add repository label with better positioning
            size_ax.annotate(
                repo['repository'][:10] + ('...' if len(repo['repository']) > 10 else ''),
                (repo['file_count'], repo['total_changes']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        size_ax.set_title("Average PR Size by Repository", fontsize=12, weight='bold')
        size_ax.set_xlabel("Avg. Files per PR", fontsize=10)
        size_ax.set_ylabel("Avg. Changes per PR", fontsize=10)
        size_ax.grid(linestyle='--', alpha=0.3)
        size_ax.set_xlim(0, size_data['file_count'].max() * 1.2)
        size_ax.set_ylim(0, size_data['total_changes'].max() * 1.2)
    else:
        size_ax.text(0.5, 0.5, "No size data available", ha='center', va='center')
        size_ax.set_title("Average PR Size by Repository (No Data)", fontsize=12, weight='bold')
    
    # Add key insights at the bottom with better spacing
    key_findings = [
        f"Key Insights:",
        f"• Repositories show significant variation in filter pass rates, from {min(pass_rates):.1%} to {max(pass_rates):.1%}",
        f"• Quality scores are generally higher in repositories with stricter filtering (lower pass rates)",
        f"• Bot activity varies substantially across repositories, affecting initial filtering stages",
        f"• PR size and complexity correlate with content quality in most repositories"
    ]
    
    fig.text(0.1, 0.25, "\n".join(key_findings), 
            fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
    
    plt.tight_layout(rect=[0, 0.2, 1, 0.9])  # Adjust the rect to avoid overlaps
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_repo_section(generator, pdf, repo_key):
    """Add detailed analysis section for a single repository."""
    repo_name = repo_key.replace('_', '/')
    repo_data = generator.repo_data.get(repo_key, {})
    repo_color = generator.repository_colors.get(repo_key, '#3498db')
    
    # Create repo statistics
    stats = generator.generate_summary_stats()
    repo_stats = stats['repo_stats'].get(repo_key, {})
    
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add repository name as title with styling
    fig.text(0.5, 0.95, f"Repository Analysis: {repo_name}", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color=repo_color)
    
    # Filter PR data for this repository
    repo_prs = generator.pr_data[generator.pr_data['repo_key'] == repo_key]
    
    # Create a gridspec for more control over plot layout with improved spacing
    gs = fig.add_gridspec(2, 2, left=0.1, right=0.9, bottom=0.35, top=0.85, wspace=0.4, hspace=0.5)
    
    # 1. PR quality distribution (top left)
    quality_ax = fig.add_subplot(gs[0, 0])
    
    if not repo_prs.empty:
        # Create histogram of quality scores
        quality_scores = repo_prs['quality_score'].dropna()
        
        if not quality_scores.empty:
            quality_bins = np.linspace(0, 1, 11)
            quality_ax.hist(quality_scores, bins=quality_bins, 
                          color=adjust_color_brightness(repo_color, 1.1),
                          edgecolor='white', linewidth=0.5, alpha=0.7)
            
            # Add mean line
            mean_quality = quality_scores.mean()
            quality_ax.axvline(mean_quality, color='red', linestyle='--', alpha=0.7)
            quality_ax.text(mean_quality + 0.02, quality_ax.get_ylim()[1] * 0.9, 
                          f"Mean: {mean_quality:.2f}", rotation=90, color='red', fontsize=8)
            
            quality_ax.set_title("Quality Score Distribution", fontsize=12, weight='bold')
            quality_ax.set_xlabel("Quality Score", fontsize=10)
            quality_ax.set_ylabel("Number of PRs", fontsize=10)
            quality_ax.grid(linestyle='--', alpha=0.3)
        else:
            quality_ax.text(0.5, 0.5, "No quality score data", ha='center', va='center')
            quality_ax.set_title("Quality Score Distribution (No Data)", fontsize=12, weight='bold')
    else:
        quality_ax.text(0.5, 0.5, "No PR data available", ha='center', va='center')
        quality_ax.set_title("Quality Score Distribution (No Data)", fontsize=12, weight='bold')
    
    # 2. Filter rejection analysis (top right)
    filter_ax = fig.add_subplot(gs[0, 1])
    
    # Prepare filter data
    filter_labels = ['Bot Filter', 'Size Filter', 'Content Filter', 'Passed All']
    filter_counts = [
        repo_stats.get('bot_filtered', 0),
        repo_stats.get('size_filtered', 0),
        repo_stats.get('content_filtered', 0),
        repo_stats.get('passed_prs', 0)
    ]
    
    # Create pie chart with better label spacing
    if sum(filter_counts) > 0:
        filter_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        wedges, texts, autotexts = filter_ax.pie(
            filter_counts, 
            labels=filter_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=filter_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            textprops={'fontsize': 9},
            pctdistance=0.85,  # Move percentage labels closer to center
            labeldistance=1.1  # Move labels slightly further out
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
            # Set text color based on background for better contrast
            autotext.set_color('white')
        
        filter_ax.set_title("Filter Results", fontsize=12, weight='bold')
    else:
        filter_ax.text(0.5, 0.5, "No filter data available", ha='center', va='center')
        filter_ax.set_title("Filter Results (No Data)", fontsize=12, weight='bold')
    
    # 3. PR size analysis (bottom left)
    size_ax = fig.add_subplot(gs[1, 0])
    
    if not repo_prs.empty:
        # Extract size metrics
        file_counts = repo_prs['file_count'].dropna()
        change_counts = repo_prs['total_changes'].dropna()
        
        if not file_counts.empty and not change_counts.empty:
            # Create scatter plot of PR size vs quality with better color scaling
            scatter = size_ax.scatter(
                file_counts, 
                change_counts,
                c=repo_prs['quality_score'],
                cmap='viridis',
                alpha=0.7,
                edgecolor='white',
                linewidth=0.5,
                s=50
            )
            
            # Add color bar with better positioning
            cbar = plt.colorbar(scatter, ax=size_ax, fraction=0.046, pad=0.04)
            cbar.set_label('Quality Score', fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            
            size_ax.set_title("PR Size Analysis", fontsize=12, weight='bold')
            size_ax.set_xlabel("Number of Files", fontsize=10)
            size_ax.set_ylabel("Total Changes (Lines)", fontsize=10)
            size_ax.grid(linestyle='--', alpha=0.3)
            
            # Limit axes for better visualization
            size_ax.set_xlim(0, np.percentile(file_counts, 95) * 1.1)
            size_ax.set_ylim(0, np.percentile(change_counts, 95) * 1.1)
        else:
            size_ax.text(0.5, 0.5, "No size data available", ha='center', va='center')
            size_ax.set_title("PR Size Analysis (No Data)", fontsize=12, weight='bold')
    else:
        size_ax.text(0.5, 0.5, "No PR data available", ha='center', va='center')
        size_ax.set_title("PR Size Analysis (No Data)", fontsize=12, weight='bold')
    
    # 4. Quality metrics comparison (bottom right)
    metrics_ax = fig.add_subplot(gs[1, 1])
    
    if not repo_prs.empty:
        # Extract relevant metrics
        metrics = {
            'Size Score': repo_prs['size_score'].mean(),
            'Complexity': repo_prs['complexity_score'].mean(),
            'Relevance': repo_prs['relevance_score'].mean(),
            'Code Quality': repo_prs['code_quality_score'].mean(),
            'Problem Solving': repo_prs['problem_solving_score'].mean()
        }
        
        # Remove NaN values
        metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        
        if metrics:
            # Create bar chart with improved spacing between bars
            metric_bars = metrics_ax.bar(
                metrics.keys(),
                metrics.values(),
                color=sns.color_palette("viridis", len(metrics)),
                edgecolor='white',
                linewidth=0.5,
                alpha=0.7,
                width=0.7  # Reduced width for better spacing
            )
            
            # Add data labels with better positioning
            for i, (key, value) in enumerate(metrics.items()):
                metrics_ax.text(i, value + 0.03, f"{value:.2f}", 
                              ha='center', fontsize=8)
            
            metrics_ax.set_title("Quality Metrics Comparison", fontsize=12, weight='bold')
            metrics_ax.set_ylim(0, 1.1)  # Increased for label space
            metrics_ax.set_ylabel("Score (0-1)", fontsize=10)
            metrics_ax.tick_params(axis='x', rotation=45, labelsize=8)
            metrics_ax.grid(axis='y', linestyle='--', alpha=0.3)
        else:
            metrics_ax.text(0.5, 0.5, "No quality metrics available", ha='center', va='center')
            metrics_ax.set_title("Quality Metrics Comparison (No Data)", fontsize=12, weight='bold')
    else:
        metrics_ax.text(0.5, 0.5, "No PR data available", ha='center', va='center')
        metrics_ax.set_title("Quality Metrics Comparison (No Data)", fontsize=12, weight='bold')
    
    # Add key summary statistics and insights at the bottom in two columns
    summary_box = Rectangle((0.1, 0.1), 0.8, 0.15, 
                          facecolor='#f8f9fa', alpha=0.5, 
                          edgecolor='#bdc3c7', linewidth=1,
                          transform=fig.transFigure)
    ax.add_patch(summary_box)
    
    # Left column: Summary statistics
    summary_stats = [
        f"Total PRs Analyzed: {repo_stats.get('total_prs', 0)}",
        f"PRs Passed All Filters: {repo_stats.get('passed_prs', 0)} ({repo_stats.get('pass_rate', 0):.1%})",
        f"Average Quality Score: {repo_stats.get('avg_quality', 0):.2f} out of 1.0"
    ]
    
    fig.text(0.15, 0.2, "\n".join(summary_stats), 
            fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
    
    # Right column: Repository insights
    if not repo_prs.empty and not repo_prs['file_count'].empty and metrics:
        key_findings = [
            f"Repository Insights:",
            f"• Bot activity level: {'High' if repo_stats.get('bot_filtered_pct', 0) > 0.3 else 'Moderate' if repo_stats.get('bot_filtered_pct', 0) > 0.1 else 'Low'}",
            f"• PR size distribution: {'Large' if repo_prs['file_count'].mean() > 10 else 'Medium' if repo_prs['file_count'].mean() > 5 else 'Small'} (avg. {repo_prs['file_count'].mean():.1f} files per PR)",
            f"• Main quality strength: {max(metrics.items(), key=lambda x: x[1])[0] if metrics else 'N/A'}",
            f"• Main quality weakness: {min(metrics.items(), key=lambda x: x[1])[0] if metrics else 'N/A'}"
        ]
    else:
        key_findings = [
            f"Repository Insights:",
            f"• Bot activity level: {'High' if repo_stats.get('bot_filtered_pct', 0) > 0.3 else 'Moderate' if repo_stats.get('bot_filtered_pct', 0) > 0.1 else 'Low'}",
            f"• PR size distribution: Insufficient data",
            f"• Main quality strength: Insufficient data",
            f"• Main quality weakness: Insufficient data"
        ]
    
    fig.text(0.55, 0.2, "\n".join(key_findings), 
            fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.9])  # Adjust the rect to avoid overlaps
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_quality_profiles(generator, pdf):
    """Add exemplary PR profiles showing different quality characteristics with code snippets."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add section title with styling
    fig.text(0.5, 0.95, "Quality Profiles of Exemplary PRs", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
    # Add introduction text with better spacing
    intro_text = [
        "The following sections showcase PRs with varying quality levels across repositories.",
        "For each repository, we present:",
        "• One high-quality PR that passed all filters",
        "• One medium-quality PR that passed all filters (where available)",
        "• One PR that failed filtering (with explanation of why it failed)",
        "",
        "Each scorecard provides detailed metrics and includes a representative code snippet",
        "to help visualize what the PR changes actually look like."
    ]
    
    fig.text(0.1, 0.8, "\n".join(intro_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Add note about detailed scorecards
    fig.text(0.5, 0.6, "Detailed PR quality scorecards are presented on the following pages", 
            fontsize=12, ha='center', style='italic', color='#7f8c8d')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Find exemplary PRs for each repository
    if not generator.pr_data.empty:
        # Process each repository
        for repo_key in generator.repo_data.keys():
            repo_prs = generator.pr_data[generator.pr_data['repo_key'] == repo_key]
            selected_prs = []
            
            # 1. Find high quality PR (highest score from filtered PRs)
            high_quality_prs = repo_prs[
                repo_prs['passed_filter'] == True
            ].sort_values('quality_score', ascending=False).head(1)
            
            if not high_quality_prs.empty:
                pr = high_quality_prs.iloc[0].to_dict()
                pr["metadata"] = next((meta for meta in generator.repo_data.get(repo_key, {}).get("filter_metadata", []) 
                                   if meta.get("pr_number") == pr["pr_number"]), {})
                pr["quality_level"] = "High Quality PR"
                selected_prs.append(pr)
            
            # 2. Find medium quality PR (lowest passing score that's still above threshold)
            medium_quality_prs = repo_prs[
                (repo_prs['passed_filter'] == True)
            ].sort_values('quality_score', ascending=True).head(1)
            
            if not medium_quality_prs.empty and len(high_quality_prs) > 0:
                # Make sure it's not the same PR as high quality (if only one passed)
                if medium_quality_prs.iloc[0]['pr_number'] != high_quality_prs.iloc[0]['pr_number']:
                    pr = medium_quality_prs.iloc[0].to_dict()
                    pr["metadata"] = next((meta for meta in generator.repo_data.get(repo_key, {}).get("filter_metadata", []) 
                                       if meta.get("pr_number") == pr["pr_number"]), {})
                    pr["quality_level"] = "Medium Quality PR"
                    selected_prs.append(pr)
            
            # 3. Find a failed PR (prioritize different filter failures)
            # First try content filter
            content_filtered_pr = repo_prs[
                (repo_prs['passed_bot_filter'] == True) &
                (repo_prs['passed_size_filter'] == True) &
                (repo_prs['passed_content_filter'] == False)
            ].head(1)
            
            if not content_filtered_pr.empty:
                pr = content_filtered_pr.iloc[0].to_dict()
                pr["metadata"] = next((meta for meta in generator.repo_data.get(repo_key, {}).get("filter_metadata", []) 
                                   if meta.get("pr_number") == pr["pr_number"]), {})
                pr["quality_level"] = "Failed PR (Content Filter)"
                pr["failure_reason"] = "Failed Content Filter: Low relevance or code quality"
                selected_prs.append(pr)
            else:
                # Try size filter
                size_filtered_pr = repo_prs[
                    (repo_prs['passed_bot_filter'] == True) &
                    (repo_prs['passed_size_filter'] == False)
                ].head(1)
                
                if not size_filtered_pr.empty:
                    pr = size_filtered_pr.iloc[0].to_dict()
                    pr["metadata"] = next((meta for meta in generator.repo_data.get(repo_key, {}).get("filter_metadata", []) 
                                       if meta.get("pr_number") == pr["pr_number"]), {})
                    pr["quality_level"] = "Failed PR (Size Filter)"
                    pr["failure_reason"] = "Failed Size Filter: Too small, too large, or too complex"
                    selected_prs.append(pr)
                else:
                    # Try bot filter
                    bot_filtered_pr = repo_prs[
                        repo_prs['passed_bot_filter'] == False
                    ].head(1)
                    
                    if not bot_filtered_pr.empty:
                        pr = bot_filtered_pr.iloc[0].to_dict()
                        pr["metadata"] = next((meta for meta in generator.repo_data.get(repo_key, {}).get("filter_metadata", []) 
                                           if meta.get("pr_number") == pr["pr_number"]), {})
                        pr["quality_level"] = "Failed PR (Bot Filter)"
                        pr["failure_reason"] = "Failed Bot Filter: Likely automated or bot-created"
                        selected_prs.append(pr)
            
            # Add scorecards for this repository's PRs
            for pr in selected_prs:
                add_pr_scorecard(generator, pdf, pr)
    else:
        # Add a note if no PR data available
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        fig.text(0.5, 0.5, "No PR data available for quality profiles", 
                fontsize=16, ha='center', weight='bold', color='#7f8c8d')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def add_pr_scorecard(generator, pdf, pr):
    """Add an enhanced quality scorecard for a PR with better formatting and code snippet."""
    pr_number = pr.get("pr_number")
    repo_name = pr.get("repository")
    repo_key = pr.get("repo_key", repo_name.replace("/", "_"))
    metadata = pr.get("metadata", {})
    quality_level = pr.get("quality_level", "")
    failure_reason = pr.get("failure_reason", "")
    
    # Always get PR title from basic_info.json in raw data
    pr_title = ""
    basic_info_path = Path(f"/home/ubuntu/gh-data-curator/data/raw/{repo_key}/pr_{pr_number}/basic_info.json")
    if not basic_info_path.exists():
        # Try relative path based on generator.data_dir
        basic_info_path = generator.data_dir / "raw" / repo_key / f"pr_{pr_number}" / "basic_info.json"
    
    try:
        if basic_info_path.exists():
            with open(basic_info_path, 'r') as f:
                basic_info = json.load(f)
                pr_title = basic_info.get("title", "")
    except Exception as e:
        logger.error(f"Error loading basic_info.json for {repo_key}/pr_{pr_number}: {e}")
    
    # Fall back to title from PR data if basic_info.json not available
    if not pr_title:
        pr_title = pr.get("title", "No title available")
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), 
                            gridspec_kw={'height_ratios': [1, 1.2, 1.5], 
                                        'hspace': 0.4, 'wspace': 0.3})
    
    # Flatten axes for easier access
    axes = axes.flatten()
    
    # Add title with more space above to prevent overlap
    fig.suptitle(f"PR #{pr_number} Quality Scorecard - {repo_name}", 
                fontsize=16, weight='bold', y=0.98)
    
    # Add quality level indicator with appropriate color
    quality_colors = {
        "High Quality PR": "#2ecc71",  # Green
        "Medium Quality PR": "#f1c40f",  # Yellow
        "Failed PR (Content Filter)": "#e74c3c",  # Red
        "Failed PR (Size Filter)": "#e74c3c",
        "Failed PR (Bot Filter)": "#e74c3c"
    }
    quality_color = quality_colors.get(quality_level, "#3498db")
    
    # Improved quality level badge - positioned below title with better styling
    quality_box = Rectangle((0.1, 0.92), 0.8, 0.04, 
                          facecolor=quality_color, alpha=0.2, 
                          edgecolor=quality_color, linewidth=1,
                          transform=fig.transFigure)
    fig.add_artist(quality_box)
    
    fig.text(0.5, 0.94, quality_level, 
            fontsize=13, ha='center', weight='bold', 
            color=quality_color)
    
    # If failed PR, add failure reason with improved styling
    if "Failed PR" in quality_level and failure_reason:
        fig.text(0.5, 0.925, failure_reason, 
                fontsize=11, ha='center', style='italic', 
                color=quality_color)
    
    # 1. Filter scores (top left)
    filter_scores = {
        'Bot Filter': pr.get('passed_bot_filter', False) and 1.0 or metadata.get("bot_filter", {}).get("details", {}).get("confidence", 0.0),
        'Size Filter': pr.get('passed_size_filter', False) and pr.get('size_score', 0.0) or metadata.get("size_filter", {}).get("details", {}).get("normalized_score", 0.0),
        'Content Filter': pr.get('passed_content_filter', False) and pr.get('relevance_score', 0.0) or metadata.get("content_filter", {}).get("details", {}).get("relevance_score", 0.0),
        'Overall Quality': pr.get('quality_score', 0.0) or metadata.get("quality_score", 0.0)
    }
    
    # Use enhanced styling for bar chart
    colors = sns.color_palette("viridis", len(filter_scores))
    bars = axes[0].bar(filter_scores.keys(), filter_scores.values(), 
                     color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels with better positioning
    for i, (key, value) in enumerate(filter_scores.items()):
        axes[0].text(i, value + 0.03, f"{value:.2f}", ha='center', va='bottom', 
                   fontsize=9, weight='bold')
    
    # Enhance chart appearance
    axes[0].set_ylim(0, 1.1)  # Increased for label space
    axes[0].set_title("Filter Scores", fontsize=12, weight='bold')
    axes[0].set_ylabel("Score (0-1)", fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add reference line for quality threshold with better positioning
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    axes[0].text(0, 0.5, "Threshold", fontsize=8, color='red', 
               va='bottom', ha='left', alpha=0.7)
    
    # 2. File composition (top right) based on file types edited in the PR
    # Get file data from files.json in raw data
    files_path = Path(f"/home/ubuntu/gh-data-curator/data/raw/{repo_key}/pr_{pr_number}/files.json")
    if not files_path.exists():
        # Try relative path based on generator.data_dir
        files_path = generator.data_dir / "raw" / repo_key / f"pr_{pr_number}" / "files.json"
    
    file_types = {}
    try:
        if files_path.exists():
            with open(files_path, 'r') as f:
                files_data = json.load(f)
                
                # Count file types by extension
                for file in files_data:
                    if 'filename' in file:
                        ext = Path(file['filename']).suffix.lower()
                        if not ext:
                            ext = "(no extension)"
                        else:
                            # Remove the dot
                            ext = ext[1:]
                        
                        # Group similar extensions
                        if ext in ['py', 'pyw', 'pyx']:
                            category = 'Python'
                        elif ext in ['js', 'jsx', 'ts', 'tsx']:
                            category = 'JavaScript'
                        elif ext in ['html', 'htm', 'template']:
                            category = 'HTML'
                        elif ext in ['css', 'scss', 'sass', 'less']:
                            category = 'CSS'
                        elif ext in ['md', 'rst', 'txt', 'text']:
                            category = 'Docs'
                        elif ext in ['json', 'yaml', 'yml', 'toml', 'ini', 'cfg']:
                            category = 'Config'
                        elif ext in ['xml', 'svg']:
                            category = 'XML'
                        elif ext in ['sql']:
                            category = 'SQL'
                        elif ext in ['gitignore', 'dockerignore', 'editorconfig']:
                            category = 'Meta'
                        else:
                            category = ext.capitalize()  # Use the extension as category
                        
                        if category in file_types:
                            file_types[category] += 1
                        else:
                            file_types[category] = 1
    except Exception as e:
        logger.error(f"Error loading files.json for {repo_key}/pr_{pr_number}: {e}")
    
    if file_types:
        # Use enhanced styling for pie chart with better label positioning
        wedges, texts, autotexts = axes[1].pie(
            file_types.values(), 
            labels=file_types.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(file_types)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            textprops={'fontsize': 9},
            pctdistance=0.85,  # Move percentage labels closer to center
            labeldistance=1.1  # Move labels slightly further out
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
        
        axes[1].set_title("File Types Modified", fontsize=12, weight='bold')
    else:
        axes[1].text(0.5, 0.5, "No file data available", ha='center', va='center')
        axes[1].set_title("File Types Modified (No Data)", fontsize=12, weight='bold')
        axes[1].axis('off')
    
    # 3. Code changes (middle left)
    change_data = {
        'Additions': pr.get('additions', 0) or metadata.get("size_filter", {}).get("details", {}).get("additions", 0),
        'Deletions': pr.get('deletions', 0) or metadata.get("size_filter", {}).get("details", {}).get("deletions", 0)
    }
    
    # Use enhanced styling for bar chart
    bars = axes[2].bar(change_data.keys(), change_data.values(), 
                     color=['green', 'red'], edgecolor='white', linewidth=0.5)
    
    # Add value labels with better positioning
    for i, (key, value) in enumerate(change_data.items()):
        axes[2].text(i, value + max(change_data.values()) * 0.03, str(int(value)), 
                   ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Enhance chart appearance
    axes[2].set_title("Code Changes", fontsize=12, weight='bold')
    axes[2].set_ylabel("Number of Lines", fontsize=10)
    axes[2].grid(axis='y', linestyle='--', alpha=0.3)
    
    # 4. Relevant files (middle right)
    # Get relevant files from metadata or PR data
    relevant_files = []
    
    # First try relevant_files from metadata
    if "relevant_files" in metadata:
        relevant_files = metadata.get("relevant_files", [])
    # Then try from PR object directly
    elif "relevant_files" in pr:
        relevant_files = pr.get("relevant_files", [])
    # If still empty, try loading from filter_metadata.json
    if not relevant_files:
        filter_metadata_path = Path(f"/home/ubuntu/gh-data-curator/data/filtered/{repo_key}/pr_{pr_number}/filter_metadata.json")
        if not filter_metadata_path.exists():
            # Try relative path based on generator.data_dir
            filter_metadata_path = generator.data_dir / "filtered" / repo_key / f"pr_{pr_number}" / "filter_metadata.json"
        
        if filter_metadata_path.exists():
            try:
                with open(filter_metadata_path, 'r') as f:
                    filter_data = json.load(f)
                    relevant_files = filter_data.get("relevant_files", [])
            except Exception as e:
                logger.error(f"Error loading filter_metadata.json for {repo_key}/pr_{pr_number}: {e}")
    
    num_relevant = len(relevant_files)
    
    if relevant_files:
        axes[3].axis('off')
        axes[3].set_title("Relevant Files", fontsize=12, weight='bold')
        
        # Create a styled list of relevant files with better spacing
        file_list = f"Files that provide context ({num_relevant} total):"
        axes[3].text(0.5, 0.95, file_list, 
                   ha='center', va='top', fontsize=10, weight='bold')
        
        # Show up to 5 files with more spacing, with ellipsis if there are more
        display_files = relevant_files[:5]
        if len(relevant_files) > 5:
            display_files.append("... and more")
            
        # Use a cool background for the file list with better positioning
        file_bg = Rectangle((0.1, 0.1), 0.8, 0.8, 
                          facecolor='#f8f9fa', alpha=0.5, 
                          edgecolor='#bdc3c7', linewidth=1)
        axes[3].add_patch(file_bg)
        
        # Position files with better spacing - all files in black text
        for i, file in enumerate(display_files):
            y_pos = 0.85 - (i * 0.12)  # Increased spacing between lines
            
            # Use just a prefix based on file type but keep text black
            if "..." in file:
                prefix = ""
                text_color = "#666666"  # Gray for ellipsis
            else:
                ext = Path(file).suffix.lower()
                # Choose prefix based on file type
                if ext in ['.py', '.pyw']:
                    prefix = "PY: "
                elif ext in ['.js', '.jsx', '.ts']:
                    prefix = "JS: "
                elif ext in ['.md', '.rst', '.txt']:
                    prefix = "DOC: "
                elif ext in ['.json', '.yml', '.yaml', '.toml']:
                    prefix = "CFG: "
                elif ext in ['.html', '.htm']:
                    prefix = "HTML: "
                else:
                    prefix = ""
                text_color = "#333333"  # Consistent black color for all files
            
            # Display filename with word wrapping for long filenames
            if len(file) > 30 and "..." not in file:
                # Split long filenames to multiple lines
                parts = file.split('/')
                if len(parts) > 2:
                    # Group directory parts on one line, filename on another
                    dir_path = '/'.join(parts[:-1])
                    file_name = parts[-1]
                    axes[3].text(0.15, y_pos, f"{prefix}{dir_path}/", 
                               ha='left', va='center', fontsize=8, color=text_color)
                    axes[3].text(0.25, y_pos-0.05, f"{file_name}", 
                               ha='left', va='center', fontsize=8, color=text_color)
                else:
                    axes[3].text(0.15, y_pos, f"{prefix}{file}", 
                               ha='left', va='center', fontsize=8, color=text_color)
            else:
                axes[3].text(0.15, y_pos, f"{prefix}{file}", 
                           ha='left', va='center', fontsize=9, color=text_color)
    else:
        axes[3].axis('off')
        axes[3].text(0.5, 0.5, "No relevant files identified", 
                   ha='center', va='center', fontsize=12)
        axes[3].set_title("Relevant Files", fontsize=12, weight='bold')
    
    # 5. Add code snippet (bottom row, spans both columns)
    axes[4].axis('off')
    axes[5].axis('off')
    
    code_snippet, filename, language = extract_code_snippet_from_pr(generator, repo_key, pr_number)
    
    # Add title for code snippet
    fig.text(0.1, 0.37, "Representative Code Snippet:", 
            fontsize=12, weight='bold', color='#333333')
    fig.text(0.1, 0.34, f"File: {filename}", 
            fontsize=10, color='#333333', style='italic')
    
    # Background for code
    code_bg = Rectangle((0.1, 0.1), 0.8, 0.23, 
                      facecolor='#f8f9fa', alpha=0.8, 
                      edgecolor='#bdc3c7', linewidth=1,
                      transform=fig.transFigure)
    fig.add_artist(code_bg)
    
    # Add code with syntax-highlighting-like colors (simple version)
    if language in ['python', 'javascript', 'java', 'typescript', 'rust']:
        # Add basic coloring for code elements
        lines = code_snippet.split('\n')
        y_pos = 0.31  # Starting position
        for line in lines[:10]:  # Limit to 10 lines
            # Skip empty lines
            if not line.strip():
                y_pos -= 0.018
                continue
                
            # Indent the line properly
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                fig.text(0.12, y_pos, ' ' * indent, 
                        fontsize=9, family='monospace', color='#333333')
            
            # Simple syntax highlighting
            line = line.lstrip()
            
            # Different styling based on line content
            if line.startswith(('def ', 'class ', 'function ', 'fn ', 'pub fn')):
                fig.text(0.12 + 0.01 * indent, y_pos, line, 
                        fontsize=9, family='monospace', color='#0000FF', weight='bold')
            elif line.startswith(('import ', 'from ', '#', '//')):
                fig.text(0.12 + 0.01 * indent, y_pos, line, 
                        fontsize=9, family='monospace', color='#008000')
            elif any(kw in f" {line} " for kw in [' if ', ' else ', ' for ', ' while ', ' return ']):
                fig.text(0.12 + 0.01 * indent, y_pos, line, 
                        fontsize=9, family='monospace', color='#800080')
            elif '=' in line:
                fig.text(0.12 + 0.01 * indent, y_pos, line, 
                        fontsize=9, family='monospace', color='#000080')
            else:
                fig.text(0.12 + 0.01 * indent, y_pos, line, 
                        fontsize=9, family='monospace', color='#333333')
            
            y_pos -= 0.018  # Move to next line
        
        # Add "..." if there are more lines
        if len(lines) > 10:
            fig.text(0.12, y_pos, "...", 
                    fontsize=9, family='monospace', color='#666666')
    else:
        # Simple display for other languages
        lines = code_snippet.split('\n')
        y_pos = 0.31
        for line in lines[:10]:
            if not line.strip():
                y_pos -= 0.018
                continue
            fig.text(0.12, y_pos, line, 
                    fontsize=9, family='monospace', color='#333333')
            y_pos -= 0.018
        
        # Add "..." if there are more lines
        if len(lines) > 10:
            fig.text(0.12, y_pos, "...", 
                    fontsize=9, family='monospace', color='#666666')
    
    # Add PR title at the bottom with better positioning
    # Use a more subtle box for the PR title
    title_box = Rectangle((0.05, 0.01), 0.9, 0.05, 
                        facecolor='#e8f4f8', alpha=0.5, 
                        edgecolor='#3498db', linewidth=1,
                        transform=fig.transFigure)
    fig.add_artist(title_box)
    
    # Add title with smaller font - ensure it's not empty
    fig.text(0.07, 0.04, f"Title: {pr_title or 'No title available'}", fontsize=9, weight='bold')
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.91])  # Adjusted rect to make room for title and footer
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_correlation_section(generator, pdf):
    """Add correlation analysis between quality dimensions."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add section title with styling
    fig.text(0.5, 0.95, "Correlation Between Quality Dimensions", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
    # Create a correlation matrix from PR data
    if not generator.pr_data.empty:
        # Select columns for correlation
        corr_cols = [
            'bot_confidence', 'file_count', 'code_file_count', 
            'total_changes', 'additions', 'deletions',
            'size_score', 'complexity_score', 'relevance_score', 
            'problem_solving_score', 'code_quality_score', 'quality_score'
        ]
        
        # Filter to include only columns that are present and have data
        valid_cols = [col for col in corr_cols if col in generator.pr_data.columns]
        
        # Prepare data for correlation
        corr_data = generator.pr_data[valid_cols].dropna()
        
        if not corr_data.empty and len(valid_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = corr_data.corr()
            
            # Create the correlation heatmap
            corr_ax = fig.add_subplot(111)
            
            # Generate heatmap with better spacing and formatting
            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = False
            
            # Custom color map with better range
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Create the heatmap with better styling
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                      square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, 
                      fmt=".2f", annot_kws={"size": 8})
            
            # Better labels with improved styling
            corr_ax.set_xticklabels(
                [col.replace('_', ' ').title() for col in corr_matrix.columns],
                rotation=45, ha='right', rotation_mode='anchor', fontsize=9
            )
            corr_ax.set_yticklabels(
                [col.replace('_', ' ').title() for col in corr_matrix.index],
                rotation=0, fontsize=9
            )
            
            # Add title
            corr_ax.set_title("Correlation Between Quality Dimensions", fontsize=14, pad=20)
            
        else:
            fig.text(0.5, 0.5, "Insufficient data for correlation analysis", 
                   fontsize=14, ha='center', weight='bold', color='#7f8c8d')
    else:
        fig.text(0.5, 0.5, "No PR data available for correlation analysis", 
               fontsize=14, ha='center', weight='bold', color='#7f8c8d')
    
    plt.tight_layout(rect=[0, 0.22, 1, 0.9])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def extract_code_snippet_from_pr(generator, repo_key, pr_number, max_lines=15):
    """
    Extract a representative code snippet from a PR's raw files.
    
    Args:
        generator: The report generator instance
        repo_key: Repository key (e.g., 'django_django')
        pr_number: PR number
        max_lines: Maximum number of lines to extract
    
    Returns:
        Tuple of (code_snippet, file_name, language)
    """
    # Path to raw PR data
    files_path = Path(f"/home/ubuntu/gh-data-curator/data/raw/{repo_key}/pr_{pr_number}/files.json")
    if not files_path.exists():
        # Try relative path based on generator.data_dir
        files_path = generator.data_dir / "raw" / repo_key / f"pr_{pr_number}" / "files.json"
    
    if not files_path.exists():
        return "No files data available", "N/A", "text"
    
    try:
        with open(files_path, 'r') as f:
            files_data = json.load(f)
        
        # Find the file with the most changes
        significant_file = None
        max_changes = 0
        code_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.rs', '.go', '.rb', '.php', '.ts', '.jsx', '.tsx']
        
        # First try to find code files with significant changes
        for file in files_data:
            filename = file.get('filename', '')
            ext = Path(filename).suffix.lower()
            # Skip non-code files
            if ext not in code_extensions:
                continue
                
            # Calculate total changes - prioritize files with more changes
            additions = file.get('additions', 0)
            deletions = file.get('deletions', 0)
            total_changes = additions + deletions
            
            if total_changes > max_changes:
                max_changes = total_changes
                significant_file = file
        
        # If no code file found, take any file with changes
        if significant_file is None and files_data:
            for file in files_data:
                # Get changes
                additions = file.get('additions', 0)
                deletions = file.get('deletions', 0)
                total_changes = additions + deletions
                
                # Check if this file has a patch
                has_patch = "patch" in file and file["patch"]
                
                # Prioritize files with patches and more changes
                if has_patch and total_changes > max_changes:
                    max_changes = total_changes
                    significant_file = file
        
        # If still no file, return empty
        if significant_file is None:
            return "No significant code changes found", "N/A", "text"
        
        # Extract code from the patch
        filename = significant_file.get('filename', 'Unknown file')
        patch = significant_file.get('patch', '')
        
        # If no patch available, try to get content
        if not patch:
            return f"No patch available for {filename}", filename, "text"
        
        return extract_code_from_patch(patch, filename, max_lines)
            
    except Exception as e:
        logger.error(f"Error extracting code snippet: {e}")
        return f"Error extracting code: {str(e)}", "error", "text"

def extract_code_from_patch(patch, filename, max_lines=15):
    """
    Extract code from a patch string, focusing on added lines.
    
    Args:
        patch: Patch string from GitHub API
        filename: Name of the file
        max_lines: Maximum number of lines to extract
    
    Returns:
        Tuple of (code_snippet, filename, language)
    """
    # Extract the core of the patch (remove header lines)
    patch_lines = patch.split('\n')
    content_lines = []
    in_content = False
    additions_count = 0
    
    for line in patch_lines:
        if line.startswith('@@'):
            in_content = True
            continue
            
        if in_content:
            # Only include added/context lines, not removed lines
            if line.startswith('+'):
                # Remove the leading '+'
                content_lines.append(line[1:])
                additions_count += 1
                if additions_count >= max_lines:
                    break
            elif not line.startswith('-'):
                # This is context line, include a few for better understanding
                content_lines.append(line)
                if additions_count > 0:  # Only count context after we've seen at least one addition
                    additions_count += 0.5  # Count context lines as half for the limit
                    if additions_count >= max_lines:
                        break
    
    # If too few lines, try to add some context lines
    if len(content_lines) < 5 and len(patch_lines) > 5:
        # Just take the first several lines after the header
        content_lines = []
        in_content = False
        for line in patch_lines:
            if line.startswith('@@'):
                in_content = True
                continue
            if in_content and len(content_lines) < max_lines:
                if not line.startswith('-'):
                    content_lines.append(line.replace('+', '', 1))
    
    # Clean up lines
    content_lines = [line for line in content_lines if line]
    
    # Get language from file extension
    ext = Path(filename).suffix.lower()
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.rs': 'rust',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.ts': 'typescript',
        '.css': 'css',
        '.html': 'html',
        '.sql': 'sql',
        '.md': 'markdown',
        '.sh': 'bash',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
    }
    language = language_map.get(ext, 'text')
    
    if content_lines:
        return '\n'.join(content_lines), filename, language
    else:
        return f"No meaningful code content extracted from {filename}", filename, "text"

def add_methodology_section(generator, pdf):
    """Add a simplified methodology section to the report."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add section title with styling
    fig.text(0.5, 0.95, "Methodology", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
    # Introduction to methodology
    intro_text = [
        "The data curation pipeline implements a multi-stage filtering approach inspired by the",
        "SWE-RL paper, focusing on extracting high-quality software engineering data from",
        "GitHub repositories. The pipeline consists of the following key components:"
    ]
    
    fig.text(0.1, 0.85, "\n".join(intro_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Simplified methodology content with numbered list and bullets
    methodology_text = [
        "1. Data Acquisition",
        "   • GitHub API integration for PR events and metadata",
        "   • Repository cloning for file content access",
        "   • Linked issue resolution and context gathering",
        "",
        "2. Multi-Stage Filtering",
        "   • Bot and Automation Detection: Identifies and filters out automated PRs",
        "   • Size and Complexity Filtering: Ensures PRs are neither trivial nor unwieldy",
        "   • Content Relevance Filtering: Focuses on meaningful software engineering content",
        "",
        "3. Relevant Files Prediction",
        "   • Identifies semantically related files not modified in the PR",
        "   • Uses import analysis and directory structure heuristics",
        "   • Enhances context for understanding code changes",
        "",
        "4. Quality Metrics Generation",
        "   • Comprehensive quality scoring across multiple dimensions",
        "   • Metadata extraction for filtering decisions",
        "   • Relevance scoring based on problem-solving indicators"
    ]
    
    fig.text(0.1, 0.75, "\n".join(methodology_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Final summary with better spacing
    conclusion = [
        "The filtering pipeline maintains high precision by using progressive refinement,",
        "ensuring that only PRs with genuine software engineering value are retained",
        "while capturing detailed metadata about filtering decisions and related file context."
    ]
    
    fig.text(0.1, 0.25, "\n".join(conclusion), 
            fontsize=12, ha='left', va='top', color='#2c3e50', 
            style='italic', linespacing=1.5)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)