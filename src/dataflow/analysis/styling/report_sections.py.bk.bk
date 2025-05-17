# #!/usr/bin/env python3
# """
# Report Sections Module - Simplified version to fix imports
# """
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib.patches import FancyBboxPatch, Rectangle
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_pdf import PdfPages
# import logging

# from dataflow.analysis.functionality.visualization_utils import (
#     add_gradient_background, 
#     add_gradient_line, 
#     get_quality_color,
#     add_styled_box, 
#     adjust_color_brightness
# )

# logger = logging.getLogger("enhanced_report_generator")

# def add_cover_page(generator, pdf):
#     """Add an attractive cover page to the report."""
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     ax.axis('off')
    
#     # Add title
#     fig.text(0.5, 0.7, "Enhanced Data Curation Report", 
#             fontsize=32, ha='center', va='center', weight='bold', color='#2c3e50')
    
#     # Add timestamp
#     fig.text(0.5, 0.3, f"Generated: {generator.timestamp}", 
#             fontsize=12, ha='center', va='center', color='#7f8c8d')
    
#     pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)

# def add_executive_summary(generator, pdf):
#     """Add executive summary with key metrics and charts."""
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     ax.axis('off')
    
#     fig.text(0.5, 0.95, "Executive Summary", 
#             fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
#     pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)

# def add_cross_repo_comparison(generator, pdf):
#     """Add cross-repository comparison section."""
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     ax.axis('off')
    
#     fig.text(0.5, 0.95, "Cross-Repository Comparison", 
#             fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
#     pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)

# def add_repo_section(generator, pdf, repo_key):
#     """Add detailed analysis section for a single repository."""
#     repo_name = repo_key.replace('_', '/')
    
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     ax.axis('off')
    
#     fig.text(0.5, 0.95, f"Repository Analysis: {repo_name}", 
#             fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
#     pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)

# def add_quality_profiles(generator, pdf):
#     """Add exemplary PR profiles showing different quality characteristics."""
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     ax.axis('off')
    
#     fig.text(0.5, 0.95, "Quality Profile Analysis", 
#             fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
#     pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)

# def add_pr_scorecard(generator, pdf, pr):
#     """Add an enhanced quality scorecard for a PR."""
#     pr_number = pr.get("pr_number")
#     repo_name = pr.get("repository")
    
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     ax.axis('off')
    
#     fig.text(0.5, 0.95, f"PR #{pr_number} Quality Scorecard - {repo_name}", 
#             fontsize=16, ha='center', weight='bold')
    
#     pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)

# def add_methodology_section(generator, pdf):
#     """Add an enhanced methodology section to the report."""
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     ax.axis('off')
    
#     fig.text(0.5, 0.95, "Methodology", 
#             fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
#     pdf.savefig(fig, bbox_inches='tight')
#     plt.close(fig)

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
    
    # Add a gradient background
    add_gradient_background(ax)
    
    # Add title with styling
    fig.text(0.5, 0.7, "Enhanced Data Curation Report", 
            fontsize=32, ha='center', va='center', weight='bold', color='#2c3e50')
    
    # Add subtitle
    fig.text(0.5, 0.63, "Comprehensive Analysis of GitHub PR Filtering Results", 
            fontsize=18, ha='center', va='center', color='#34495e')
    
    # Add horizontal line
    add_gradient_line(fig, 0.2, 0.8, 0.6)
    
    # Add summary statistics
    stats = generator.generate_summary_stats()
    
    summary_box = add_styled_box(fig, 0.25, 0.4, 0.5, 0.15, 
                               color='#f8f9fa', edge_color='#bdc3c7')
    
    # Add key statistics
    fig.text(0.5, 0.5, f"Total PRs Analyzed: {stats['total_prs']}", 
            fontsize=14, ha='center', va='center', color='#2c3e50')
    fig.text(0.5, 0.45, f"PRs Passed All Filters: {stats['passed_prs']} ({stats['pass_rate']:.1%})", 
            fontsize=14, ha='center', va='center', color='#2c3e50')
    fig.text(0.5, 0.4, f"Data Reduction: {stats['data_reduction']:.1%}", 
            fontsize=14, ha='center', va='center', color='#2c3e50')
    
    # Add timestamp and footer
    fig.text(0.5, 0.3, f"Generated: {generator.timestamp}", 
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
    
    # Add summary text
    summary_text = [
        f"This report provides a comprehensive analysis of the GitHub PR filtering results.",
        f"A total of {stats['total_prs']} PRs were processed through the multi-stage filtering pipeline,",
        f"with {stats['passed_prs']} PRs ({stats['pass_rate']:.1%}) passing all quality filters.",
        f"The filtering pipeline achieved a data reduction of {stats['data_reduction']:.1%}, focusing on",
        f"high-quality software engineering data while filtering out automated, trivial, or irrelevant PRs."
    ]
    
    fig.text(0.1, 0.87, "\n".join(summary_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Create a 2x2 grid of small subplots for summary visualizations
    gs = fig.add_gridspec(2, 2, left=0.1, right=0.9, bottom=0.3, top=0.8, wspace=0.3, hspace=0.4)
    
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
    
    # Add data labels
    for i, v in enumerate(sorted_rates):
        repo_ax.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=8)
    
    repo_ax.set_title("Pass Rate by Repository", fontsize=12, weight='bold')
    repo_ax.set_xlabel("Pass Rate", fontsize=10)
    repo_ax.set_xlim(0, max(repo_pass_rates) * 1.2)
    repo_ax.grid(axis='x', linestyle='--', alpha=0.3)
    repo_ax.set_yticks(range(len(sorted_repos)))
    repo_ax.set_yticklabels([repo[:15] + '...' if len(repo) > 15 else repo for repo in sorted_repos], fontsize=8)
    
    # 3. Filter rejection reasons (bottom left)
    reject_ax = fig.add_subplot(gs[1, 0])
    
    rejection_labels = ['Bot Filter', 'Size Filter', 'Content Filter']
    rejection_counts = [stats['bot_filtered'], stats['size_filtered'], stats['content_filtered']]
    
    if sum(rejection_counts) > 0:  # Only create pie chart if there are rejections
        # Create pie chart
        wedges, texts, autotexts = reject_ax.pie(
            rejection_counts, 
            labels=rejection_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(rejection_labels)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
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
                      color='#3498db',  # Use a single color instead of a palette
                      edgecolor='white', linewidth=0.5)
        
        quality_ax.set_title("Quality Score Distribution", fontsize=12, weight='bold')
        quality_ax.set_xlabel("Quality Score", fontsize=10)
        quality_ax.set_ylabel("Number of PRs", fontsize=10)
        quality_ax.grid(linestyle='--', alpha=0.3)
    else:
        quality_ax.text(0.5, 0.5, "No quality score data", ha='center', va='center')
        quality_ax.set_title("Quality Score Distribution (No Data)", fontsize=12, weight='bold')
    
    # Add key insights at the bottom
    key_findings = [
        f"Key Findings:",
        f"• {stats['bot_filtered_pct']:.1%} of PRs were filtered as likely automated or bot-created",
        f"• {stats['size_filtered_pct']:.1%} of PRs were filtered due to size or complexity issues",
        f"• {stats['content_filtered_pct']:.1%} of PRs were filtered due to content relevance issues",
        f"• Average quality score of passing PRs: {stats['avg_quality']:.2f} out of 1.0"
    ]
    
    fig.text(0.1, 0.2, "\n".join(key_findings), 
            fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
    
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
    
    # Add introduction text
    intro_text = [
        "This section compares metrics across all analyzed repositories to identify patterns",
        "and differences in PR quality, size, and filtering rates. The visualizations highlight",
        "repository-specific characteristics that impact data quality and filter effectiveness."
    ]
    
    fig.text(0.1, 0.87, "\n".join(intro_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Create a gridspec for more control over plot layout
    gs = fig.add_gridspec(2, 2, left=0.1, right=0.9, bottom=0.3, top=0.8, wspace=0.3, hspace=0.4)
    
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
    
    # Add data labels
    for i, v in enumerate(sorted_rates):
        pass_ax.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=8)
    
    pass_ax.set_title("PR Pass Rate by Repository", fontsize=12, weight='bold')
    pass_ax.set_xlabel("Pass Rate", fontsize=10)
    pass_ax.set_xlim(0, 1.0)
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
    
    # Add data labels
    for i, v in enumerate(q_sorted_quality):
        quality_ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=8)
    
    quality_ax.set_title("Average Quality Score by Repository", fontsize=12, weight='bold')
    quality_ax.set_xlabel("Quality Score (0-1)", fontsize=10)
    quality_ax.set_xlim(0, 1.0)
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
    
    # Create stacked bar chart
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
    filter_ax.set_xlim(0, 1.0)
    filter_ax.grid(axis='x', linestyle='--', alpha=0.3)
    filter_ax.set_yticks(range(len(f_sorted_repos)))
    filter_ax.set_yticklabels([repo[:12] + '...' if len(repo) > 12 else repo for repo in f_sorted_repos], fontsize=8)
    filter_ax.legend(fontsize=8, loc='lower right')
    
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
        
        # Create scatter plot
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
            
            # Add repository label
            size_ax.annotate(
                repo['repository'][:10] + ('...' if len(repo['repository']) > 10 else ''),
                (repo['file_count'], repo['total_changes']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
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
    
    # Add key insights at the bottom
    key_findings = [
        f"Key Insights:",
        f"• Repositories show significant variation in filter pass rates, from {min(pass_rates):.1%} to {max(pass_rates):.1%}",
        f"• Quality scores are generally higher in repositories with stricter filtering (lower pass rates)",
        f"• Bot activity varies substantially across repositories, affecting initial filtering stages",
        f"• PR size and complexity correlate with content quality in most repositories"
    ]
    
    fig.text(0.1, 0.2, "\n".join(key_findings), 
            fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
    
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
    
    # Add repository summary
    summary_text = [
        f"Total PRs Analyzed: {repo_stats.get('total_prs', 0)}",
        f"PRs Passed All Filters: {repo_stats.get('passed_prs', 0)} ({repo_stats.get('pass_rate', 0):.1%})",
        f"Average Quality Score: {repo_stats.get('avg_quality', 0):.2f} out of 1.0"
    ]
    
    fig.text(0.1, 0.87, "\n".join(summary_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Create a gridspec for more control over plot layout
    gs = fig.add_gridspec(2, 2, left=0.1, right=0.9, bottom=0.3, top=0.8, wspace=0.3, hspace=0.4)
    
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
    
    # Create pie chart
    if sum(filter_counts) > 0:
        filter_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        wedges, texts, autotexts = filter_ax.pie(
            filter_counts, 
            labels=filter_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=filter_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            textprops={'fontsize': 9}
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
        
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
            # Create scatter plot of PR size vs quality
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
            
            # Add color bar
            cbar = plt.colorbar(scatter, ax=size_ax)
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
            # Create radar chart (simplified as bar chart if radar not available)
            metric_bars = metrics_ax.bar(
                metrics.keys(),
                metrics.values(),
                color=sns.color_palette("viridis", len(metrics)),
                edgecolor='white',
                linewidth=0.5,
                alpha=0.7
            )
            
            # Add data labels
            for i, (key, value) in enumerate(metrics.items()):
                metrics_ax.text(i, value + 0.02, f"{value:.2f}", 
                              ha='center', fontsize=8)
            
            metrics_ax.set_title("Quality Metrics Comparison", fontsize=12, weight='bold')
            metrics_ax.set_ylim(0, 1.0)
            metrics_ax.set_ylabel("Score (0-1)", fontsize=10)
            metrics_ax.tick_params(axis='x', rotation=45, labelsize=8)
            metrics_ax.grid(axis='y', linestyle='--', alpha=0.3)
        else:
            metrics_ax.text(0.5, 0.5, "No quality metrics available", ha='center', va='center')
            metrics_ax.set_title("Quality Metrics Comparison (No Data)", fontsize=12, weight='bold')
    else:
        metrics_ax.text(0.5, 0.5, "No PR data available", ha='center', va='center')
        metrics_ax.set_title("Quality Metrics Comparison (No Data)", fontsize=12, weight='bold')
    
    # Add key insights at the bottom
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
    
    fig.text(0.1, 0.2, "\n".join(key_findings), 
            fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_quality_profiles(generator, pdf):
    """Add exemplary PR profiles showing different quality characteristics."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add section title with styling
    fig.text(0.5, 0.95, "Quality Profile Analysis", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
    # Add introduction text
    intro_text = [
        "This section presents detailed quality profiles for selected PRs that exemplify",
        "different quality levels and characteristics. The scorecards provide insights into",
        "how the filtering pipeline evaluates PRs across multiple dimensions."
    ]
    
    fig.text(0.1, 0.87, "\n".join(intro_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Find exemplary PRs to profile
    if not generator.pr_data.empty:
        # Find high quality PR (top 10%)
        high_quality_prs = generator.pr_data[
            generator.pr_data['passed_filter'] == True
        ].sort_values('quality_score', ascending=False).head(1)
        
        # Find medium quality PR (middle range)
        medium_quality_prs = generator.pr_data[
            (generator.pr_data['passed_filter'] == True) &
            (generator.pr_data['quality_score'] > 0.4) &
            (generator.pr_data['quality_score'] < 0.6)
        ].head(1)
        
        # Find PRs filtered at different stages
        bot_filtered_pr = generator.pr_data[
            generator.pr_data['passed_bot_filter'] == False
        ].head(1)
        
        size_filtered_pr = generator.pr_data[
            (generator.pr_data['passed_bot_filter'] == True) &
            (generator.pr_data['passed_size_filter'] == False)
        ].head(1)
        
        content_filtered_pr = generator.pr_data[
            (generator.pr_data['passed_bot_filter'] == True) &
            (generator.pr_data['passed_size_filter'] == True) &
            (generator.pr_data['passed_content_filter'] == False)
        ].head(1)
        
        # Add scorecards for each exemplary PR
        if not high_quality_prs.empty:
            pr = high_quality_prs.iloc[0].to_dict()
            pr["metadata"] = next((meta for meta in generator.repo_data.get(pr["repo_key"], {}).get("filter_metadata", []) 
                               if meta.get("pr_number") == pr["pr_number"]), {})
            add_pr_scorecard(generator, pdf, pr)
        
        if not medium_quality_prs.empty:
            pr = medium_quality_prs.iloc[0].to_dict()
            pr["metadata"] = next((meta for meta in generator.repo_data.get(pr["repo_key"], {}).get("filter_metadata", []) 
                               if meta.get("pr_number") == pr["pr_number"]), {})
            add_pr_scorecard(generator, pdf, pr)
        
        if not bot_filtered_pr.empty:
            pr = bot_filtered_pr.iloc[0].to_dict()
            pr["metadata"] = next((meta for meta in generator.repo_data.get(pr["repo_key"], {}).get("filter_metadata", []) 
                               if meta.get("pr_number") == pr["pr_number"]), {})
            add_pr_scorecard(generator, pdf, pr)
        
        if not size_filtered_pr.empty:
            pr = size_filtered_pr.iloc[0].to_dict()
            pr["metadata"] = next((meta for meta in generator.repo_data.get(pr["repo_key"], {}).get("filter_metadata", []) 
                               if meta.get("pr_number") == pr["pr_number"]), {})
            add_pr_scorecard(generator, pdf, pr)
        
        if not content_filtered_pr.empty:
            pr = content_filtered_pr.iloc[0].to_dict()
            pr["metadata"] = next((meta for meta in generator.repo_data.get(pr["repo_key"], {}).get("filter_metadata", []) 
                               if meta.get("pr_number") == pr["pr_number"]), {})
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
    """Add an enhanced quality scorecard for a PR with better formatting and spacing."""
    pr_number = pr.get("pr_number")
    repo_name = pr.get("repository")
    repo_key = repo_name.replace("/", "_")
    metadata = pr.get("metadata", {})
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Flatten axes for easier access
    axes = axes.flatten()
    
    # Add title with more space above to prevent overlap
    fig.suptitle(f"PR #{pr_number} Quality Scorecard - {repo_name}", 
                fontsize=16, weight='bold', y=0.98)
    
    # 1. Filter scores (top left)
    filter_scores = {
        'Bot Filter': 1.0 - metadata.get("bot_filter", {}).get("details", {}).get("confidence", 0.0),
        'Size Filter': metadata.get("size_filter", {}).get("details", {}).get("normalized_score", 0.0),
        'Content Filter': metadata.get("content_filter", {}).get("details", {}).get("relevance_score", 0.0),
        'Overall Quality': metadata.get("quality_score", 0.0)
    }
    
    # Use enhanced styling for bar chart
    colors = sns.color_palette("viridis", len(filter_scores))
    bars = axes[0].bar(filter_scores.keys(), filter_scores.values(), 
                     color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for i, (key, value) in enumerate(filter_scores.items()):
        axes[0].text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom', 
                   fontsize=9, weight='bold')
    
    # Enhance chart appearance
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Filter Scores", fontsize=12, weight='bold')
    axes[0].set_ylabel("Score (0-1)", fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add reference line for quality threshold
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    axes[0].text(0, 0.5, "Threshold", fontsize=8, color='red', 
               va='bottom', ha='left', alpha=0.7)
    
    # 2. File composition (top right)
    file_counts = {
        'Code': metadata.get("size_filter", {}).get("details", {}).get("code_file_count", 0),
        'Docs': metadata.get("size_filter", {}).get("details", {}).get("doc_file_count", 0),
        'Config': metadata.get("size_filter", {}).get("details", {}).get("config_file_count", 0),
        'Generated': metadata.get("size_filter", {}).get("details", {}).get("generated_file_count", 0),
        'Other': metadata.get("size_filter", {}).get("details", {}).get("other_file_count", 0)
    }
    
    # Remove zero values
    file_counts = {k: v for k, v in file_counts.items() if v > 0}
    
    if file_counts:
        # Use enhanced styling for pie chart
        wedges, texts, autotexts = axes[1].pie(
            file_counts.values(), 
            labels=file_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(file_counts)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            textprops={'fontsize': 9},
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
        
        axes[1].set_title("File Type Composition", fontsize=12, weight='bold')
    else:
        axes[1].text(0.5, 0.5, "No file data available", ha='center', va='center')
        axes[1].set_title("File Type Composition (No Data)", fontsize=12, weight='bold')
        axes[1].axis('off')
    
    # 3. Code changes (bottom left)
    change_data = {
        'Additions': metadata.get("size_filter", {}).get("details", {}).get("additions", 0),
        'Deletions': metadata.get("size_filter", {}).get("details", {}).get("deletions", 0)
    }
    
    # Use enhanced styling for bar chart
    bars = axes[2].bar(change_data.keys(), change_data.values(), 
                     color=['green', 'red'], edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for i, (key, value) in enumerate(change_data.items()):
        axes[2].text(i, value + max(change_data.values()) * 0.02, str(int(value)), 
                   ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Enhance chart appearance
    axes[2].set_title("Code Changes", fontsize=12, weight='bold')
    axes[2].set_ylabel("Number of Lines", fontsize=10)
    axes[2].grid(axis='y', linestyle='--', alpha=0.3)
    
    # 4. Relevant files (bottom right) - with improved spacing
    relevant_files = pr.get("relevant_files", [])
    num_relevant = len(relevant_files)
    
    if relevant_files:
        axes[3].axis('off')
        axes[3].set_title("Relevant Files", fontsize=12, weight='bold')
        
        # Create a styled list of relevant files with better spacing
        file_list = f"Files that provide context ({num_relevant} total):"
        axes[3].text(0.5, 0.95, file_list, 
                   ha='center', va='top', fontsize=10, weight='bold')
        
        # Show up to 6 files, with ellipsis if there are more
        display_files = relevant_files[:6]
        if len(relevant_files) > 6:
            display_files.append("... and more")
            
        # Use a cool background for the file list
        file_bg = Rectangle((0.1, 0.1), 0.8, 0.8, 
                          facecolor='#f8f9fa', alpha=0.5, 
                          edgecolor='#bdc3c7', linewidth=1)
        axes[3].add_patch(file_bg)
        
        # Position files with better spacing
        for i, file in enumerate(display_files):
            y_pos = 0.85 - (i * 0.09)  # Increased spacing
            
            # Use different styling for different file types
            if file.endswith(".py"):
                color = "#3572A5"  # Python color
                prefix = "PY "
            elif file.endswith(".js"):
                color = "#f1e05a"  # JavaScript color
                prefix = "JS "
            elif file.endswith(".md"):
                color = "#083fa1"  # Markdown color
                prefix = "MD "
            elif file.endswith(".json") or file.endswith(".yml") or file.endswith(".yaml"):
                color = "#cb171e"  # Config color
                prefix = "CF "
            elif "..." in file:
                color = "#666666"  # For ellipsis
                prefix = ""
            else:
                color = "#333333"  # Default color
                prefix = "FL "
            
            # Display filename with word wrapping for long filenames
            if len(file) > 30:
                # Split long filenames to multiple lines
                parts = file.split('/')
                if len(parts) > 2:
                    # Group directory parts on one line, filename on another
                    dir_path = '/'.join(parts[:-1])
                    file_name = parts[-1]
                    axes[3].text(0.15, y_pos, f"{prefix}{dir_path}/", 
                               ha='left', va='center', fontsize=8, color=color)
                    axes[3].text(0.25, y_pos-0.04, f"{file_name}", 
                               ha='left', va='center', fontsize=8, color=color)
                else:
                    axes[3].text(0.15, y_pos, f"{prefix}{file}", 
                               ha='left', va='center', fontsize=8, color=color)
            else:
                axes[3].text(0.15, y_pos, f"{prefix}{file}", 
                           ha='left', va='center', fontsize=9, color=color)
    else:
        axes[3].axis('off')
        axes[3].text(0.5, 0.5, "No relevant files identified", 
                   ha='center', va='center', fontsize=12)
        axes[3].set_title("Relevant Files", fontsize=12, weight='bold')
    
    # Add PR title and description at the bottom with better positioning
    pr_title = pr.get("title", "")
    pr_desc = pr.get("body", "")
    
    # Truncate description if too long
    if pr_desc and len(pr_desc) > 150:  # Further reduced for clarity
        pr_desc = pr_desc[:147] + "..."
        
    # Use a more subtle box for the PR details
    pr_box = Rectangle((0.05, 0.02), 0.9, 0.07, 
                     facecolor='#e8f4f8', alpha=0.5, 
                     edgecolor='#3498db', linewidth=1)
    fig.add_artist(pr_box)
    
    # Add title and truncated description with smaller font
    fig.text(0.07, 0.07, f"Title: {pr_title}", fontsize=9, weight='bold')
    if pr_desc:
        fig.text(0.07, 0.04, f"Description: {pr_desc}", fontsize=8, linespacing=1.2)
    
    # Add more space between sections
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make room for title and footer
    
    # Add to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_methodology_section(generator, pdf):
    """Add an enhanced methodology section to the report with better layout."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add a light background
    add_gradient_background(ax, alpha=0.1)
    
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
    
    # Add introduction text with better spacing
    fig.text(0.1, 0.87, "\n".join(intro_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Use styled boxes for each component with increased vertical spacing
    component_colors = ['#e8f8f5', '#eafaf1', '#fef9e7', '#fae5d3']
    component_borders = ['#1abc9c', '#2ecc71', '#f1c40f', '#e67e22']
    component_icons = ['1️⃣', '2️⃣', '3️⃣', '4️⃣']
    
    # Components with enhanced styling
    components = [
        ("1. Data Acquisition", [
            "• GitHub API integration for PR events and metadata",
            "• Repository cloning for file content access",
            "• Linked issue resolution and context gathering"
        ]),
        ("2. Multi-Stage Filtering", [
            "• Bot and Automation Detection: Identifies and filters out automated PRs",
            "• Size and Complexity Filtering: Ensures PRs are neither trivial nor unwieldy",
            "• Content Relevance Filtering: Focuses on meaningful software engineering content"
        ]),
        ("3. Relevant Files Prediction", [
            "• Identifies semantically related files not modified in the PR",
            "• Uses import analysis and directory structure heuristics",
            "• Enhances context for understanding code changes"
        ]),
        ("4. Quality Metrics Generation", [
            "• Comprehensive quality scoring across multiple dimensions",
            "• Metadata extraction for filtering decisions",
            "• Relevance scoring based on problem-solving indicators"
        ])
    ]
    
    # Position for components with increased spacing
    y_pos = 0.75
    for i, (title, details) in enumerate(components):
        # Create box with enhanced styling
        box_height = 0.13
        box = FancyBboxPatch((0.1, y_pos-box_height), 0.8, box_height, 
                           fill=True, facecolor=component_colors[i], alpha=0.7,
                           boxstyle="round,pad=0.02",
                           transform=fig.transFigure, edgecolor=component_borders[i], 
                           linewidth=2, zorder=1)
        ax.add_patch(box)
        
        # Add number and title with enhanced styling
        fig.text(0.15, y_pos-0.03, component_icons[i], fontsize=14, ha='left', 
                va='center', color=component_borders[i], weight='bold')
        fig.text(0.2, y_pos-0.03, title, fontsize=14, ha='left', 
                va='center', color='#34495e', weight='bold')
        
        # Add details with better styling and line spacing
        detail_text = "\n".join(details)
        fig.text(0.2, y_pos-0.06, detail_text, fontsize=10, 
                va='top', ha='left', color='#34495e', linespacing=1.3)
        
        # Increase spacing between components
        y_pos -= 0.18
    
    # Add process flow diagram with cleaner arrows
    ax.arrow(0.3, 0.35, 0, -0.05, head_width=0.02, head_length=0.01, 
            fc=component_borders[0], ec=component_borders[0], transform=fig.transFigure)
    ax.arrow(0.5, 0.35, 0, -0.05, head_width=0.02, head_length=0.01, 
            fc=component_borders[1], ec=component_borders[1], transform=fig.transFigure)
    ax.arrow(0.7, 0.35, 0, -0.05, head_width=0.02, head_length=0.01, 
            fc=component_borders[2], ec=component_borders[2], transform=fig.transFigure)
    
    # Final summary with enhanced styling
    summary_box = FancyBboxPatch((0.1, 0.1), 0.8, 0.1, fill=True, 
                               facecolor='#eaecee', alpha=0.7, 
                               boxstyle="round,pad=0.02",
                               transform=fig.transFigure,
                               edgecolor='#7f8c8d', linewidth=1)
    ax.add_patch(summary_box)
    
    conclusion = [
        "The filtering pipeline maintains high precision by using progressive refinement,",
        "ensuring that only PRs with genuine software engineering value are retained",
        "while capturing detailed metadata about filtering decisions and related file context."
    ]
    
    fig.text(0.5, 0.15, "\n".join(conclusion), fontsize=11, ha='center', 
            va='center', color='#2c3e50', style='italic', linespacing=1.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)