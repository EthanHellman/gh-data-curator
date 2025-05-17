#!/usr/bin/env python3
"""
Enhanced Report Generator

This module enhances the report generation capabilities with improved formatting
and additional cross-repository visualizations including clustering analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add src to Python path
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EnhancedReportGenerator:
    """
    Generate a comprehensive report on data curation filtering results
    with enhanced visualizations and clustering analysis.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the enhanced report generator.
        
        Args:
            data_dir: Base directory containing filtered PR data
        """
        self.data_dir = data_dir
        self.filtered_dir = data_dir / "filtered"
        self.results_dir = data_dir / "analysis_results"
        self.output_dir = data_dir / "reports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Get current timestamp for report naming
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Set up default styles for better appearance
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.figsize': (8.5, 11),
            'figure.dpi': 100,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
        })
        
        # Consistent color palettes
        self.color_palette = sns.color_palette("viridis", 6)
        self.repository_colors = {}
        
        # Create figures directory
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load repository data
        self.repo_data = self._load_repository_data()
        self.pr_data = self._load_pr_data()
    
    def _load_repository_data(self):
        """Load repository data from filtered directories."""
        repo_data = {}
        
        for repo_dir in self.filtered_dir.iterdir():
            if repo_dir.is_dir():
                repo_key = repo_dir.name
                metadata_path = repo_dir / "filter_metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        filter_metadata = json.load(f)
                    
                    # Load filtered PRs
                    filtered_index_path = repo_dir / "filtered_index.json"
                    filtered_prs = []
                    if filtered_index_path.exists():
                        with open(filtered_index_path, "r") as f:
                            filtered_prs = json.load(f)
                    
                    # Load metrics if available
                    metrics_path = self.results_dir / f"{repo_key}_metrics.json"
                    metrics = {}
                    if metrics_path.exists():
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                    
                    repo_data[repo_key] = {
                        "filter_metadata": filter_metadata,
                        "filtered_prs": filtered_prs,
                        "metrics": metrics
                    }
        
        # Assign consistent colors to repositories
        for i, repo_key in enumerate(sorted(repo_data.keys())):
            self.repository_colors[repo_key] = self.color_palette[i % len(self.color_palette)]
        
        return repo_data
    
    def _load_pr_data(self):
        """Load PR data into a DataFrame for analysis."""
        pr_records = []
        
        for repo_key, data in self.repo_data.items():
            for meta in data["filter_metadata"]:
                # Extract repository name for better display
                repo_name = repo_key.replace("_", "/")
                
                # Extract basic PR info
                pr_record = {
                    "repository": repo_name,
                    "repo_key": repo_key,
                    "pr_number": meta.get("pr_number", 0),
                    "passed_filter": meta.get("passed_filter", False),
                }
                
                # Extract bot filter data
                bot_filter = meta.get("bot_filter", {})
                pr_record.update({
                    "passed_bot_filter": bot_filter.get("passed", False),
                    "bot_confidence": bot_filter.get("details", {}).get("confidence", 0),
                })
                
                # Extract size filter data
                size_filter = meta.get("size_filter", {})
                size_details = size_filter.get("details", {})
                pr_record.update({
                    "passed_size_filter": size_filter.get("passed", False),
                    "file_count": size_details.get("total_files", 0),
                    "code_file_count": size_details.get("code_file_count", 0),
                    "total_changes": size_details.get("total_changes", 0),
                    "additions": size_details.get("additions", 0),
                    "deletions": size_details.get("deletions", 0),
                    "size_score": size_details.get("size_score", 0),
                    "complexity_score": size_details.get("complexity_score", 0),
                })
                
                # Extract content filter data
                content_filter = meta.get("content_filter", {})
                content_details = content_filter.get("details", {})
                pr_record.update({
                    "passed_content_filter": content_filter.get("passed", False),
                    "relevance_score": content_details.get("relevance_score", 0),
                    "problem_solving_score": content_details.get("problem_solving_score", 0),
                    "code_quality_score": content_details.get("code_quality_score", 0),
                })
                
                # Add overall quality score
                pr_record["quality_score"] = meta.get("quality_score", 0)
                
                pr_records.append(pr_record)
        
        # Create DataFrame
        return pd.DataFrame(pr_records)
    
    def generate_report(self) -> Path:
        """
        Generate a comprehensive PDF report with enhanced visualizations.
        
        Returns:
            Path to the generated report
        """
        logger.info("Generating enhanced comprehensive filtering report...")
        
        # Create output PDF
        report_path = self.output_dir / f"enhanced_data_curation_report_{self.timestamp}.pdf"
        
        with PdfPages(report_path) as pdf:
            # Cover page
            self._add_enhanced_cover_page(pdf)
            
            # Executive summary
            self._add_enhanced_executive_summary(pdf)
            
            # Cross-repository comparison
            self._add_enhanced_cross_repo_comparison(pdf)
            
            # Add enhanced cross-repository visualizations
            self._add_quality_scatter_plot(pdf)
            self._add_dimension_correlation_heatmap(pdf)
            self._add_pr_clustering_visualization(pdf)
            self._add_parallel_coordinates_plot(pdf)
            
            # Individual repository analyses
            for repo_key in sorted(self.repo_data.keys()):
                self._add_enhanced_repo_section(pdf, repo_key)
            
            # Add exemplary PR profiles
            self._add_enhanced_quality_profiles(pdf)
            
            # Methodology
            self._add_enhanced_methodology_section(pdf)
        
        logger.info(f"Enhanced report generated successfully: {report_path}")
        return report_path
    
    def _add_enhanced_cover_page(self, pdf):
        """Add an enhanced cover page to the report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Use a subtle gradient background
        gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
        ax.imshow(gradient, cmap='Blues', alpha=0.2, aspect='auto',
                 extent=[0, 1, 0, 1], transform=fig.transFigure)
        
        # Draw a decorative border with rounded corners
        from matplotlib.patches import FancyBboxPatch
        border_rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, fill=False, 
                                    ec='steelblue', lw=3, boxstyle="round,pad=0.02", 
                                    transform=fig.transFigure)
        fig.patches.extend([border_rect])
        
        # Add a logo-like element - more sophisticated
        logo_rect = plt.Rectangle((0.35, 0.75), 0.3, 0.1, fill=True, 
                                 fc='steelblue', ec='none', alpha=0.9,
                                 transform=fig.transFigure)
        fig.patches.extend([logo_rect])
        
        # Title with enhanced styling - add drop shadow effect
        for offset in [(0.003, -0.003), (0.002, -0.002), (0.001, -0.001)]:
            fig.text(0.5 + offset[0], 0.7 + offset[1], "Data Curation Pipeline", 
                    fontsize=28, ha='center', weight='bold', color='#2c3e50', alpha=0.3)
        
        fig.text(0.5, 0.7, "Data Curation Pipeline", 
                fontsize=28, ha='center', weight='bold', color='#2c3e50')
        fig.text(0.5, 0.62, "Filtering Results Report", 
                fontsize=22, ha='center', color='#34495e')
        
        # Date and time with nice formatting
        date_str = datetime.now().strftime("%B %d, %Y")
        fig.text(0.5, 0.53, date_str, 
                fontsize=16, ha='center', color='#7f8c8d', style='italic')
        
        # Repository information
        repo_count = len(self.repo_data)
        fig.text(0.5, 0.45, f"Analysis of {repo_count} GitHub Repositories", 
                fontsize=18, ha='center', color='#2980b9')
        
        # Add decorative elements with a data visualization theme
        icon_x = 0.5
        for i, (color, icon) in enumerate([
            ('#3498db', '📊'), 
            ('#2ecc71', '📈'), 
            ('#e74c3c', '🔍'), 
            ('#f39c12', '📉')
        ]):
            y_pos = 0.3 - (i * 0.03)
            fig.text(icon_x, y_pos, icon, fontsize=14, ha='center', color=color)
        
        # Footer
        fig.text(0.5, 0.15, "Enhanced Analysis with Clustering and Cross-Repository Insights", 
                fontsize=12, ha='center', style='italic', color='#7f8c8d')
        
        # Add version and timestamp in small text
        fig.text(0.5, 0.1, f"v2.0 • Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=9, ha='center', color='#95a5a6')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_enhanced_executive_summary(self, pdf):
        """Add an enhanced executive summary with improved formatting."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add a light background with gradient
        gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
        ax.imshow(gradient, cmap='Blues', alpha=0.1, aspect='auto',
                 extent=[0, 1, 0, 1], transform=fig.transFigure)
        
        # Section header with improved styling
        fig.text(0.5, 0.95, "Executive Summary", 
                fontsize=24, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title with gradient
        line_gradient = np.linspace(0.1, 0.9, 100)
        for i, x in enumerate(line_gradient):
            alpha = 1 - abs(2 * (x - 0.5))
            ax.plot([x, x+0.01], [0.92, 0.92], color='#3498db', alpha=alpha, linewidth=2, transform=fig.transFigure)
        
        # Generate summary statistics
        summary_stats = self._generate_enhanced_summary_stats()
        
        # Format text in a more readable way with improved layout
        summary_text = self._format_executive_summary(summary_stats)
        
        # Add the text with better styling and columns
        fig.text(0.1, 0.87, summary_text, 
                fontsize=11, va='top', ha='left', 
                linespacing=1.8, color='#333333',
                transform=fig.transFigure)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Include enhanced visualizations
        self._add_enhanced_filter_rates_chart(pdf)
        self._add_enhanced_data_reduction_chart(pdf)
    
    def _generate_enhanced_summary_stats(self):
        """Generate enhanced summary statistics from repository data."""
        total_prs = sum(len(data["filter_metadata"]) for data in self.repo_data.values())
        passed_prs = sum(len(data["filtered_prs"]) for data in self.repo_data.values())
        
        # Calculate filter statistics
        bot_filtered = 0
        size_filtered = 0
        content_filtered = 0
        
        for data in self.repo_data.values():
            for meta in data["filter_metadata"]:
                if not meta.get("bot_filter", {}).get("passed", False):
                    bot_filtered += 1
                elif not meta.get("size_filter", {}).get("passed", False):
                    size_filtered += 1
                elif not meta.get("content_filter", {}).get("passed", False):
                    content_filtered += 1
        
        # Calculate quality metrics
        quality_scores = []
        for data in self.repo_data.values():
            for meta in data["filter_metadata"]:
                if meta.get("passed_filter", False):
                    quality_scores.append(meta.get("quality_score", 0))
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Calculate repository-specific stats
        repo_stats = {}
        for repo_key, data in self.repo_data.items():
            repo_filtered_prs = len(data["filtered_prs"])
            repo_total_prs = len(data["filter_metadata"])
            repo_pass_rate = repo_filtered_prs / repo_total_prs if repo_total_prs > 0 else 0
            
            repo_quality_scores = [
                meta.get("quality_score", 0) 
                for meta in data["filter_metadata"] 
                if meta.get("passed_filter", False)
            ]
            repo_avg_quality = sum(repo_quality_scores) / len(repo_quality_scores) if repo_quality_scores else 0
            
            repo_stats[repo_key] = {
                "total_prs": repo_total_prs,
                "passed_prs": repo_filtered_prs,
                "pass_rate": repo_pass_rate,
                "avg_quality": repo_avg_quality
            }
        
        return {
            "total_prs": total_prs,
            "passed_prs": passed_prs,
            "pass_rate": passed_prs / total_prs if total_prs > 0 else 0,
            "bot_filtered": bot_filtered,
            "size_filtered": size_filtered,
            "content_filtered": content_filtered,
            "avg_quality": avg_quality,
            "repo_stats": repo_stats
        }
    
    def _format_executive_summary(self, stats):
        """Format the executive summary text with improved layout."""
        # Format summary text
        summary = [
            "This enhanced report presents the results of a data curation pipeline designed to extract",
            "and filter high-quality software engineering data from GitHub repositories. The analysis",
            "includes advanced cross-repository comparisons and clustering of pull requests.",
            "",
            f"Across {len(self.repo_data)} repositories, a total of {stats['total_prs']} PRs were processed through",
            f"the filtering pipeline, resulting in {stats['passed_prs']} high-quality PRs that passed all filters.",
            f"This represents an overall pass rate of {stats['pass_rate']:.1%}, with a data reduction ratio",
            f"of {1-stats['pass_rate']:.1%}.",
            "",
            "Filtering Breakdown:",
            f"• Bot Filter: {stats['bot_filtered']} PRs ({stats['bot_filtered']/stats['total_prs']:.1%} of total)",
            f"• Size/Complexity Filter: {stats['size_filtered']} PRs ({stats['size_filtered']/stats['total_prs']:.1%} of total)",
            f"• Content Relevance Filter: {stats['content_filtered']} PRs ({stats['content_filtered']/stats['total_prs']:.1%} of total)",
            "",
            f"The average quality score for passing PRs was {stats['avg_quality']:.2f} on a scale of 0-1.",
            "",
            "Repository Performance:",
        ]
        
        # Add repository-specific stats
        for repo_key, repo_stats in stats["repo_stats"].items():
            repo_name = repo_key.replace("_", "/")
            summary.append(f"• {repo_name}: {repo_stats['pass_rate']:.1%} pass rate, {repo_stats['avg_quality']:.2f} avg quality")
        
        summary.extend([
            "",
            "Key Findings:",
            "• Bot-generated PRs constitute a significant portion of repository activity",
            "• Size and complexity filters effectively remove both trivial and unwieldy changes",
            "• Content relevance filtering ensures focus on meaningful software engineering content",
            "• Cross-repository analysis reveals distinct quality patterns by project type",
            "• Clustering analysis identifies groups of PRs with similar characteristics",
            "",
            "The following pages provide detailed analyses, including cross-repository comparisons,",
            "quality clustering, and dimension correlations to provide deeper insights into the data."
        ])
        
        return "\n".join(summary)
    
    def _add_enhanced_filter_rates_chart(self, pdf):
        """Add an enhanced filter rates comparison chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for the chart
        repo_names = []
        bot_rates = []
        size_rates = []
        content_rates = []
        pass_rates = []
        
        for repo_key, data in sorted(self.repo_data.items()):
            repo_names.append(repo_key.replace("_", "/"))
            
            total_prs = len(data["filter_metadata"])
            if total_prs == 0:
                continue
                
            # Count PRs filtered at each stage
            bot_filtered = 0
            size_filtered = 0
            content_filtered = 0
            
            for meta in data["filter_metadata"]:
                if not meta.get("bot_filter", {}).get("passed", False):
                    bot_filtered += 1
                elif not meta.get("size_filter", {}).get("passed", False):
                    size_filtered += 1
                elif not meta.get("content_filter", {}).get("passed", False):
                    content_filtered += 1
            
            passed_prs = len(data["filtered_prs"])
            
            # Calculate rates
            bot_rates.append(bot_filtered / total_prs)
            size_rates.append(size_filtered / total_prs)
            content_rates.append(content_filtered / total_prs)
            pass_rates.append(passed_prs / total_prs)
        
        # Set up bar positions
        x = np.arange(len(repo_names))
        width = 0.2
        
        # Create grouped bar chart with enhanced styling
        bars1 = ax.bar(x - width*1.5, bot_rates, width, label='Bot Filter Rate', 
                     color=sns.color_palette("colorblind", 4)[0], alpha=0.85)
        bars2 = ax.bar(x - width/2, size_rates, width, label='Size Filter Rate', 
                     color=sns.color_palette("colorblind", 4)[1], alpha=0.85)
        bars3 = ax.bar(x + width/2, content_rates, width, label='Content Filter Rate', 
                     color=sns.color_palette("colorblind", 4)[2], alpha=0.85)
        bars4 = ax.bar(x + width*1.5, pass_rates, width, label='Overall Pass Rate', 
                     color=sns.color_palette("colorblind", 4)[3], alpha=0.85)
        
        # Add data labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, color='#333333')
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        add_labels(bars4)
        
        # Enhance chart appearance
        ax.set_xlabel('Repository', fontsize=12, weight='bold')
        ax.set_ylabel('Rate (percentage)', fontsize=12, weight='bold')
        ax.set_title('Filter Rates Comparison Across Repositories', 
                   fontsize=16, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(repo_names, rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Add grid for readability (horizontal only, more subtle)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Enhance legend
        ax.legend(title='Filter Type', frameon=True, framealpha=0.9, 
                 loc='upper right', fontsize=10)
        
        # Add subtle background shading to distinguish repositories
        for i in range(len(repo_names)):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.1)
        
        plt.tight_layout()
        
        # Save to figures directory for reference
        plt.savefig(self.figures_dir / "enhanced_cross_repo_filter_rates.png", dpi=300)
        
        # Add to PDF
        plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_enhanced_data_reduction_chart(self, pdf):
        """Add an enhanced data reduction comparison chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for the chart
        repo_names = []
        reduction_ratios = []
        pass_rates = []
        total_prs = []
        
        for repo_key, data in sorted(self.repo_data.items(), 
                                    key=lambda x: len(x[1]["filtered_prs"]) / len(x[1]["filter_metadata"]) 
                                    if len(x[1]["filter_metadata"]) > 0 else 0):
            repo_name = repo_key.replace("_", "/")
            repo_names.append(repo_name)
            
            repo_total = len(data["filter_metadata"])
            if repo_total == 0:
                reduction_ratios.append(0)
                pass_rates.append(0)
                total_prs.append(0)
                continue
                
            repo_passed = len(data["filtered_prs"])
            pass_rate = repo_passed / repo_total
            reduction_ratio = 1 - pass_rate
            
            reduction_ratios.append(reduction_ratio)
            pass_rates.append(pass_rate)
            total_prs.append(repo_total)
        
        # Create a horizontal bar chart with enhanced styling
        bars = ax.barh(repo_names, reduction_ratios, 
                      color=[sns.color_palette("viridis", len(repo_names))[i] 
                             for i in range(len(repo_names))])
        
        # Add a secondary axis for the pass rate (inverted reduction ratio)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_xticklabels(['100%', '80%', '60%', '40%', '20%', '0%'])
        ax2.set_xlabel('Pass Rate', fontsize=12, weight='bold')
        
        # Add data labels with total PR counts
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x = max(0.02, width - 0.05)  # Ensure label is visible
            
            # Add data reduction percentage
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{width:.0%} reduction', 
                   va='center', ha='right', color='white', 
                   fontweight='bold', fontsize=9)
            
            # Add total PR count and pass rate
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'({total_prs[i]} PRs, {pass_rates[i]:.0%} passed)', 
                   va='center', ha='left', fontsize=9, color='#555555')
        
        # Enhance chart appearance
        ax.set_xlabel('Data Reduction Ratio', fontsize=12, weight='bold')
        ax.set_ylabel('Repository', fontsize=12, weight='bold')
        ax.set_title('Data Reduction Ratio by Repository', 
                   fontsize=16, weight='bold', pad=20)
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Add grid for readability (vertical only, more subtle)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add subtle background shading to distinguish repositories
        for i in range(len(repo_names)):
            if i % 2 == 0:
                ax.axhspan(i - 0.4, i + 0.4, color='gray', alpha=0.1)
        
        plt.tight_layout()
        
        # Save to figures directory for reference
        plt.savefig(self.figures_dir / "enhanced_data_reduction.png", dpi=300)
        
        # Add to PDF
        plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_enhanced_cross_repo_comparison(self, pdf):
        """Add enhanced cross-repository comparison section."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add section title with enhanced styling
        fig.text(0.5, 0.95, "Cross-Repository Comparison", 
                fontsize=24, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title with gradient
        line_gradient = np.linspace(0.1, 0.9, 100)
        for i, x in enumerate(line_gradient):
            alpha = 1 - abs(2 * (x - 0.5))
            ax.plot([x, x+0.01], [0.92, 0.92], color='#3498db', alpha=alpha, linewidth=2, transform=fig.transFigure)
        
        # Add introduction text
        intro_text = [
            "This section presents comparative analyses across all repositories, highlighting",
            "patterns and differences in filtering performance and PR quality metrics.",
            "The visualizations provide insights into how different repositories compare",
            "in terms of data quality, filtering patterns, and clustering characteristics."
        ]
        
        fig.text(0.1, 0.87, "\n".join(intro_text), 
                fontsize=12, va='top', ha='left', linespacing=1.5, 
                color='#333333', transform=fig.transFigure)
        
        # Add quality metrics comparison chart
        self._add_quality_metrics_comparison_chart(fig, ax)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add cross-repository metrics heatmap as a separate page
        self._add_cross_repo_metrics_heatmap(pdf)
    
    def _add_quality_metrics_comparison_chart(self, fig, ax):
        """Add quality metrics comparison chart to the cross-repo page."""
        # Create axes for the chart
        chart_ax = fig.add_axes([0.1, 0.4, 0.8, 0.35])
        
        # Extract data for the chart
        repo_names = []
        quality_scores = []
        relevance_scores = []
        
        for repo_key, data in sorted(self.repo_data.items()):
            repo_names.append(repo_key.replace("_", "/"))
            
            # Average quality score for passing PRs
            repo_quality_scores = [
                meta.get("quality_score", 0) 
                for meta in data["filter_metadata"] 
                if meta.get("passed_filter", False)
            ]
            avg_quality = sum(repo_quality_scores) / len(repo_quality_scores) if repo_quality_scores else 0
            quality_scores.append(avg_quality)
            
            # Average relevance score for passing PRs
            repo_relevance_scores = [
                meta.get("content_filter", {}).get("details", {}).get("relevance_score", 0)
                for meta in data["filter_metadata"]
                if meta.get("passed_filter", False)
            ]
            avg_relevance = sum(repo_relevance_scores) / len(repo_relevance_scores) if repo_relevance_scores else 0
            relevance_scores.append(avg_relevance)
        
        # Set up bar positions
        x = np.arange(len(repo_names))
        width = 0.35
        
        # Create grouped bar chart with enhanced styling
        bars1 = chart_ax.bar(x - width/2, quality_scores, width, label='Avg Quality Score', 
                          color=sns.color_palette("viridis", 2)[0], alpha=0.8, edgecolor='white', linewidth=0.5)
        bars2 = chart_ax.bar(x + width/2, relevance_scores, width, label='Avg Relevance Score', 
                          color=sns.color_palette("viridis", 2)[1], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add data labels
        for bar in bars1:
            height = bar.get_height()
            chart_ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, color='#333333')
        
        for bar in bars2:
            height = bar.get_height()
            chart_ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, color='#333333')
        
        # Enhance chart appearance
        chart_ax.set_xlabel('Repository', fontsize=11, weight='bold')
        chart_ax.set_ylabel('Score (0-1)', fontsize=11, weight='bold')
        chart_ax.set_title('Quality Metrics Comparison Across Repositories', 
                         fontsize=14, weight='bold')
        chart_ax.set_xticks(x)
        chart_ax.set_xticklabels(repo_names, rotation=45, ha='right')
        
        # Set y-axis limits for better proportion
        chart_ax.set_ylim(0, 1.0)
        
        # Add grid for readability (horizontal only, more subtle)
        chart_ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Enhance legend
        chart_ax.legend(frameon=True, framealpha=0.9, fontsize=10)
        
        # Add subtle background shading to distinguish repositories
        for i in range(len(repo_names)):
            if i % 2 == 0:
                chart_ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.1)
    
    def _add_cross_repo_metrics_heatmap(self, pdf):
        """Add a heatmap of cross-repository metrics."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Extract data for the heatmap
        repo_keys = sorted(self.repo_data.keys())
        repo_names = [key.replace("_", "/") for key in repo_keys]
        
        # Metrics to include in the heatmap
        metrics = [
            'Total PRs', 'Passed PRs', 'Bot Filtered', 'Size Filtered', 
            'Content Filtered', 'Pass Rate (%)', 'Data Reduction (%)', 
            'Avg Quality Score'
        ]
        
        # Create data matrix
        data = np.zeros((len(repo_keys), len(metrics)))
        
        for i, repo_key in enumerate(repo_keys):
            repo_data = self.repo_data[repo_key]
            
            total_prs = len(repo_data["filter_metadata"])
            passed_prs = len(repo_data["filtered_prs"])
            
            # Count PRs filtered at each stage
            bot_filtered = 0
            size_filtered = 0
            content_filtered = 0
            
            for meta in repo_data["filter_metadata"]:
                if not meta.get("bot_filter", {}).get("passed", False):
                    bot_filtered += 1
                elif not meta.get("size_filter", {}).get("passed", False):
                    size_filtered += 1
                elif not meta.get("content_filter", {}).get("passed", False):
                    content_filtered += 1
            
            # Calculate pass rate and data reduction
            pass_rate = passed_prs / total_prs * 100 if total_prs > 0 else 0
            data_reduction = (1 - passed_prs / total_prs) * 100 if total_prs > 0 else 0
            
            # Calculate average quality score
            quality_scores = [
                meta.get("quality_score", 0) 
                for meta in repo_data["filter_metadata"] 
                if meta.get("passed_filter", False)
            ]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Fill the data matrix
            data[i, 0] = total_prs
            data[i, 1] = passed_prs
            data[i, 2] = bot_filtered
            data[i, 3] = size_filtered
            data[i, 4] = content_filtered
            data[i, 5] = pass_rate
            data[i, 6] = data_reduction
            data[i, 7] = avg_quality
        
        # Create a pandas DataFrame for the heatmap
        df = pd.DataFrame(data, index=repo_names, columns=metrics)
        
        # Create the heatmap with enhanced styling
        cmap = sns.color_palette("viridis", as_cmap=True)
        
        # Normalize each column separately for better visualization
        norm_data = df.copy()
        for col in df.columns:
            if df[col].max() > 0:
                norm_data[col] = df[col] / df[col].max()
        
        # Plot the heatmap
        sns.heatmap(norm_data, annot=df, fmt=".1f", cmap=cmap, 
                   linewidths=0.5, cbar=False, ax=ax)
        
        # Enhance the heatmap appearance
        ax.set_title("Cross-Repository Metrics Comparison", fontsize=16, weight='bold', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        # Add a note about normalization
        fig.text(0.5, 0.02, "Note: Colors represent normalized values within each column for better visibility.", 
                ha='center', fontsize=10, style='italic', color='#555555')
        
        plt.tight_layout()
        
        # Save to figures directory for reference
        plt.savefig(self.figures_dir / "enhanced_cross_repo_metrics_heatmap.png", dpi=300)
        
        # Add to PDF
        plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_quality_scatter_plot(self, pdf):
        """Add a quality scatter plot colored by repository."""
        # Filter for PRs that passed at least one filter for better visualization
        filtered_df = self.pr_data[self.pr_data['passed_bot_filter'] == True]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Set up colors for repositories
        colors = []
        for repo in filtered_df['repo_key']:
            colors.append(self.repository_colors.get(repo, 'gray'))
        
        # Create scatter plot with size as the marker size
        scatter = ax.scatter(
            filtered_df['size_score'], 
            filtered_df['relevance_score'],
            c=colors,
            s=filtered_df['code_file_count'] * 10 + 20,  # Adjust size based on code files
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Add labels for passed PRs
        passed_prs = filtered_df[filtered_df['passed_filter'] == True]
        texts = []
        for _, pr in passed_prs.iterrows():
            if pr['quality_score'] > 0.8:  # Only label high-quality PRs
                text = ax.text(
                    pr['size_score'], 
                    pr['relevance_score'], 
                    f"PR#{pr['pr_number']}",
                    fontsize=8,
                    ha='center',
                    va='center',
                    alpha=0.8
                )
                texts.append(text)
        
        # Adjust text positions to minimize overlap
        if texts:
            try:
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5))
            except:
                # If adjust_text fails, just use the original positions
                pass
        
        # Create a custom legend for repositories
        legend_elements = []
        for repo_key, color in self.repository_colors.items():
            if repo_key in filtered_df['repo_key'].values:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=color, markersize=10,
                                               label=repo_key.replace('_', '/')))
        
        # Add a legend for passed/failed PRs
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                        markeredgecolor='white', markersize=10, alpha=0.5,
                                        label='Failed Filters'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                        markeredgecolor='white', markersize=15, alpha=0.5,
                                        label='Larger PR (more files)'))
        
        # Add the legend to the plot
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        # Add reference lines
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Content Filter Threshold')
        ax.axvline(x=0.5, color='blue', linestyle='--', alpha=0.3, label='Size Filter Threshold')
        
        # Add diagonal line for reference (combined quality)
        x = np.linspace(0, 1, 100)
        y = x
        ax.plot(x, y, color='green', linestyle='--', alpha=0.3, label='Equal Weight Line')
        
        # Highlight different quality zones
        high_quality = plt.Rectangle((0.6, 0.6), 0.4, 0.4, color='green', alpha=0.1)
        low_quality = plt.Rectangle((0, 0), 0.4, 0.4, color='red', alpha=0.1)
        ax.add_patch(high_quality)
        ax.add_patch(low_quality)
        ax.text(0.8, 0.8, 'High Quality Zone', fontsize=9, ha='center', va='center', alpha=0.7)
        ax.text(0.2, 0.2, 'Low Quality Zone', fontsize=9, ha='center', va='center', alpha=0.7)
        
        # Enhance chart appearance
        ax.set_xlabel('Size Score (higher is better)', fontsize=12, weight='bold')
        ax.set_ylabel('Content Relevance Score', fontsize=12, weight='bold')
        ax.set_title('PR Quality Distribution by Repository', fontsize=16, weight='bold', pad=20)
        
        # Set axis limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.3)
        
        # Add annotations explaining the plot
        annotation_text = (
            "This plot shows the relationship between size score and content relevance score for PRs.\n"
            "Each point represents a PR, with color indicating repository and size indicating number of files.\n"
            "PRs that pass all filters are labeled with their PR number.\n"
            "High-quality PRs appear in the upper-right quadrant."
        )
        fig.text(0.5, 0.01, annotation_text, ha='center', fontsize=9, style='italic',
                color='#555555', bbox=dict(facecolor='white', alpha=0.7, pad=5))
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # Adjust layout to make room for the annotation
        
        # Save to figures directory for reference
        plt.savefig(self.figures_dir / "quality_scatter_plot.png", dpi=300)
        
        # Add to PDF
        plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_dimension_correlation_heatmap(self, pdf):
        """Add a correlation heatmap of quality dimensions."""
        # Select numerical features for correlation analysis
        features = [
            'bot_confidence', 'file_count', 'code_file_count', 'total_changes',
            'additions', 'deletions', 'size_score', 'complexity_score',
            'relevance_score', 'problem_solving_score', 'code_quality_score',
            'quality_score'
        ]
        
        # Filter data to PRs that have been processed by all filters
        corr_data = self.pr_data[self.pr_data['passed_bot_filter'] == True][features]
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a more readable correlation heatmap
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True  # Mask upper triangle
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .7}, annot=True,
                   fmt=".2f", annot_kws={"size": 8})
        
        # Improve feature labels for readability
        feature_labels = [
            'Bot Confidence', 'File Count', 'Code Files', 'Total Changes',
            'Additions', 'Deletions', 'Size Score', 'Complexity Score',
            'Relevance Score', 'Problem Solving', 'Code Quality',
            'Overall Quality'
        ]
        
        ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(feature_labels, rotation=0, fontsize=10)
        
        # Enhance chart appearance
        ax.set_title('Correlation Between Quality Dimensions', fontsize=16, weight='bold', pad=20)
        
        # Add annotations explaining key correlations
        # Find the 3 strongest positive and negative correlations (excluding self-correlations)
        correlations = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                correlations.append((feature_labels[i], feature_labels[j], corr_matrix.iloc[i, j]))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        top_correlations = correlations[:5]
        
        # Create annotation text
        correlation_text = "Notable Correlations:\n"
        for feat1, feat2, corr in top_correlations:
            correlation_text += f"• {feat1} & {feat2}: {corr:.2f}\n"
        
        # Add the annotation to the figure
        fig.text(0.15, 0.02, correlation_text, ha='left', fontsize=9,
                color='#333333', bbox=dict(facecolor='white', alpha=0.7, pad=5))
        
        # Add interpretation note
        interpretation = (
            "Correlation values range from -1 (strong negative) to +1 (strong positive).\n"
            "This heatmap reveals how different quality metrics relate to each other across all repositories."
        )
        fig.text(0.65, 0.02, interpretation, ha='left', fontsize=9, style='italic',
                color='#555555', bbox=dict(facecolor='white', alpha=0.7, pad=5))
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # Adjust layout to make room for annotations
        
        # Save to figures directory for reference
        plt.savefig(self.figures_dir / "dimension_correlation_heatmap.png", dpi=300)
        
        # Add to PDF
        plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_pr_clustering_visualization(self, pdf):
        """Add a clustering visualization of PRs using PCA."""
        # Select features for clustering
        features = [
            'file_count', 'code_file_count', 'total_changes',
            'size_score', 'complexity_score', 'relevance_score', 
            'problem_solving_score', 'code_quality_score'
        ]
        
        # Filter data to PRs that have been processed by at least the bot filter
        cluster_data = self.pr_data[self.pr_data['passed_bot_filter'] == True]
        
        # Check if we have enough data
        if len(cluster_data) < 10:
            logger.warning("Not enough data for clustering visualization")
            return
        
        # Prepare data for PCA
        X = cluster_data[features]
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
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
        
        # Define colors for clusters and repositories
        cluster_colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
        
        # Create scatter plot with custom styling
        scatter = ax.scatter(
            pca_df['PCA1'], 
            pca_df['PCA2'],
            c=[cluster_colors[label] for label in pca_df['Cluster']],
            s=pca_df['Quality'] * 100 + 30,  # Size based on quality score
            alpha=0.7,
            edgecolors=[self.repository_colors.get(repo, 'gray') for repo in pca_df['repo_key']],
            linewidth=2
        )
        
        # Add markers for passed PRs
        passed_prs = pca_df[pca_df['Passed'] == True]
        ax.scatter(
            passed_prs['PCA1'],
            passed_prs['PCA2'],
            s=passed_prs['Quality'] * 100 + 30,
            facecolors='none',
            edgecolors='black',
            linewidth=2,
            alpha=0.5
        )
        
        # Add labels for notable PRs (high quality or cluster centroids)
        notable_prs = pca_df[(pca_df['Quality'] > 0.8) | 
                            (pca_df['PCA1'].abs() > np.percentile(pca_df['PCA1'].abs(), 90)) |
                            (pca_df['PCA2'].abs() > np.percentile(pca_df['PCA2'].abs(), 90))]
        
        texts = []
        for _, pr in notable_prs.iterrows():
            text = ax.text(
                pr['PCA1'], 
                pr['PCA2'], 
                f"PR#{pr['PR']}",
                fontsize=8,
                ha='center',
                va='center',
                alpha=0.8
            )
            texts.append(text)
        
        # Adjust text positions to minimize overlap
        if texts:
            try:
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5))
            except:
                # If adjust_text fails, just use the original positions
                pass
        
        # Add cluster centers
        centers = pca.transform(scaler.transform(kmeans.cluster_centers_))
        ax.scatter(
            centers[:, 0], 
            centers[:, 1], 
            s=200,
            marker='X',
            color=cluster_colors,
            edgecolor='black',
            linewidth=2,
            alpha=0.8,
            label='Cluster Centers'
        )
        
        # Add labels for cluster centers
        for i, (x, y) in enumerate(centers):
            ax.text(x, y + 0.2, f'Cluster {i+1}', fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
        
        # Create a custom legend for repositories
        legend_elements = []
        for repo_key, color in self.repository_colors.items():
            if repo_key in pca_df['repo_key'].values:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor='gray', markeredgecolor=color,
                                               markersize=10, markeredgewidth=2,
                                               label=repo_key.replace('_', '/')))
        
        # Add legend elements for clusters
        for i, color in enumerate(cluster_colors):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=color, markersize=10,
                                           label=f'Cluster {i+1}'))
        
        # Add legend element for passed PRs
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor='none', markeredgecolor='black',
                                       markersize=10, markeredgewidth=2,
                                       label='Passed All Filters'))
        
        # Add the legend to the plot
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Enhance chart appearance
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                    fontsize=12, weight='bold')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                    fontsize=12, weight='bold')
        ax.set_title('PR Clustering Analysis', fontsize=16, weight='bold', pad=20)
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.3)
        
        # Add cluster interpretations based on feature loadings
        feature_loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=features
        )
        
        # Find top contributing features for each principal component
        pc1_features = feature_loadings.sort_values('PC1', key=abs, ascending=False)['PC1'].head(3)
        pc2_features = feature_loadings.sort_values('PC2', key=abs, ascending=False)['PC2'].head(3)
        
        # Format nice feature names
        feature_names = {
            'file_count': 'File Count',
            'code_file_count': 'Code Files',
            'total_changes': 'Total Changes',
            'size_score': 'Size Score',
            'complexity_score': 'Complexity Score',
            'relevance_score': 'Relevance Score',
            'problem_solving_score': 'Problem Solving',
            'code_quality_score': 'Code Quality'
        }
        
        # Create interpretation text
        pc1_text = "PC1 reflects: " + ", ".join([f"{feature_names.get(feat, feat)} ({val:.2f})" 
                                             for feat, val in pc1_features.items()])
        pc2_text = "PC2 reflects: " + ", ".join([f"{feature_names.get(feat, feat)} ({val:.2f})" 
                                             for feat, val in pc2_features.items()])
        
        interpretation = (
            "This plot shows PRs clustered by their quality dimensions, reduced to 2 dimensions via PCA.\n"
            f"{pc1_text}\n"
            f"{pc2_text}\n"
            "Point size indicates quality score. Marker edge color indicates repository.\n"
            "Black outlines indicate PRs that passed all filters."
        )
        
        fig.text(0.5, 0.01, interpretation, ha='center', fontsize=9, style='italic',
                color='#555555', bbox=dict(facecolor='white', alpha=0.7, pad=5))
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.98])  # Adjust layout to make room for the annotation
        
        # Save to figures directory for reference
        plt.savefig(self.figures_dir / "pr_clustering.png", dpi=300)
        
        # Add to PDF
        plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # def _add_parallel_coordinates_plot(self, pdf):
    #     """Add a parallel coordinates plot to visualize PR dimensions across repositories."""
    #     # Select features for the parallel plot
    #     features = [
    #         'file_count', 'code_file_count', 
    #         'size_score', 'relevance_score', 
    #         'problem_solving_score', 'quality_score'
    #     ]
        
    #     # Readable feature names for the plot
    #     feature_names = {
    #         'file_count': 'File Count',
    #         'code_file_count': 'Code Files',
    #         'size_score': 'Size Score',
    #         'relevance_score': 'Relevance Score',
    #         'problem_solving_score': 'Problem Solving',
    #         'quality_score': 'Quality Score'
    #     }
        
    #     # Filter to PRs that pass at least the bot filter
    #     parallel_data = self.pr_data[self.pr_data['passed_bot_filter'] == True]
        
    #     # Normalize the data for better visualization
    #     normalized_data = parallel_data.copy()
    #     for feature in features:
    #         if feature in ['file_count', 'code_file_count']:
    #             # For count features, use log scaling
    #             max_val = parallel_data[feature].max()
    #             if max_val > 0:
    #                 normalized_data[feature] = np.log1p(parallel_data[feature]) / np.log1p(max_val)
    #         else:
    #             # For score features, they're already in [0, 1]
    #             pass
        
    #     # Add repository and filter status for coloring
    #     normalized_data['Repository'] = parallel_data['repository']
    #     normalized_data['Passed'] = parallel_data['passed_filter']
        
    #     # Create the figure
    #     fig, ax = plt.subplots(figsize=(10, 6))
        
    #     # Create parallel coordinates plot
    #     # We'll use pandas plotting which is simpler than creating a custom parallel plot
        
    #     # Get only the features we want to plot
    #     plot_data = normalized_data[features + ['Repository', 'Passed']]
        
    #     # Rename columns for better display
    #     plot_data = plot_data.rename(columns=feature_names)
        
    #     # Create a color map based on repositories
    #     repo_colors = {
    #         repo: self.repository_colors.get(key, 'gray')
    #         for key, repo in zip(parallel_data['repo_key'], parallel_data['repository'])
    #     }
        
    #     # Group by repository and filter status
    #     grouped = plot_data.groupby(['Repository', 'Passed'])
        
    #     # Plot each group with appropriate color and alpha
    #     for (repo, passed), group in grouped:
    #         color = repo_colors.get(repo, 'gray')
    #         alpha = 0.8 if passed else 0.3
    #         linestyle = '-' if passed else ':'
    #         label = f"{repo} - {'Passed' if passed else 'Failed'}"
            
    #         pd.plotting.parallel_coordinates(
    #             group.drop(['Repository', 'Passed'], axis=1),
    #             'Repository',  # Dummy class column, will be ignored since we're feeding only one class
    #             color=color,
    #             alpha=alpha,
    #             linewidth=1.5 if passed else 1.0,
    #             linestyle=linestyle,
    #             ax=ax
    #         )
        
    #     # Clean up the plot
    #     ax.get_legend().remove()
        
    #     # Create a custom legend
    #     legend_elements = []
    #     for repo in sorted(plot_data['Repository'].unique()):
    #         color = repo_colors.get(repo, 'gray')
    #         legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=repo))
        
    #     # Add passed/failed to legend
    #     legend_elements.append(plt.Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Passed Filters'))
    #     legend_elements.append(plt.Line2D([0], [0], color='gray', lw=1, linestyle=':', label='Failed Filters'))
        
    #     # Add the legend to the plot
    #     ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
        
    #     # Enhance chart appearance
    #     ax.set_title('Multi-dimensional PR Quality Comparison', fontsize=16, weight='bold', pad=20)
        
    #     # Adjust axis labels for readability
    #     ax.set_ylim(-0.05, 1.05)
    #     ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    #     ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
        
    #     # Add annotations explaining the plot
    #     annotation_text = (
    #         "This parallel coordinates plot shows how PRs vary across multiple dimensions simultaneously.\n"
    #         "Each line represents a PR, with repository indicated by color and filter status by line style.\n"
    #         "File and code counts are log-normalized. All other dimensions are on a 0-1 scale."
    #     )
    #     fig.text(0.5, 0.01, annotation_text, ha='center', fontsize=9, style='italic',
    #             color='#555555', bbox=dict(facecolor='white', alpha=0.7, pad=5))
        
    #     plt.tight_layout(rect=[0, 0.07, 1, 0.98])  # Adjust layout to make room for the annotation
        
    #     # Save to figures directory for reference
    #     plt.savefig(self.figures_dir / "parallel_coordinates_plot.png", dpi=300)
        
    #     # Add to PDF
    #     plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    #     pdf.savefig(fig, bbox_inches='tight')
    #     plt.close(fig)

    def _add_parallel_coordinates_plot(self, pdf):
        """Add a parallel coordinates plot to visualize PR dimensions across repositories."""
        # Select features for the parallel plot
        features = [
            'file_count', 'code_file_count', 
            'size_score', 'relevance_score', 
            'problem_solving_score', 'quality_score'
        ]
        
        # Readable feature names for the plot
        feature_names = {
            'file_count': 'File Count',
            'code_file_count': 'Code Files',
            'size_score': 'Size Score',
            'relevance_score': 'Relevance Score',
            'problem_solving_score': 'Problem Solving',
            'quality_score': 'Quality Score'
        }
        
        # Filter to PRs that pass at least the bot filter
        parallel_data = self.pr_data[self.pr_data['passed_bot_filter'] == True]
        
        if len(parallel_data) < 5:  # Not enough data for a meaningful plot
            logger.warning("Not enough data for parallel coordinates plot")
            return
        
        # Normalize the data for better visualization
        normalized_data = parallel_data.copy()
        for feature in features:
            if feature in ['file_count', 'code_file_count']:
                # For count features, use log scaling
                max_val = parallel_data[feature].max()
                if max_val > 0:
                    normalized_data[feature] = np.log1p(parallel_data[feature]) / np.log1p(max_val)
            else:
                # For score features, they're already in [0, 1]
                pass
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a color map based on repositories
        repo_colors = {
            repo_key: self.repository_colors.get(repo_key, 'gray')
            for repo_key in parallel_data['repo_key'].unique()
        }
        
        # Plot each repository's data separately
        for repo_key in parallel_data['repo_key'].unique():
            # Get data for this repository
            repo_data = normalized_data[normalized_data['repo_key'] == repo_key]
            
            # Skip if not enough data
            if len(repo_data) < 2:
                continue
            
            # Plot passed and failed PRs separately
            for passed, group_data in repo_data.groupby('passed_filter'):
                if len(group_data) < 1:
                    continue
                    
                color = repo_colors.get(repo_key, 'gray')
                alpha = 0.8 if passed else 0.3
                linestyle = '-' if passed else ':'
                
                # Manual implementation of parallel coordinates
                # First, get features only
                df = group_data[features].copy()
                
                # Add a dummy class column (all same value)
                df['dummy_class'] = 1
                
                # Plot
                pd.plotting.parallel_coordinates(
                    df,
                    'dummy_class',  # Use the dummy class column
                    color=color,
                    alpha=alpha,
                    linewidth=1.5 if passed else 1.0,
                    linestyle=linestyle,
                    ax=ax
                )
        
        # Clean up the plot
        if hasattr(ax, 'legend_') and ax.legend_:
            ax.legend_.remove()
        
        # Rename the x-tick labels to our readable feature names
        ax.set_xticklabels([feature_names.get(feat, feat) for feat in features])
        
        # Create a custom legend
        legend_elements = []
        for repo_key in sorted(parallel_data['repo_key'].unique()):
            repo_name = parallel_data[parallel_data['repo_key'] == repo_key]['repository'].iloc[0] if len(parallel_data[parallel_data['repo_key'] == repo_key]) > 0 else repo_key.replace('_', '/')
            color = repo_colors.get(repo_key, 'gray')
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=repo_name))
        
        # Add passed/failed to legend
        legend_elements.append(plt.Line2D([0], [0], color='gray', lw=2, linestyle='-', label='Passed Filters'))
        legend_elements.append(plt.Line2D([0], [0], color='gray', lw=1, linestyle=':', label='Failed Filters'))
        
        # Add the legend to the plot
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
        
        # Enhance chart appearance
        ax.set_title('Multi-dimensional PR Quality Comparison', fontsize=16, weight='bold', pad=20)
        
        # Adjust axis labels for readability
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
        
        # Add annotations explaining the plot
        annotation_text = (
            "This parallel coordinates plot shows how PRs vary across multiple dimensions simultaneously.\n"
            "Each line represents a PR, with repository indicated by color and filter status by line style.\n"
            "File and code counts are log-normalized. All other dimensions are on a 0-1 scale."
        )
        fig.text(0.5, 0.01, annotation_text, ha='center', fontsize=9, style='italic',
                color='#555555', bbox=dict(facecolor='white', alpha=0.7, pad=5))
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.98])  # Adjust layout to make room for the annotation
        
        # Save to figures directory for reference
        plt.savefig(self.figures_dir / "parallel_coordinates_plot.png", dpi=300)
        
        # Add to PDF
        plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_enhanced_repo_section(self, pdf, repo_key):
        """Add an enhanced repository-specific section to the report."""
        repo_name = repo_key.replace("_", "/")
        repo_data = self.repo_data.get(repo_key, {})
        if not repo_data:
            logger.warning(f"No data found for repository {repo_key}")
            return
        
        filter_metadata = repo_data.get("filter_metadata", [])
        filtered_prs = repo_data.get("filtered_prs", [])
        metrics = repo_data.get("metrics", {})
        
        if not filter_metadata:
            logger.warning(f"No filter metadata found for repository {repo_key}")
            return
        
        # Create basic info page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add repository styling with a gradient background
        gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
        ax.imshow(gradient, cmap='Blues', alpha=0.05, aspect='auto',
                 extent=[0, 1, 0, 1], transform=fig.transFigure)
        
        # Repository title with enhanced styling
        fig.text(0.5, 0.95, f"Repository: {repo_name}", 
                fontsize=22, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title with gradient
        line_gradient = np.linspace(0.1, 0.9, 100)
        for i, x in enumerate(line_gradient):
            alpha = 1 - abs(2 * (x - 0.5))
            ax.plot([x, x+0.01], [0.92, 0.92], color='#3498db', alpha=alpha, linewidth=2, transform=fig.transFigure)
        
        # Create a styled box for the summary metrics
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((0.1, 0.75), 0.8, 0.12, fill=True, 
                           facecolor='#ecf0f1', alpha=0.8, 
                           boxstyle="round,pad=0.02",
                           transform=fig.transFigure,
                           edgecolor='#bdc3c7', linewidth=1)
        ax.add_patch(box)
        
        # Calculate repository metrics
        total_prs = len(filter_metadata)
        passed_prs = len(filtered_prs)
        pass_rate = passed_prs / total_prs if total_prs > 0 else 0
        data_reduction = 1 - pass_rate
        
        # Count PRs filtered at each stage
        bot_filtered = sum(1 for meta in filter_metadata if not meta.get("bot_filter", {}).get("passed", False))
        size_filtered = sum(1 for meta in filter_metadata 
                           if meta.get("bot_filter", {}).get("passed", False) 
                           and not meta.get("size_filter", {}).get("passed", False))
        content_filtered = sum(1 for meta in filter_metadata 
                              if meta.get("bot_filter", {}).get("passed", False) 
                              and meta.get("size_filter", {}).get("passed", False) 
                              and not meta.get("content_filter", {}).get("passed", False))
        
        # Calculate average quality score
        quality_scores = [meta.get("quality_score", 0) for meta in filter_metadata if meta.get("passed_filter", False)]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Summary metrics in styled box
        summary_text = [
            f"Total PRs: {total_prs}",
            f"Passed PRs: {passed_prs} ({pass_rate:.1%})",
            f"Data Reduction: {data_reduction:.1%}",
            f"Average Quality Score: {avg_quality:.2f}"
        ]
        
        fig.text(0.5, 0.81, "\n".join(summary_text), 
                fontsize=12, va='center', ha='center', color='#2c3e50', linespacing=1.8)
        
        # Add filtering breakdown
        fig.text(0.1, 0.7, "Filtering Breakdown:", 
                fontsize=14, va='top', weight='bold', color='#2c3e50')
        
        filter_breakdown = [
            f"• Bot Filter: {bot_filtered} PRs ({bot_filtered/total_prs:.1%})",
            f"• Size Filter: {size_filtered} PRs ({size_filtered/total_prs:.1%})",
            f"• Content Filter: {content_filtered} PRs ({content_filtered/total_prs:.1%})"
        ]
        
        fig.text(0.12, 0.65, "\n".join(filter_breakdown), 
                fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
        
        # Add repository visualization - filter funnel
        self._add_enhanced_filter_funnel(fig, ax, repo_key, filter_metadata)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add quality distribution in a separate page
        self._add_enhanced_quality_distribution(pdf, repo_key, filter_metadata)
        
        # Add filter analysis in a separate page
        self._add_enhanced_filter_analysis(pdf, repo_key, filter_metadata)
    
    def _add_enhanced_filter_funnel(self, fig, ax, repo_key, filter_metadata):
        """Add an enhanced filter funnel visualization to the repository page."""
        # Create axes for the funnel chart
        funnel_ax = fig.add_axes([0.1, 0.25, 0.8, 0.35])
        
        # Calculate funnel data
        total_prs = len(filter_metadata)
        
        # Count PRs at each stage
        after_bot = sum(1 for meta in filter_metadata if meta.get("bot_filter", {}).get("passed", False))
        after_size = sum(1 for meta in filter_metadata 
                        if meta.get("bot_filter", {}).get("passed", False) 
                        and meta.get("size_filter", {}).get("passed", False))
        after_content = sum(1 for meta in filter_metadata if meta.get("passed_filter", False))
        
        # Funnel data
        stages = ['Total PRs', 'After Bot Filter', 'After Size Filter', 'After Content Filter']
        counts = [total_prs, after_bot, after_size, after_content]
        
        # Create funnel with enhanced styling
        # Use a gradient of the repository color
        repo_color = self.repository_colors.get(repo_key, 'steelblue')
        colors = [repo_color] + [self._adjust_color_brightness(repo_color, factor) 
                               for factor in [0.8, 0.6, 0.4]]
        
        bars = funnel_ax.bar(stages, counts, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
        
        # Add count and percentage labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = height / total_prs * 100 if total_prs > 0 else 0
            
            funnel_ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                         f"{int(height)}\n({percentage:.1f}%)", 
                         ha='center', va='bottom', fontsize=10, weight='bold',
                         color='#333333')
        
        # Enhance chart appearance
        funnel_ax.set_title('Filter Funnel Analysis', fontsize=14, weight='bold')
        funnel_ax.set_ylabel('Number of PRs', fontsize=11)
        funnel_ax.set_ylim(0, total_prs * 1.2)  # Add space for labels
        funnel_ax.set_xticks(range(len(stages)))
        funnel_ax.set_xticklabels(stages, rotation=15, ha='center')
        
        # Add horizontal grid lines
        funnel_ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add arrows between stages to emphasize the flow
        for i in range(len(counts) - 1):
            # Calculate arrow positions
            x1 = i + 0.25
            x2 = i + 0.75
            y1 = counts[i] * 0.05
            y2 = counts[i+1] * 0.05
            
            # Add an arrow
            funnel_ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5, alpha=0.6))
    
    def _add_enhanced_quality_distribution(self, pdf, repo_key, filter_metadata):
        """Add an enhanced quality score distribution visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract quality scores
        quality_scores = [meta.get("quality_score", 0) for meta in filter_metadata]
        
        if not quality_scores:
            logger.warning(f"No quality scores found for {repo_key}")
            return
        
        # Create histogram with KDE and enhanced styling
        sns.histplot(quality_scores, kde=True, bins=20, 
                    color=self.repository_colors.get(repo_key, 'steelblue'),
                    alpha=0.7, edgecolor='white', linewidth=0.5, ax=ax)
        
        # Add mean and median lines with enhanced styling
        mean_score = np.mean(quality_scores)
        median_score = np.median(quality_scores)
        
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                 label=f'Mean: {mean_score:.2f}')
        ax.axvline(median_score, color='green', linestyle=':', linewidth=2,
                 label=f'Median: {median_score:.2f}')
        
        # Add shading for different quality zones
        ax.axvspan(0.8, 1.0, alpha=0.2, color='green', label='High Quality')
        ax.axvspan(0.5, 0.8, alpha=0.2, color='yellow', label='Medium Quality')
        ax.axvspan(0.0, 0.5, alpha=0.2, color='red', label='Low Quality')
        
        # Add text labels for quality zones
        ax.text(0.9, ax.get_ylim()[1] * 0.9, 'High', ha='center', color='darkgreen')
        ax.text(0.65, ax.get_ylim()[1] * 0.9, 'Medium', ha='center', color='darkorange')
        ax.text(0.25, ax.get_ylim()[1] * 0.9, 'Low', ha='center', color='darkred')
        
        # Enhance chart appearance
        ax.set_title(f'Quality Score Distribution: {repo_key.replace("_", "/")}', 
                   fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Quality Score', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency', fontsize=12, weight='bold')
        
        # Enhance legend
        ax.legend(frameon=True, framealpha=0.9, fontsize=10)
        
        # Add additional statistics
        stats_text = (
            f"Total PRs: {len(filter_metadata)}\n"
            f"Mean Score: {mean_score:.2f}\n"
            f"Median Score: {median_score:.2f}\n"
            f"Min Score: {min(quality_scores):.2f}\n"
            f"Max Score: {max(quality_scores):.2f}"
        )
        
        # Add stats box
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               va='top',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # Save to figures directory
        plt.savefig(self.figures_dir / f"{repo_key}_enhanced_quality_distribution.png", dpi=300)
        
        # Add to PDF
        plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_enhanced_filter_analysis(self, pdf, repo_key, filter_metadata):
        """Add enhanced filter analysis visualizations."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[1, 1])
        
        # Extract bot filter data
        bot_reasons = []
        for meta in filter_metadata:
            if not meta.get("bot_filter", {}).get("passed", False):
                reasons = meta.get("bot_filter", {}).get("details", {}).get("reasons", [])
                bot_reasons.extend(reasons)
        
        # Simplify bot reasons for better visualization
        simplified_reasons = {}
        for reason in bot_reasons:
            if "Bot username pattern match" in reason:
                key = "Username Pattern"
                simplified_reasons[key] = simplified_reasons.get(key, 0) + 1
            elif "Bot title pattern matches" in reason:
                key = "Title Pattern"
                simplified_reasons[key] = simplified_reasons.get(key, 0) + 1
            elif "Bot body pattern matches" in reason:
                key = "Body Pattern"
                simplified_reasons[key] = simplified_reasons.get(key, 0) + 1
            elif "Trivial change pattern matches" in reason:
                key = "Trivial Changes"
                simplified_reasons[key] = simplified_reasons.get(key, 0) + 1
            else:
                key = "Other"
                simplified_reasons[key] = simplified_reasons.get(key, 0) + 1
        
        # Create enhanced pie chart for bot reasons
        if simplified_reasons:
            # Use the repository color as the base for the pie chart
            repo_color = self.repository_colors.get(repo_key, 'steelblue')
            colors = [self._adjust_color_brightness(repo_color, factor) 
                    for factor in [1.0, 0.85, 0.7, 0.55, 0.4]]
            
            wedges, texts, autotexts = axes[0].pie(
                simplified_reasons.values(), 
                labels=simplified_reasons.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=colors[:len(simplified_reasons)],
                wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                textprops={'fontsize': 10},
            )
            
            # Enhance text appearance
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_weight('bold')
                autotext.set_color('white')
            
            axes[0].set_title('Bot Filter Reasons', fontsize=14, weight='bold')
        else:
            axes[0].text(0.5, 0.5, 'No bot filter rejections', 
                       ha='center', va='center', fontsize=12)
            axes[0].set_title('Bot Filter Reasons (No Data)', fontsize=14, weight='bold')
            axes[0].axis('off')
        
        # Extract size filter data
        size_stats = {
            "Too Small": 0,
            "Too Large": 0,
            "Too Many Files": 0,
            "No Code Changes": 0,
            "Only Generated Changes": 0
        }
        
        for meta in filter_metadata:
            if meta.get("bot_filter", {}).get("passed", False) and not meta.get("size_filter", {}).get("passed", False):
                details = meta.get("size_filter", {}).get("details", {})
                if details.get("too_small", False):
                    size_stats["Too Small"] += 1
                if details.get("too_large", False):
                    size_stats["Too Large"] += 1
                if details.get("too_many_files", False):
                    size_stats["Too Many Files"] += 1
                if details.get("no_code_changes", False):
                    size_stats["No Code Changes"] += 1
                if details.get("only_generated_changes", False):
                    size_stats["Only Generated Changes"] += 1
        
        # Filter out zero values
        size_stats = {k: v for k, v in size_stats.items() if v > 0}
        
        # Create enhanced bar chart for size filter stats
        if size_stats:
            # Sort by count
            sorted_data = sorted(size_stats.items(), key=lambda x: x[1], reverse=True)
            labels, counts = zip(*sorted_data)
            
            # Use the repository color with different brightness levels
            repo_color = self.repository_colors.get(repo_key, 'steelblue')
            colors = [self._adjust_color_brightness(repo_color, 1.0 - (i * 0.15)) 
                    for i in range(len(labels))]
            
            bars = axes[1].bar(labels, counts, color=colors, 
                             edgecolor='white', linewidth=0.5)
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                           str(int(height)), ha='center', va='bottom', fontsize=10)
            
            axes[1].set_title('Size Filter Rejection Reasons', fontsize=14, weight='bold')
            axes[1].set_ylabel('Number of PRs', fontsize=11)
            axes[1].set_ylim(0, max(counts) * 1.2)  # Add space for labels
            
            # Rotate labels for better readability
            axes[1].set_xticklabels(labels, rotation=30, ha='right')
            
            # Add grid
            axes[1].grid(axis='y', linestyle='--', alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No size filter rejections', 
                       ha='center', va='center', fontsize=12)
            axes[1].set_title('Size Filter Rejection Reasons (No Data)', fontsize=14, weight='bold')
            axes[1].axis('off')
        
        # Overall title
        fig.suptitle(f'Filter Analysis: {repo_key.replace("_", "/")}', 
                    fontsize=16, weight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save to figures directory
        plt.savefig(self.figures_dir / f"{repo_key}_enhanced_filter_analysis.png", dpi=300)
        
        # Add to PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_enhanced_quality_profiles(self, pdf):
        """Add enhanced quality profiles for exemplary PRs."""
        # Find exemplary PRs with high quality scores from each repository
        exemplary_prs = []
        
        for repo_key, data in self.repo_data.items():
            # Get PRs that passed all filters
            filtered_prs = data.get("filtered_prs", [])
            filter_metadata = data.get("filter_metadata", [])
            
            # Create a mapping from PR number to metadata
            pr_to_meta = {
                meta.get("pr_number"): meta 
                for meta in filter_metadata 
                if meta.get("passed_filter", False)
            }
            
            # Sort filtered PRs by quality score (highest first)
            sorted_prs = sorted(
                [(pr, pr_to_meta.get(pr.get("pr_number", 0), {}).get("quality_score", 0)) 
                 for pr in filtered_prs],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take the top PR if available
            if sorted_prs:
                top_pr, score = sorted_prs[0]
                top_pr["repository"] = repo_key.replace("_", "/")
                top_pr["quality_score"] = score
                top_pr["metadata"] = pr_to_meta.get(top_pr.get("pr_number", 0), {})
                exemplary_prs.append(top_pr)
        
        if not exemplary_prs:
            logger.warning("No exemplary PRs found for quality profiles")
            return
        
        # Create introduction page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add a light background
        gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
        ax.imshow(gradient, cmap='Blues', alpha=0.1, aspect='auto',
                 extent=[0, 1, 0, 1], transform=fig.transFigure)
        
        # Add title with styling
        fig.text(0.5, 0.95, "Quality Profiles of Exemplary PRs", 
                fontsize=24, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title with gradient
        line_gradient = np.linspace(0.1, 0.9, 100)
        for i, x in enumerate(line_gradient):
            alpha = 1 - abs(2 * (x - 0.5))
            ax.plot([x, x+0.01], [0.92, 0.92], color='#3498db', alpha=alpha, linewidth=2, transform=fig.transFigure)
        
        # Create a styled box for the explanation
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((0.1, 0.75), 0.8, 0.12, fill=True, 
                           facecolor='#e8f4f8', alpha=0.7, 
                           boxstyle="round,pad=0.02",
                           transform=fig.transFigure,
                           edgecolor='#3498db', linewidth=1)
        ax.add_patch(box)
        
        # Explanation text with improved formatting
        explanation = [
            "The following sections showcase high-quality PRs that passed all filtering stages.",
            "These PRs represent exemplary software engineering data with meaningful",
            "problem-solving content, appropriate size, and high relevance scores.",
            "",
            "Each scorecard provides detailed metrics on the PR's quality dimensions,",
            "including file composition, code changes, and identified relevant files",
            "that provide context for understanding the changes."
        ]
        
        fig.text(0.15, 0.81, "\n".join(explanation), 
                fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
        
        # Add a preview of PR scorecards
        y_pos = 0.6
        for i, pr in enumerate(exemplary_prs[:3]):  # Show up to 3 previews
            # Create a preview box
            preview_box = FancyBboxPatch((0.1, y_pos-0.15), 0.8, 0.15, fill=True, 
                                      facecolor='#f8f9fa', alpha=0.7, 
                                      boxstyle="round,pad=0.02",
                                      transform=fig.transFigure,
                                      edgecolor=self.repository_colors.get(
                                        pr.get("repo_key", pr.get("repository").replace("/", "_")), 
                                        '#bdc3c7'), 
                                      linewidth=2)
            ax.add_patch(preview_box)
            
            # Add PR title and info
            repo_name = pr.get("repository")
            pr_number = pr.get("pr_number")
            title = pr.get("title", "")
            quality_score = pr.get("quality_score", 0)
            
            pr_title = f"PR #{pr_number}: {title}"
            if len(pr_title) > 80:
                pr_title = pr_title[:77] + "..."
                
            fig.text(0.15, y_pos-0.03, pr_title, 
                    fontsize=11, va='center', weight='bold', color='#2c3e50')
            
            fig.text(0.15, y_pos-0.07, f"Repository: {repo_name}", 
                    fontsize=10, va='center', color='#34495e')
            
            fig.text(0.15, y_pos-0.11, f"Quality Score: {quality_score:.2f}", 
                    fontsize=10, va='center', color='#34495e')
            
            # Add a color bar indicating quality
            quality_rect = plt.Rectangle((0.7, y_pos-0.09), 0.15, 0.03, 
                                      color=self._get_quality_color(quality_score))
            ax.add_patch(quality_rect)
            
            # Add icons for quality metrics
            # Use text symbols as a simple alternative to icons
            icons = "★" * int(quality_score * 5)
            fig.text(0.7, y_pos-0.03, icons, 
                    fontsize=10, va='center', ha='left', color='goldenrod')
            
            y_pos -= 0.17  # Move down for next preview
        
        # Add note about viewing detailed profiles
        fig.text(0.5, 0.2, "Detailed PR quality scorecards are presented on the following pages", 
                fontsize=12, ha='center', style='italic', color='#7f8c8d')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add individual PR scorecards
        for pr in exemplary_prs:
            self._add_enhanced_pr_scorecard(pdf, pr)
    
    def _add_enhanced_pr_scorecard(self, pdf, pr):
        """Add an enhanced quality scorecard for a PR."""
        pr_number = pr.get("pr_number")
        repo_name = pr.get("repository")
        repo_key = repo_name.replace("/", "_")
        metadata = pr.get("metadata", {})
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1.2]})
        
        # Flatten axes for easier access
        axes = axes.flatten()
        
        # Add title
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
        
        # 4. Relevant files (bottom right)
        relevant_files = pr.get("relevant_files", [])
        num_relevant = len(relevant_files)
        
        if relevant_files:
            axes[3].axis('off')
            axes[3].set_title("Relevant Files", fontsize=12, weight='bold')
            
            # Create a styled list of relevant files
            file_list = f"Files that provide context ({num_relevant} total):"
            axes[3].text(0.5, 0.95, file_list, 
                       ha='center', va='top', fontsize=10, weight='bold')
            
            # Show up to 8 files, with ellipsis if there are more
            display_files = relevant_files[:8]
            if len(relevant_files) > 8:
                display_files.append("... and more")
                
            # Use a cool background for the file list
            file_bg = plt.Rectangle((0.1, 0.2), 0.8, 0.7, 
                                  facecolor='#f8f9fa', alpha=0.5, 
                                  edgecolor='#bdc3c7', linewidth=1)
            axes[3].add_patch(file_bg)
            
            for i, file in enumerate(display_files):
                y_pos = 0.85 - (i * 0.07)
                
                # Use different styling for different file types
                if file.endswith(".py"):
                    color = "#3572A5"  # Python color
                    prefix = "🐍 "
                elif file.endswith(".js"):
                    color = "#f1e05a"  # JavaScript color
                    prefix = "📜 "
                elif file.endswith(".md"):
                    color = "#083fa1"  # Markdown color
                    prefix = "📄 "
                elif file.endswith(".json") or file.endswith(".yml") or file.endswith(".yaml"):
                    color = "#cb171e"  # Config color
                    prefix = "⚙️ "
                elif "..." in file:
                    color = "#666666"  # For ellipsis
                    prefix = ""
                else:
                    color = "#333333"  # Default color
                    prefix = "📁 "
                
                axes[3].text(0.15, y_pos, prefix + file, 
                           ha='left', va='center', fontsize=9, color=color)
        else:
            axes[3].axis('off')
            axes[3].text(0.5, 0.5, "No relevant files identified", 
                       ha='center', va='center', fontsize=12)
            axes[3].set_title("Relevant Files", fontsize=12, weight='bold')
        
        # Add PR title and description at the bottom
        pr_title = pr.get("title", "")
        pr_desc = pr.get("body", "")
        
        # Truncate description if too long
        if pr_desc and len(pr_desc) > 300:
            pr_desc = pr_desc[:297] + "..."
            
        # Add a box for the PR details
        pr_box = plt.Rectangle((0.05, 0.02), 0.9, 0.08, 
                             facecolor='#e8f4f8', alpha=0.5, 
                             edgecolor='#3498db', linewidth=1)
        fig.add_artist(pr_box)
        
        # Add title and truncated description
        fig.text(0.07, 0.08, f"Title: {pr_title}", fontsize=10, weight='bold')
        if pr_desc:
            fig.text(0.07, 0.04, f"Description: {pr_desc}", fontsize=8)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make room for title and footer
        
        # Add to PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_enhanced_methodology_section(self, pdf):
        """Add an enhanced methodology section to the report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add a light background
        gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
        ax.imshow(gradient, cmap='Blues', alpha=0.1, aspect='auto',
                 extent=[0, 1, 0, 1], transform=fig.transFigure)
        
        # Add section title with styling
        fig.text(0.5, 0.95, "Methodology", 
                fontsize=24, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title with gradient
        line_gradient = np.linspace(0.1, 0.9, 100)
        for i, x in enumerate(line_gradient):
            alpha = 1 - abs(2 * (x - 0.5))
            ax.plot([x, x+0.01], [0.92, 0.92], color='#3498db', alpha=alpha, linewidth=2, transform=fig.transFigure)
        
        # Use styled boxes for each component
        component_colors = ['#e8f8f5', '#eafaf1', '#fef9e7', '#fae5d3']
        component_borders = ['#1abc9c', '#2ecc71', '#f1c40f', '#e67e22']
        component_icons = ['🔍', '⚖️', '🧩', '📊']
        
        # Introduction to methodology
        intro_text = [
            "The data curation pipeline implements a multi-stage filtering approach inspired by the",
            "SWE-RL paper, focusing on extracting high-quality software engineering data from",
            "GitHub repositories. The pipeline consists of the following key components:"
        ]
        
        # Add introduction text
        fig.text(0.1, 0.87, "\n".join(intro_text), 
                fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
        
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
        
        # Position for components
        y_pos = 0.75
        for i, (title, details) in enumerate(components):
            # Create box with enhanced styling
            from matplotlib.patches import FancyBboxPatch
            box_height = 0.13
            box = FancyBboxPatch((0.1, y_pos-box_height), 0.8, box_height, 
                               fill=True, facecolor=component_colors[i], alpha=0.7,
                               boxstyle="round,pad=0.02",
                               transform=fig.transFigure, edgecolor=component_borders[i], 
                               linewidth=2, zorder=1)
            ax.add_patch(box)
            
            # Add icon and title with enhanced styling
            fig.text(0.15, y_pos-0.03, component_icons[i], fontsize=18, ha='left', 
                    va='center', color=component_borders[i], weight='bold')
            fig.text(0.2, y_pos-0.03, title, fontsize=14, ha='left', 
                    va='center', color='#34495e', weight='bold')
            
            # Add details with better styling
            detail_text = "\n".join(details)
            fig.text(0.2, y_pos-0.07, detail_text, fontsize=10, 
                    va='top', ha='left', color='#34495e', linespacing=1.3)
            
            y_pos -= 0.17
        
        # Add process flow diagram
        ax.arrow(0.3, 0.3, 0, -0.05, head_width=0.02, head_length=0.01, 
                fc=component_borders[0], ec=component_borders[0], transform=fig.transFigure)
        ax.arrow(0.5, 0.3, 0, -0.05, head_width=0.02, head_length=0.01, 
                fc=component_borders[1], ec=component_borders[1], transform=fig.transFigure)
        ax.arrow(0.7, 0.3, 0, -0.05, head_width=0.02, head_length=0.01, 
                fc=component_borders[2], ec=component_borders[2], transform=fig.transFigure)
        
        # Final summary with enhanced styling
        from matplotlib.patches import FancyBboxPatch
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
    
    def _adjust_color_brightness(self, color, factor):
        """Adjust the brightness of a color."""
        try:
            import matplotlib.colors as mcolors
            
            # Convert to RGB
            rgb = mcolors.to_rgb(color)
            
            # Adjust brightness
            adjusted = [min(max(c * factor, 0), 1) for c in rgb]
            
            # Convert back to color
            return mcolors.rgb2hex(adjusted)
        except:
            # If conversion fails, return the original color
            return color
    
    def _get_quality_color(self, score):
        """Get a color representing a quality score."""
        if score >= 0.8:
            return '#2ecc71'  # Green for high quality
        elif score >= 0.6:
            return '#f1c40f'  # Yellow for medium-high quality
        elif score >= 0.4:
            return '#e67e22'  # Orange for medium quality
        else:
            return '#e74c3c'  # Red for low quality

def main():
    """Run the enhanced report generator."""
    parser = argparse.ArgumentParser(description="Generate enhanced filtering report")
    parser.add_argument("--data-dir", type=str, default="~/gh-data-curator/data", 
                      help="Base data directory")
    parser.add_argument("--view", action="store_true", help="Open the report after generation")
    args = parser.parse_args()
    
    # Expand user directory
    data_dir = Path(args.data_dir).expanduser()
    
    # Initialize enhanced report generator
    generator = EnhancedReportGenerator(data_dir)
    
    # Generate the report
    report_path = generator.generate_report()
    
    # Open the report if requested
    if args.view and report_path.exists():
        try:
            if os.name == 'posix':  # Unix/Linux/MacOS
                subprocess.run(['xdg-open', str(report_path)], check=False)
            elif os.name == 'nt':  # Windows
                os.startfile(str(report_path))
            else:
                logger.warning("Automatic report opening not supported on this platform")
        except Exception as e:
            logger.warning(f"Could not open report: {e}")
            logger.info(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()