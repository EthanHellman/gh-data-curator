
#!/usr/bin/env python3
"""
Generate a comprehensive report on data curation filtering results.
This script generates a PDF report combining analysis from all repositories,
highlighting key metrics, quality distributions, and insights.
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
from dataflow.filtering.filtering_pipeline import FilterPipeline
from dataflow.analysis.analyze_filter_results import FilterResultsAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generate a comprehensive report on data curation filtering results.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the report generator.
        
        Args:
            data_dir: Base directory containing filtered PR data
        """
        self.data_dir = data_dir
        self.filtered_dir = data_dir / "filtered"
        self.results_dir = data_dir / "analysis_results"
        self.output_dir = data_dir / "reports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Ensure analyzer has run
        self.analyzer = FilterResultsAnalyzer(data_dir)
        
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
            'figure.dpi': 100
        })
    
    def generate_report(self) -> Path:
        """
        Generate a comprehensive PDF report.
        
        Returns:
            Path to the generated report
        """
        logger.info("Generating comprehensive filtering report...")
        
        # Run analyzer to ensure we have all metrics
        cross_metrics = self.analyzer.analyze_all_repositories()
        
        # Create output PDF
        report_path = self.output_dir / f"data_curation_report_{self.timestamp}.pdf"
        
        with PdfPages(report_path) as pdf:
            # Cover page
            self._add_cover_page(pdf)
            
            # Executive summary
            self._add_executive_summary(pdf, cross_metrics)
            
            # Cross-repository comparison
            self._add_cross_repo_comparison(pdf)
            
            # Repository-specific sections
            for repo_dir in sorted(self.filtered_dir.iterdir()):
                if repo_dir.is_dir():
                    repo_key = repo_dir.name
                    self._add_repo_section(pdf, repo_key)
            
            # Quality profiles for top PRs
            self._add_quality_profiles(pdf)
            
            # Methodology
            self._add_methodology_section(pdf)
        
        logger.info(f"Report generated successfully: {report_path}")
        return report_path
    
    def _add_cover_page(self, pdf):
        """Add a cover page to the report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add a background color
        fig.patch.set_facecolor('#f5f5f5')
        
        # Draw a decorative border
        border_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                                   ec='steelblue', lw=3, transform=fig.transFigure)
        fig.patches.extend([border_rect])
        
        # Add a logo-like element
        logo_rect = plt.Rectangle((0.35, 0.75), 0.3, 0.1, fill=True, 
                                 fc='steelblue', ec='none', transform=fig.transFigure)
        fig.patches.extend([logo_rect])
        
        # Title with styling
        fig.text(0.5, 0.7, "Data Curation Pipeline", 
                fontsize=28, ha='center', weight='bold', color='#2c3e50')
        fig.text(0.5, 0.62, "Filtering Results Report", 
                fontsize=22, ha='center', color='#34495e')
        
        # Date and time with nice formatting
        date_str = datetime.now().strftime("%B %d, %Y")
        fig.text(0.5, 0.53, date_str, 
                fontsize=16, ha='center', color='#7f8c8d', style='italic')
        
        # Repository information
        repo_count = len(list(p for p in self.filtered_dir.iterdir() if p.is_dir()))
        fig.text(0.5, 0.45, f"Analysis of {repo_count} GitHub Repositories", 
                fontsize=18, ha='center', color='#2980b9')
        
        # Add decorative elements
        for i, color in enumerate(['#3498db', '#2ecc71', '#e74c3c', '#f39c12']):
            y_pos = 0.3 - (i * 0.03)
            fig.text(0.5, y_pos, "⬤", fontsize=14, ha='center', color=color)
        
        # Footer
        fig.text(0.5, 0.15, "Generated with SWE-RL Inspired Data Curation Pipeline", 
                fontsize=12, ha='center', style='italic', color='#7f8c8d')
        
        # Add version and timestamp in small text
        fig.text(0.5, 0.1, f"v1.0 • Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=9, ha='center', color='#95a5a6')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_executive_summary(self, pdf, cross_metrics):
        """Add an executive summary section to the report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add a light background
        fig.patch.set_facecolor('#f9f9f9')
        
        # Section header with styling
        fig.text(0.5, 0.95, "Executive Summary", 
                fontsize=24, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)
        
        # Summary text with improved formatting
        summary_text = self._generate_executive_summary(cross_metrics)
        
        # Format text in a more readable way
        # We'll wrap the text manually to ensure it fits well
        from textwrap import wrap
        wrapped_text = []
        for line in summary_text.split('\n'):
            if line.strip():
                wrapped_text.extend(wrap(line, width=80))
            else:
                wrapped_text.append('')
        
        formatted_text = '\n'.join(wrapped_text)
        
        # Add the text with better positioning and formatting
        fig.text(0.1, 0.87, formatted_text, 
                fontsize=11, va='top', ha='left', 
                linespacing=1.5, color='#333333',
                transform=fig.transFigure)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Include key visualizations with proper alignment and sizing
        self._add_image_to_pdf(pdf, "cross_repo_filter_rates.png", 
                              "Filter Rates Across Repositories", 
                               width=0.8, height=0.6, y_offset=0.25)
        
        self._add_image_to_pdf(pdf, "cross_repo_data_reduction.png", 
                              "Data Reduction Comparison", 
                               width=0.8, height=0.6, y_offset=0.25)
    
    def _add_image_to_pdf(self, pdf, image_name, title, width=0.8, height=0.6, y_offset=0.3):
        """Helper method to properly add an image to the PDF."""
        image_path = self.analyzer.figures_dir / image_name
        if not image_path.exists():
            return
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add title
        fig.text(0.5, 0.95, title, 
                fontsize=20, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)
        
        # Load and display the image properly
        try:
            img = mpimg.imread(str(image_path))
            # Calculate position to center the image
            img_aspect = img.shape[1] / img.shape[0]  # width/height
            
            # Adjust width and height to maintain aspect ratio
            if img_aspect > 1:  # Wider than tall
                adj_height = height / img_aspect
                ax.imshow(img, extent=[0.5-width/2, 0.5+width/2, 0.5-adj_height/2, 0.5+adj_height/2], 
                         transform=fig.transFigure, aspect='auto')
            else:  # Taller than wide
                adj_width = width * img_aspect
                ax.imshow(img, extent=[0.5-adj_width/2, 0.5+adj_width/2, 0.5-height/2, 0.5+height/2], 
                         transform=fig.transFigure, aspect='auto')
            
            # Add caption
            fig.text(0.5, 0.5 - height/2 - 0.05, f"Figure: {title}", 
                    fontsize=10, ha='center', style='italic', color='#555555')
            
        except Exception as e:
            logger.error(f"Error adding image {image_name}: {e}")
            fig.text(0.5, 0.5, f"Image could not be loaded: {image_name}", 
                    fontsize=14, ha='center', color='red')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_executive_summary(self, cross_metrics):
        """Generate text for the executive summary."""
        # Load repository metrics
        all_metrics = {}
        for repo_dir in self.filtered_dir.iterdir():
            if repo_dir.is_dir():
                repo_key = repo_dir.name
                metrics_path = self.results_dir / f"{repo_key}_metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        all_metrics[repo_key] = json.load(f)
        
        if not all_metrics:
            return "No metrics data available for summary."
        
        # Calculate overall statistics
        total_prs = sum(metrics["total_prs"] for metrics in all_metrics.values())
        passed_prs = sum(metrics["passed_prs"] for metrics in all_metrics.values())
        bot_filtered = sum(metrics["bot_filtered"] for metrics in all_metrics.values())
        size_filtered = sum(metrics["size_filtered"] for metrics in all_metrics.values())
        content_filtered = sum(metrics["content_filtered"] for metrics in all_metrics.values())
        
        overall_pass_rate = passed_prs / total_prs if total_prs > 0 else 0
        data_reduction = 1 - overall_pass_rate
        
        # Calculate average quality score
        quality_scores = []
        for metrics in all_metrics.values():
            if metrics["passed_prs"] > 0:
                quality_scores.append(metrics["avg_quality_score"])
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Generate summary text
        summary = [
            "This report presents the results of a data curation pipeline designed to extract and filter",
            "high-quality software engineering data from GitHub repositories.",
            "",
            f"Across {len(all_metrics)} repositories, a total of {total_prs} PRs were processed through",
            f"the filtering pipeline, resulting in {passed_prs} high-quality PRs that passed all filters.",
            f"This represents an overall pass rate of {overall_pass_rate:.1%}, with a data reduction ratio",
            f"of {data_reduction:.1%}.",
            "",
            "Filtering Breakdown:",
            f"- Bot Filter: {bot_filtered} PRs ({bot_filtered/total_prs:.1%} of total)",
            f"- Size/Complexity Filter: {size_filtered} PRs ({size_filtered/total_prs:.1%} of total)",
            f"- Content Relevance Filter: {content_filtered} PRs ({content_filtered/total_prs:.1%} of total)",
            "",
            f"The average quality score for passing PRs was {avg_quality:.2f} on a scale of 0-1.",
            "",
            "Key Findings:",
            "1. Bot-generated PRs constitute a significant portion of repository activity",
            "2. Size and complexity filters effectively remove both trivial and unwieldy changes",
            "3. Content relevance filtering ensures focus on meaningful software engineering content",
            "4. The pipeline successfully identifies related files that provide context for changes",
            "",
            "The following pages provide detailed analyses for each repository and highlight",
            "exemplary PRs that represent high-quality software engineering data."
        ]
        
        return "\n".join(summary)
    
    def _add_cross_repo_comparison(self, pdf):
        """Add cross-repository comparison section to the report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add section title
        fig.text(0.5, 0.95, "Cross-Repository Comparison", 
                fontsize=24, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)
        
        # Add visualization if available - we'll use our helper method
        quality_comp_path = self.analyzer.figures_dir / "cross_repo_quality_scores.png"
        if quality_comp_path.exists():
            fig.text(0.5, 0.87, "Quality Metrics Comparison", 
                    fontsize=18, ha='center', color='#34495e')
            
            # Load and display the image properly
            try:
                img = mpimg.imread(str(quality_comp_path))
                # Calculate position to center the image
                img_aspect = img.shape[1] / img.shape[0]  # width/height
                width = 0.8
                height = width / img_aspect
                
                ax.imshow(img, extent=[0.5-width/2, 0.5+width/2, 0.5-height/2, 0.5+height/2], 
                         transform=fig.transFigure, aspect='auto')
                
                # Add caption
                fig.text(0.5, 0.5 - height/2 - 0.05, "Figure: Quality Metrics Comparison Across Repositories", 
                        fontsize=10, ha='center', style='italic', color='#555555')
            except Exception as e:
                logger.error(f"Error adding quality comparison image: {e}")
        else:
            fig.text(0.5, 0.5, "No quality comparison visualization available", 
                    fontsize=14, ha='center', color='#7f8c8d', style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add cross-repository metrics table
        metrics_table_path = self.analyzer.figures_dir / "cross_repo_metrics_table.png"
        if metrics_table_path.exists():
            self._add_image_to_pdf(pdf, "cross_repo_metrics_table.png", 
                                  "Cross-Repository Metrics Comparison", 
                                  width=0.8, height=0.6, y_offset=0.25)
    
    def _add_repo_section(self, pdf, repo_key):
        """Add repository-specific section to the report."""
        owner, repo = repo_key.split('_', 1)
        
        # Load repository metrics
        metrics_path = self.results_dir / f"{repo_key}_metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Metrics not found for {repo_key}, skipping section")
            return
            
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add repository styling
        fig.patch.set_facecolor('#f9f9f9')
        
        # Repository title with styling
        fig.text(0.5, 0.95, f"Repository: {owner}/{repo}", 
                fontsize=22, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)
        
        # Create a styled box for the summary metrics
        box = plt.Rectangle((0.1, 0.65), 0.8, 0.22, fill=True, 
                           color='#ecf0f1', alpha=0.7, transform=fig.transFigure,
                           edgecolor='#bdc3c7', linewidth=1)
        ax.add_patch(box)
        
        # Summary metrics in styled box
        summary_text = [
            f"Total PRs: {metrics['total_prs']}",
            f"Passed PRs: {metrics['passed_prs']} ({metrics['overall_pass_rate']:.1%})",
            f"Data Reduction: {metrics['data_reduction_ratio']:.1%}",
            f"Average Quality Score: {metrics['avg_quality_score']:.2f}",
            "",
            "Filtering Breakdown:",
            f"- Bot Filter: {metrics['bot_filtered']} PRs ({metrics['bot_filter_rate']:.1%})",
            f"- Size Filter: {metrics['size_filtered']} PRs ({metrics['size_filter_rate']:.1%})",
            f"- Content Filter: {metrics['content_filtered']} PRs ({metrics['content_filter_rate']:.1%})"
        ]
        
        fig.text(0.15, 0.85, "\n".join(summary_text), 
                fontsize=12, va='top', color='#2c3e50', linespacing=1.3)
        
        # Add repository visualization
        funnel_path = self.analyzer.figures_dir / f"{repo_key}_filter_funnel.png"
        if funnel_path.exists():
            try:
                img = mpimg.imread(str(funnel_path))
                # Calculate position to center the image
                img_aspect = img.shape[1] / img.shape[0]  # width/height
                width = 0.7
                height = width / img_aspect
                
                # Position below the summary box
                ax.imshow(img, extent=[0.5-width/2, 0.5+width/2, 0.35-height/2, 0.35+height/2], 
                         transform=fig.transFigure, aspect='auto')
                
                # Add caption
                fig.text(0.5, 0.35 - height/2 - 0.05, "Figure: Filter Funnel Analysis", 
                        fontsize=10, ha='center', style='italic', color='#555555')
            except Exception as e:
                logger.error(f"Error adding filter funnel image: {e}")
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add quality distribution in a separate page
        quality_path = self.analyzer.figures_dir / f"{repo_key}_quality_distribution.png"
        if quality_path.exists():
            self._add_image_to_pdf(pdf, f"{repo_key}_quality_distribution.png", 
                                  f"Quality Distribution: {owner}/{repo}", 
                                  width=0.8, height=0.6, y_offset=0.25)
        
        # Add filter reason charts in a separate page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Section title
        fig.text(0.5, 0.95, f"Filter Analysis: {owner}/{repo}", 
                fontsize=22, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)
        
        # Add bot reasons chart
        bot_path = self.analyzer.figures_dir / f"{repo_key}_bot_reasons.png"
        if bot_path.exists():
            try:
                img = mpimg.imread(str(bot_path))
                # Calculate position to center the image
                img_aspect = img.shape[1] / img.shape[0]  # width/height
                width = 0.7
                height = width / img_aspect
                
                # Position in the upper half
                ax.imshow(img, extent=[0.5-width/2, 0.5+width/2, 0.65-height/2, 0.65+height/2], 
                         transform=fig.transFigure, aspect='auto')
                
                # Add caption
                fig.text(0.5, 0.65 - height/2 - 0.05, "Figure: Bot Filter Reasons", 
                        fontsize=10, ha='center', style='italic', color='#555555')
            except Exception as e:
                logger.error(f"Error adding bot reasons image: {e}")
        
        # Add size stats chart
        size_path = self.analyzer.figures_dir / f"{repo_key}_size_stats.png"
        if size_path.exists():
            try:
                img = mpimg.imread(str(size_path))
                # Calculate position to center the image
                img_aspect = img.shape[1] / img.shape[0]  # width/height
                width = 0.7
                height = width / img_aspect
                
                # Position in the lower half
                ax.imshow(img, extent=[0.5-width/2, 0.5+width/2, 0.3-height/2, 0.3+height/2], 
                         transform=fig.transFigure, aspect='auto')
                
                # Add caption
                fig.text(0.5, 0.3 - height/2 - 0.05, "Figure: Size Filter Stats", 
                        fontsize=10, ha='center', style='italic', color='#555555')
            except Exception as e:
                logger.error(f"Error adding size stats image: {e}")
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_quality_profiles(self, pdf):
        """Add quality profiles for exemplary PRs."""
        # Find PR scorecard images
        scorecard_paths = list(self.analyzer.figures_dir.glob("*_pr*_scorecard.png"))
        
        if not scorecard_paths:
            logger.warning("No PR scorecards found, skipping quality profiles section")
            return
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add page styling
        fig.patch.set_facecolor('#f9f9f9')
        
        # Add title with styling
        fig.text(0.5, 0.95, "Quality Profiles of Exemplary PRs", 
                fontsize=24, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)
        
        # Create a styled box for the explanation
        box = plt.Rectangle((0.1, 0.6), 0.8, 0.22, fill=True, 
                           color='#e8f4f8', alpha=0.7, transform=fig.transFigure,
                           edgecolor='#3498db', linewidth=1)
        ax.add_patch(box)
        
        # Explanation text with improved formatting
        explanation = [
            "The following pages showcase high-quality PRs that passed all filtering stages.",
            "These PRs represent exemplary software engineering data with meaningful",
            "problem-solving content, appropriate size, and high relevance scores.",
            "",
            "Each scorecard provides detailed metrics on the PR's quality dimensions,",
            "including file composition, code changes, and identified relevant files",
            "that provide context for understanding the changes."
        ]
        
        fig.text(0.15, 0.8, "\n".join(explanation), 
                fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
        
        # Add a preview of PR scorecards
        y_pos = 0.5
        for i, path in enumerate(scorecard_paths[:2]):
            try:
                # Extract PR info from filename
                filename = path.name
                repo_pr = filename.split('_scorecard')[0].replace('_', '/')
                
                # Create a mini-preview
                thumb_height = 0.15
                thumb_width = 0.3
                thumb_img = mpimg.imread(str(path))
                thumb_ax = fig.add_axes([0.35, y_pos - thumb_height/2, thumb_width, thumb_height])
                thumb_ax.imshow(thumb_img)
                thumb_ax.axis('off')
                
                # Add label
                fig.text(0.2, y_pos, f"PR: {repo_pr}", 
                        fontsize=10, ha='left', va='center', color='#34495e')
                
                y_pos -= 0.2
            except Exception as e:
                logger.error(f"Error adding PR thumbnail: {e}")
        
        # Add note about viewing detailed profiles
        fig.text(0.5, 0.2, "Detailed PR quality scorecards are presented on the following pages", 
                fontsize=12, ha='center', style='italic', color='#7f8c8d')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add PR scorecards (up to 3 for brevity)
        for path in scorecard_paths[:3]:
            try:
                # Extract PR info from filename
                filename = path.name
                repo_pr = filename.split('_scorecard')[0]
                
                # Create a new page for this scorecard
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                # Section title
                fig.text(0.5, 0.95, f"PR Quality Scorecard: {repo_pr.replace('_', '/')}", 
                        fontsize=20, ha='center', weight='bold', color='#2c3e50')
                
                # Add a horizontal line under the title
                ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)
                
                # Load the scorecard image
                img = mpimg.imread(str(path))
                
                # Calculate position to center the image
                img_aspect = img.shape[1] / img.shape[0]  # width/height
                width = 0.85
                height = width / img_aspect
                
                # Position in center of page
                ax.imshow(img, extent=[0.5-width/2, 0.5+width/2, 0.5-height/2, 0.5+height/2], 
                         transform=fig.transFigure, aspect='auto')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error adding PR scorecard: {e}")
    
    def _add_methodology_section(self, pdf):
        """Add methodology section to the report."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Add page styling
        fig.patch.set_facecolor('#f9f9f9')
        
        # Section title with styling
        fig.text(0.5, 0.95, "Methodology", 
                fontsize=24, ha='center', weight='bold', color='#2c3e50')
        
        # Add a horizontal line under the title
        ax.axhline(y=0.92, xmin=0.1, xmax=0.9, color='#3498db', linewidth=2)
        
        # Create styled boxes for each methodology component
        component_colors = ['#e8f8f5', '#eafaf1', '#fef9e7', '#fae5d3']
        component_borders = ['#1abc9c', '#2ecc71', '#f1c40f', '#e67e22']
        component_icons = ['🔍', '🧹', '🔗', '📊']
        
        # Methodology text
        methodology = [
            "The data curation pipeline implements a multi-stage filtering approach inspired by the",
            "SWE-RL paper, focusing on extracting high-quality software engineering data from",
            "GitHub repositories. The pipeline consists of the following key components:"
        ]
        
        # Add introduction text
        fig.text(0.1, 0.87, "\n".join(methodology), 
                fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
        
        # Components
        components = [
            ("1. Data Acquisition", [
                "- GitHub API integration for PR events and metadata",
                "- Repository cloning for file content access",
                "- Linked issue resolution and context gathering"
            ]),
            ("2. Multi-Stage Filtering", [
                "- Bot and Automation Detection: Identifies and filters out automated PRs",
                "- Size and Complexity Filtering: Ensures PRs are neither trivial nor unwieldy",
                "- Content Relevance Filtering: Focuses on meaningful software engineering content"
            ]),
            ("3. Relevant Files Prediction", [
                "- Identifies semantically related files not modified in the PR",
                "- Uses import analysis and directory structure heuristics",
                "- Enhances context for understanding code changes"
            ]),
            ("4. Quality Metrics Generation", [
                "- Comprehensive quality scoring across multiple dimensions",
                "- Metadata extraction for filtering decisions",
                "- Relevance scoring based on problem-solving indicators"
            ])
        ]
        
        # Position for components
        y_pos = 0.75
        for i, (title, details) in enumerate(components):
            # Create box
            box_height = 0.13
            box = plt.Rectangle((0.1, y_pos-box_height), 0.8, box_height, 
                               fill=True, color=component_colors[i], alpha=0.7,
                               transform=fig.transFigure, edgecolor=component_borders[i], 
                               linewidth=2, zorder=1)
            ax.add_patch(box)
            
            # Add icon and title
            fig.text(0.15, y_pos-0.03, component_icons[i], fontsize=18, ha='left', 
                    va='center', color=component_borders[i], weight='bold')
            fig.text(0.2, y_pos-0.03, title, fontsize=14, ha='left', 
                    va='center', color='#34495e', weight='bold')
            
            # Add details
            detail_text = "\n".join(details)
            fig.text(0.2, y_pos-0.07, detail_text, fontsize=10, 
                    va='top', ha='left', color='#34495e', linespacing=1.3)
            
            y_pos -= 0.17
        
        # Final summary
        conclusion = [
            "The filtering pipeline maintains high precision by using progressive refinement,",
            "ensuring that only PRs with genuine software engineering value are retained",
            "while capturing detailed metadata about filtering decisions."
        ]
        
        # Create conclusion box
        concl_box = plt.Rectangle((0.1, 0.1), 0.8, 0.08, fill=True, 
                               color='#eaecee', alpha=0.7, transform=fig.transFigure,
                               edgecolor='#7f8c8d', linewidth=1)
        ax.add_patch(concl_box)
        
        fig.text(0.5, 0.14, "\n".join(conclusion), fontsize=11, ha='center', 
                va='center', color='#2c3e50', style='italic', linespacing=1.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def main():
    """Run the report generator."""
    parser = argparse.ArgumentParser(description="Generate comprehensive filtering report")
    parser.add_argument("--data-dir", type=str, default="~/gh-data-curator/data", 
                      help="Base data directory")
    parser.add_argument("--view", action="store_true", help="Open the report after generation")
    args = parser.parse_args()
    
    # Expand user directory
    data_dir = Path(args.data_dir).expanduser()
    
    # Initialize report generator
    generator = ReportGenerator(data_dir)
    
    # Generate the report
    report_path = generator.generate_report()
    
    # Open the report if requested
    # if args.view and report_path.exists():
    #     try:
    #         if os.name == 'posix':  # Unix/Linux/MacOS
    #             # subprocess.run(['xdg-open', str(report_path)], check=False)
    #             pass
    #         elif os.name == 'nt':  # Windows
    #             os.startfile(str(report_path))
    #         else:
    #             logger.warning("Automatic report opening not supported on this platform")
    #     except Exception as e:
    #         logger.warning(f"Could not open report: {e}")
    #         logger.info(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()