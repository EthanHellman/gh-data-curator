#!/usr/bin/env python3
"""
Generate a comprehensive report on data curation filtering results.

This script generates a PDF report combining analysis from all repositories,
highlighting key metrics, quality distributions, and insights.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from dataflow.filtering.filtering_pipeline import FilterPipeline
from analyze_filter_results import FilterResultsAnalyzer

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
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.7, "Data Curation Pipeline", fontsize=24, ha='center', weight='bold')
        plt.text(0.5, 0.6, "Filtering Results Report", fontsize=20, ha='center')
        
        # Date and time
        date_str = datetime.now().strftime("%B %d, %Y")
        plt.text(0.5, 0.5, date_str, fontsize=14, ha='center')
        
        # Repository information
        repo_count = len(list(self.filtered_dir.iterdir()))
        plt.text(0.5, 0.4, f"Analysis of {repo_count} GitHub Repositories", fontsize=16, ha='center')
        
        # Footer
        plt.text(0.5, 0.1, "Generated with SWE-RL Inspired Data Curation Pipeline", 
                fontsize=10, ha='center', style='italic')
        
        pdf.savefig()
        plt.close()
    
    def _add_executive_summary(self, pdf, cross_metrics):
        """Add an executive summary section to the report."""
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, "Executive Summary", fontsize=20, ha='center', weight='bold')
        
        # Summary text
        summary_text = self._generate_executive_summary(cross_metrics)
        plt.text(0.1, 0.85, summary_text, fontsize=12, va='top', ha='left', wrap=True)
        
        # Add summary metrics visualization if available
        metrics_image_path = self.analyzer.figures_dir / "cross_repo_metrics_table.png"
        if metrics_image_path.exists():
            img = plt.imread(str(metrics_image_path))
            plt.figimage(img, 100, 100, zorder=3)
        
        pdf.savefig()
        plt.close()
        
        # Include key visualizations
        filter_rates_path = self.analyzer.figures_dir / "cross_repo_filter_rates.png"
        if filter_rates_path.exists():
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.title("Filter Rates Across Repositories", fontsize=16)
            img = plt.imread(str(filter_rates_path))
            plt.figimage(img, 50, 200, zorder=3)
            pdf.savefig()
            plt.close()
        
        data_reduction_path = self.analyzer.figures_dir / "cross_repo_data_reduction.png" 
        if data_reduction_path.exists():
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.title("Data Reduction Comparison", fontsize=16)
            img = plt.imread(str(data_reduction_path))
            plt.figimage(img, 50, 200, zorder=3)
            pdf.savefig()
            plt.close()
    
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
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, "Cross-Repository Comparison", fontsize=20, ha='center', weight='bold')
        
        # Add visualization if available
        quality_comp_path = self.analyzer.figures_dir / "cross_repo_quality_scores.png"
        if quality_comp_path.exists():
            plt.text(0.5, 0.9, "Quality Metrics Comparison", fontsize=16, ha='center')
            img = plt.imread(str(quality_comp_path))
            plt.figimage(img, 50, 400, zorder=3)
        
        pdf.savefig()
        plt.close()
    
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
        
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, f"Repository: {owner}/{repo}", fontsize=18, ha='center', weight='bold')
        
        # Summary metrics
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
        
        plt.text(0.1, 0.85, "\n".join(summary_text), fontsize=12, va='top')
        
        # Add repository visualizations if available
        funnel_path = self.analyzer.figures_dir / f"{repo_key}_filter_funnel.png"
        if funnel_path.exists():
            img = plt.imread(str(funnel_path))
            plt.figimage(img, 300, 100, zorder=3)
        
        pdf.savefig()
        plt.close()
        
        # Add quality distribution if available
        quality_path = self.analyzer.figures_dir / f"{repo_key}_quality_distribution.png"
        if quality_path.exists():
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.text(0.5, 0.95, f"Quality Distribution: {owner}/{repo}", 
                   fontsize=16, ha='center', weight='bold')
            img = plt.imread(str(quality_path))
            plt.figimage(img, 50, 200, zorder=3)
            pdf.savefig()
            plt.close()
        
        # Add filter reason charts if available
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.95, f"Filter Analysis: {owner}/{repo}", 
               fontsize=16, ha='center', weight='bold')
        
        y_pos = 200
        bot_path = self.analyzer.figures_dir / f"{repo_key}_bot_reasons.png"
        if bot_path.exists():
            img = plt.imread(str(bot_path))
            plt.figimage(img, 50, y_pos, zorder=3)
            y_pos += 300
        
        size_path = self.analyzer.figures_dir / f"{repo_key}_size_stats.png"
        if size_path.exists():
            img = plt.imread(str(size_path))
            plt.figimage(img, 50, y_pos, zorder=3)
        
        pdf.savefig()
        plt.close()
    
    def _add_quality_profiles(self, pdf):
        """Add quality profiles for exemplary PRs."""
        # Find PR scorecard images
        scorecard_paths = list(self.analyzer.figures_dir.glob("*_pr*_scorecard.png"))
        
        if not scorecard_paths:
            logger.warning("No PR scorecards found, skipping quality profiles section")
            return
        
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, "Quality Profiles of Exemplary PRs", 
               fontsize=18, ha='center', weight='bold')
        
        # Explanation text
        explanation = [
            "The following pages showcase high-quality PRs that passed all filtering stages.",
            "These PRs represent exemplary software engineering data with meaningful",
            "problem-solving content, appropriate size, and high relevance scores.",
            "",
            "Each scorecard provides detailed metrics on the PR's quality dimensions,",
            "including file composition, code changes, and identified relevant files",
            "that provide context for understanding the changes."
        ]
        
        plt.text(0.1, 0.85, "\n".join(explanation), fontsize=12, va='top')
        
        pdf.savefig()
        plt.close()
        
        # Add PR scorecards (up to 3 for brevity)
        for path in scorecard_paths[:3]:
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            
            # Extract PR info from filename
            filename = path.name
            repo_pr = filename.split('_scorecard')[0]
            
            plt.text(0.5, 0.95, f"PR Quality Scorecard: {repo_pr}", 
                   fontsize=16, ha='center', weight='bold')
            
            img = plt.imread(str(path))
            plt.figimage(img, 50, 150, zorder=3)
            
            pdf.savefig()
            plt.close()
    
    def _add_methodology_section(self, pdf):
        """Add methodology section to the report."""
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, "Methodology", fontsize=18, ha='center', weight='bold')
        
        # Methodology text
        methodology = [
            "The data curation pipeline implements a multi-stage filtering approach inspired by the",
            "SWE-RL paper, focusing on extracting high-quality software engineering data from",
            "GitHub repositories. The pipeline consists of the following key components:",
            "",
            "1. Data Acquisition",
            "   - GitHub API integration for PR events and metadata",
            "   - Repository cloning for file content access",
            "   - Linked issue resolution and context gathering",
            "",
            "2. Multi-Stage Filtering",
            "   - Bot and Automation Detection: Identifies and filters out automated PRs",
            "   - Size and Complexity Filtering: Ensures PRs are neither trivial nor unwieldy",
            "   - Content Relevance Filtering: Focuses on meaningful software engineering content",
            "",
            "3. Relevant Files Prediction",
            "   - Identifies semantically related files not modified in the PR",
            "   - Uses import analysis and directory structure heuristics",
            "   - Enhances context for understanding code changes",
            "",
            "4. Quality Metrics Generation",
            "   - Comprehensive quality scoring across multiple dimensions",
            "   - Metadata extraction for filtering decisions",
            "   - Relevance scoring based on problem-solving indicators",
            "",
            "The filtering pipeline maintains high precision by using progressive refinement,",
            "ensuring that only PRs with genuine software engineering value are retained",
            "while capturing detailed metadata about filtering decisions."
        ]
        
        plt.text(0.1, 0.85, "\n".join(methodology), fontsize=11, va='top')
        
        pdf.savefig()
        plt.close()

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
    if args.view and report_path.exists():
        if os.name == 'posix':  # Unix/Linux/MacOS
            subprocess.run(['xdg-open', str(report_path)], check=False)
        elif os.name == 'nt':  # Windows
            os.startfile(str(report_path))
        else:
            logger.warning("Automatic report opening not supported on this platform")

if __name__ == "__main__":
    main()