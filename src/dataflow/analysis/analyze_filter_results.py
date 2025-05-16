#!/usr/bin/env python3
"""
Filter Results Analysis Module

This module analyzes the results of the filtering pipeline, generating metrics,
visualizations, and insights about data quality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
import os
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FilterResultsAnalyzer:
    """
    Analyzes results from the filtering pipeline and generates metrics,
    visualizations, and insights about data quality.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Base directory containing filtered PR data
        """
        self.data_dir = data_dir
        self.filtered_dir = data_dir / "filtered"
        self.results_dir = data_dir / "analysis_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Store metrics for all repositories
        self.repo_metrics = {}
        self.combined_metrics = defaultdict(list)
        
        # Output paths
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('ggplot')
        sns.set_theme(style="whitegrid")
    
    def analyze_repository(self, owner: str, repo: str) -> Dict:
        """
        Analyze filtering results for a single repository.
        
        Args:
            owner: Repository owner/organization
            repo: Repository name
            
        Returns:
            Dictionary of metrics and statistics
        """
        repo_key = f"{owner}_{repo}"
        logger.info(f"Analyzing filtering results for {repo_key}...")
        
        # Directory containing filtered data
        repo_dir = self.filtered_dir / repo_key
        if not repo_dir.exists():
            logger.error(f"Filtered repository data not found: {repo_dir}")
            return {}
        
        # Load filter metadata
        metadata_path = repo_dir / "filter_metadata.json"
        if not metadata_path.exists():
            logger.error(f"Filter metadata not found: {metadata_path}")
            return {}
        
        with open(metadata_path, "r") as f:
            filter_metadata = json.load(f)
        
        # Load filtered index for PRs that passed
        filtered_index_path = repo_dir / "filtered_index.json"
        filtered_prs = []
        if filtered_index_path.exists():
            with open(filtered_index_path, "r") as f:
                filtered_prs = json.load(f)
        
        # Calculate metrics
        metrics = self._calculate_metrics(filter_metadata, filtered_prs)
        
        # Generate repository-specific visualizations
        self._generate_repo_visualizations(metrics, filter_metadata, filtered_prs, repo_key)
        
        # Store metrics for combined analysis
        self.repo_metrics[repo_key] = metrics
        
        # Add to combined metrics for cross-repository analysis
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.combined_metrics[key].append((repo_key, value))
        
        # Save metrics to file
        metrics_path = self.results_dir / f"{repo_key}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Analysis completed for {repo_key}")
        return metrics
    
    def analyze_all_repositories(self) -> Dict:
        """
        Analyze filtering results for all repositories.
        
        Returns:
            Dictionary of combined metrics and cross-repository comparisons
        """
        logger.info("Analyzing all repositories...")
        
        # Get all repository directories
        repo_dirs = [d for d in self.filtered_dir.iterdir() if d.is_dir()]
        
        if not repo_dirs:
            logger.error("No filtered repository data found")
            return {}
        
        # Analyze each repository
        for repo_dir in repo_dirs:
            repo_key = repo_dir.name
            owner, repo = repo_key.split('_', 1)
            self.analyze_repository(owner, repo)
        
        # Generate cross-repository comparisons
        cross_repo_metrics = self._generate_cross_repo_analysis()
        
        # Save combined metrics
        combined_path = self.results_dir / "combined_metrics.json"
        with open(combined_path, "w") as f:
            # Convert defaultdict to regular dict for JSON serialization
            json.dump(dict(cross_repo_metrics), f, indent=2)
        
        logger.info("Cross-repository analysis completed")
        return cross_repo_metrics
    
    def _calculate_metrics(self, filter_metadata: List[Dict], filtered_prs: List[Dict]) -> Dict:
        """
        Calculate metrics for a repository.
        
        Args:
            filter_metadata: List of filter metadata for all PRs
            filtered_prs: List of PRs that passed filtering
            
        Returns:
            Dictionary of metrics
        """
        # Basic counts
        total_prs = len(filter_metadata)
        passed_prs = len(filtered_prs)
        
        # Count PRs filtered at each stage
        bot_filtered = sum(1 for meta in filter_metadata if not meta["bot_filter"]["passed"])
        size_filtered = sum(1 for meta in filter_metadata 
                          if meta["bot_filter"]["passed"] and not meta["size_filter"]["passed"])
        content_filtered = sum(1 for meta in filter_metadata 
                             if meta["bot_filter"]["passed"] and meta["size_filter"]["passed"] 
                             and not meta["content_filter"]["passed"])
        
        # Calculate filter rates
        bot_filter_rate = bot_filtered / total_prs if total_prs > 0 else 0
        size_filter_rate = size_filtered / total_prs if total_prs > 0 else 0
        content_filter_rate = content_filtered / total_prs if total_prs > 0 else 0
        overall_pass_rate = passed_prs / total_prs if total_prs > 0 else 0
        
        # Data volume reduction
        data_reduction_ratio = 1 - overall_pass_rate
        
        # Quality metrics for passing PRs
        quality_scores = [meta["quality_score"] for meta in filter_metadata if meta["passed_filter"]]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Extract bot filter reasons
        bot_filter_reasons = []
        for meta in filter_metadata:
            if not meta["bot_filter"]["passed"]:
                reasons = meta["bot_filter"]["details"].get("reasons", [])
                bot_filter_reasons.extend(reasons)
        
        bot_reason_counts = Counter(bot_filter_reasons)
        
        # Extract size filter stats
        size_stats = {
            "too_small": 0,
            "too_large": 0,
            "too_many_files": 0,
            "no_code_changes": 0,
            "only_generated_changes": 0
        }
        
        for meta in filter_metadata:
            if not meta["size_filter"]["passed"]:
                details = meta["size_filter"]["details"]
                for key in size_stats:
                    if details.get(key, False):
                        size_stats[key] += 1
        
        # Extract content filter stats
        content_relevance_scores = []
        for meta in filter_metadata:
            if meta["content_filter"]["passed"]:
                details = meta["content_filter"]["details"]
                relevance_score = details.get("relevance_score", 0)
                content_relevance_scores.append(relevance_score)
        
        # Calculate content relevance statistics
        avg_relevance_score = sum(content_relevance_scores) / len(content_relevance_scores) if content_relevance_scores else 0
        
        # Relevant files statistics
        relevant_files_counts = []
        for pr in filtered_prs:
            relevant_files = pr.get("relevant_files", [])
            relevant_files_counts.append(len(relevant_files))
        
        avg_relevant_files = sum(relevant_files_counts) / len(relevant_files_counts) if relevant_files_counts else 0
        
        # Compile metrics
        metrics = {
            "total_prs": total_prs,
            "passed_prs": passed_prs,
            "bot_filtered": bot_filtered,
            "size_filtered": size_filtered,
            "content_filtered": content_filtered,
            "bot_filter_rate": bot_filter_rate,
            "size_filter_rate": size_filter_rate,
            "content_filter_rate": content_filter_rate,
            "overall_pass_rate": overall_pass_rate,
            "data_reduction_ratio": data_reduction_ratio,
            "avg_quality_score": avg_quality_score,
            "bot_reason_counts": dict(bot_reason_counts),
            "size_filter_stats": size_stats,
            "avg_relevance_score": avg_relevance_score,
            "avg_relevant_files": avg_relevant_files,
        }
        
        return metrics
    
    def _generate_repo_visualizations(self, metrics: Dict, filter_metadata: List[Dict], 
                                     filtered_prs: List[Dict], repo_key: str) -> None:
        """
        Generate visualizations for a repository.
        
        Args:
            metrics: Repository metrics
            filter_metadata: List of filter metadata for all PRs
            filtered_prs: List of PRs that passed filtering
            repo_key: Repository key (owner_repo)
        """
        # 1. Filter funnel visualization
        self._generate_filter_funnel(metrics, repo_key)
        
        # 2. Quality score distribution
        self._generate_quality_distribution(filter_metadata, repo_key)
        
        # 3. Bot filter reasons pie chart
        self._generate_bot_reasons_chart(metrics, repo_key)
        
        # 4. Size filter stats bar chart
        self._generate_size_stats_chart(metrics, repo_key)
        
        # 5. PR quality scorecard for a sample PR
        if filtered_prs:
            # Select a high-quality PR for the scorecard
            quality_scores = [(i, filter_metadata[i]["quality_score"]) 
                             for i in range(len(filter_metadata))
                             if filter_metadata[i]["passed_filter"]]
            
            if quality_scores:
                # Choose PR with highest quality score
                quality_scores.sort(key=lambda x: x[1], reverse=True)
                best_pr_idx = quality_scores[0][0]
                best_pr_meta = filter_metadata[best_pr_idx]
                best_pr_number = best_pr_meta["pr_number"]
                
                # Find the corresponding PR in filtered_prs
                best_pr = next((pr for pr in filtered_prs if pr["pr_number"] == best_pr_number), None)
                
                if best_pr:
                    self._generate_pr_scorecard(best_pr, best_pr_meta, repo_key)
    
    def _generate_filter_funnel(self, metrics: Dict, repo_key: str) -> None:
        """Generate a filter funnel visualization."""
        plt.figure(figsize=(10, 6))
        
        # Funnel data
        stages = ['Total PRs', 'After Bot Filter', 'After Size Filter', 'After Content Filter']
        counts = [
            metrics["total_prs"],
            metrics["total_prs"] - metrics["bot_filtered"],
            metrics["total_prs"] - metrics["bot_filtered"] - metrics["size_filtered"],
            metrics["passed_prs"]
        ]
        
        # Create funnel
        plt.bar(stages, counts, color=sns.color_palette("viridis", 4))
        
        # Add count and percentage labels
        for i, count in enumerate(counts):
            percentage = count / metrics["total_prs"] * 100 if metrics["total_prs"] > 0 else 0
            plt.text(i, count + 0.5, f"{count}\n({percentage:.1f}%)", 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Filter Funnel for {repo_key}', fontsize=15)
        plt.ylabel('Number of PRs')
        plt.ylim(0, metrics["total_prs"] * 1.2)  # Add space for labels
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.figures_dir / f"{repo_key}_filter_funnel.png", dpi=300)
        plt.close()
    
    def _generate_quality_distribution(self, filter_metadata: List[Dict], repo_key: str) -> None:
        """Generate quality score distribution visualization."""
        # Extract quality scores
        quality_scores = [meta["quality_score"] for meta in filter_metadata if "quality_score" in meta]
        
        if not quality_scores:
            logger.warning(f"No quality scores found for {repo_key}")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Create histogram with KDE
        sns.histplot(quality_scores, kde=True, bins=20, color='skyblue')
        
        # Add mean and median lines
        mean_score = np.mean(quality_scores)
        median_score = np.median(quality_scores)
        
        plt.axvline(mean_score, color='red', linestyle='--', 
                   label=f'Mean: {mean_score:.2f}')
        plt.axvline(median_score, color='green', linestyle=':', 
                   label=f'Median: {median_score:.2f}')
        
        plt.title(f'Quality Score Distribution for {repo_key}', fontsize=15)
        plt.xlabel('Quality Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.figures_dir / f"{repo_key}_quality_distribution.png", dpi=300)
        plt.close()
    
    def _generate_bot_reasons_chart(self, metrics: Dict, repo_key: str) -> None:
        """Generate pie chart of bot filter reasons."""
        reasons = metrics.get("bot_reason_counts", {})
        
        if not reasons:
            logger.warning(f"No bot filter reasons found for {repo_key}")
            return
        
        # Simplify reason texts
        simplified_reasons = {}
        for reason, count in reasons.items():
            # Extract the core reason pattern
            match = re.search(r"Bot (username|title|body) pattern match", reason)
            if match:
                pattern_type = match.group(1)
                key = f"{pattern_type.capitalize()} pattern"
                simplified_reasons[key] = simplified_reasons.get(key, 0) + count
            else:
                simplified_reasons["Other"] = simplified_reasons.get("Other", 0) + count
        
        plt.figure(figsize=(10, 6))
        
        # Create pie chart
        plt.pie(simplified_reasons.values(), labels=simplified_reasons.keys(), 
               autopct='%1.1f%%', startangle=90, colors=sns.color_palette("coolwarm", len(simplified_reasons)))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.title(f'Bot Filter Reasons for {repo_key}', fontsize=15)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.figures_dir / f"{repo_key}_bot_reasons.png", dpi=300)
        plt.close()
    
    def _generate_size_stats_chart(self, metrics: Dict, repo_key: str) -> None:
        """Generate bar chart of size filter stats."""
        size_stats = metrics.get("size_filter_stats", {})
        
        if not size_stats:
            logger.warning(f"No size filter stats found for {repo_key}")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        reasons = list(size_stats.keys())
        counts = list(size_stats.values())
        
        # Humanize reason labels
        human_reasons = [reason.replace('_', ' ').title() for reason in reasons]
        
        # Sort by count
        sorted_data = sorted(zip(human_reasons, counts), key=lambda x: x[1], reverse=True)
        human_reasons, counts = zip(*sorted_data) if sorted_data else ([], [])
        
        plt.bar(human_reasons, counts, color=sns.color_palette("viridis", len(reasons)))
        
        # Add count labels
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.title(f'Size Filter Rejection Reasons for {repo_key}', fontsize=15)
        plt.ylabel('Number of PRs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.figures_dir / f"{repo_key}_size_stats.png", dpi=300)
        plt.close()
    
    def _generate_pr_scorecard(self, pr: Dict, meta: Dict, repo_key: str) -> None:
        """Generate quality scorecard for a sample PR."""
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"PR #{pr['pr_number']} Quality Scorecard - {repo_key}", fontsize=16)
        
        # 1. Filter scores (top left)
        filter_scores = {
            'Bot Filter': 1.0 - meta["bot_filter"]["details"].get("confidence", 0.0),
            'Size Filter': meta["size_filter"]["details"].get("normalized_score", 0.0),
            'Content Filter': meta["content_filter"]["details"].get("relevance_score", 0.0),
            'Overall Quality': meta["quality_score"]
        }
        
        axs[0, 0].bar(filter_scores.keys(), filter_scores.values(), 
                    color=sns.color_palette("viridis", len(filter_scores)))
        
        for i, (key, value) in enumerate(filter_scores.items()):
            axs[0, 0].text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom')
        
        axs[0, 0].set_ylim(0, 1.1)
        axs[0, 0].set_title("Filter Scores")
        axs[0, 0].set_ylabel("Score (0-1)")
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. File composition (top right)
        file_counts = {
            'Code': meta["size_filter"]["details"].get("code_file_count", 0),
            'Docs': meta["size_filter"]["details"].get("doc_file_count", 0),
            'Config': meta["size_filter"]["details"].get("config_file_count", 0),
            'Generated': meta["size_filter"]["details"].get("generated_file_count", 0),
            'Other': meta["size_filter"]["details"].get("other_file_count", 0)
        }
        
        # Remove zero values
        file_counts = {k: v for k, v in file_counts.items() if v > 0}
        
        if file_counts:
            axs[0, 1].pie(file_counts.values(), labels=file_counts.keys(), 
                        autopct='%1.1f%%', startangle=90, 
                        colors=sns.color_palette("Set2", len(file_counts)))
            axs[0, 1].set_title("File Type Composition")
        else:
            axs[0, 1].text(0.5, 0.5, "No file data available", ha='center', va='center')
            axs[0, 1].set_title("File Type Composition (No Data)")
        
        # 3. Code changes (bottom left)
        change_data = {
            'Additions': meta["size_filter"]["details"].get("additions", 0),
            'Deletions': meta["size_filter"]["details"].get("deletions", 0)
        }
        
        axs[1, 0].bar(change_data.keys(), change_data.values(), 
                    color=['green', 'red'])
        
        for i, (key, value) in enumerate(change_data.items()):
            axs[1, 0].text(i, value + 1, str(value), ha='center', va='bottom')
        
        axs[1, 0].set_title("Code Changes")
        axs[1, 0].set_ylabel("Number of Lines")
        axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Relevant files (bottom right)
        relevant_files = pr.get("relevant_files", [])
        num_relevant = len(relevant_files)
        
        if relevant_files:
            # Show up to 5 relevant files
            file_display = relevant_files[:5]
            if len(relevant_files) > 5:
                file_display.append("... and more")
            
            axs[1, 1].axis('off')
            axs[1, 1].text(0.5, 0.9, f"Relevant Files ({num_relevant} total)", 
                         ha='center', va='top', fontsize=12, fontweight='bold')
            
            for i, file in enumerate(file_display):
                axs[1, 1].text(0.5, 0.8 - (i * 0.1), file, ha='center', va='top')
        else:
            axs[1, 1].axis('off')
            axs[1, 1].text(0.5, 0.5, "No relevant files identified", 
                         ha='center', va='center', fontsize=12)
        
        axs[1, 1].set_title("Relevant Files")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure
        plt.savefig(self.figures_dir / f"{repo_key}_pr{pr['pr_number']}_scorecard.png", dpi=300)
        plt.close()
    
    def _generate_cross_repo_analysis(self) -> Dict:
        """
        Generate cross-repository analysis and visualizations.
        
        Returns:
            Dictionary of cross-repository metrics
        """
        if not self.repo_metrics:
            logger.warning("No repository metrics available for cross-repository analysis")
            return {}
        
        # Combine metrics for comparison
        cross_metrics = {}
        
        # 1. Compare filter rates across repositories
        self._generate_filter_rates_comparison()
        
        # 2. Compare quality scores across repositories
        self._generate_quality_comparison()
        
        # 3. Compare data reduction ratios
        self._generate_data_reduction_comparison()
        
        # 4. Generate summary table
        summary_table = self._generate_summary_table()
        cross_metrics["summary_table"] = summary_table
        
        return cross_metrics
    
    def _generate_filter_rates_comparison(self) -> None:
        """Generate comparison of filter rates across repositories."""
        if not self.repo_metrics:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Extract filter rates for each repository
        repos = list(self.repo_metrics.keys())
        bot_rates = [metrics["bot_filter_rate"] for metrics in self.repo_metrics.values()]
        size_rates = [metrics["size_filter_rate"] for metrics in self.repo_metrics.values()]
        content_rates = [metrics["content_filter_rate"] for metrics in self.repo_metrics.values()]
        pass_rates = [metrics["overall_pass_rate"] for metrics in self.repo_metrics.values()]
        
        # Set up bar positions
        x = np.arange(len(repos))
        width = 0.2
        
        # Create grouped bar chart
        plt.bar(x - width*1.5, bot_rates, width, label='Bot Filter Rate', color='salmon')
        plt.bar(x - width/2, size_rates, width, label='Size Filter Rate', color='skyblue')
        plt.bar(x + width/2, content_rates, width, label='Content Filter Rate', color='lightgreen')
        plt.bar(x + width*1.5, pass_rates, width, label='Overall Pass Rate', color='purple')
        
        # Add labels and title
        plt.xlabel('Repository', fontsize=12)
        plt.ylabel('Rate (percentage)', fontsize=12)
        plt.title('Filter Rates Comparison Across Repositories', fontsize=15)
        plt.xticks(x, [repo.replace('_', '/') for repo in repos], rotation=45, ha='right')
        plt.legend()
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Add grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.figures_dir / "cross_repo_filter_rates.png", dpi=300)
        plt.close()
    
    def _generate_quality_comparison(self) -> None:
        """Generate comparison of quality scores across repositories."""
        if not self.repo_metrics:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Extract quality scores for each repository
        repos = list(self.repo_metrics.keys())
        quality_scores = [metrics["avg_quality_score"] for metrics in self.repo_metrics.values()]
        relevance_scores = [metrics["avg_relevance_score"] for metrics in self.repo_metrics.values()]
        
        # Set up bar positions
        x = np.arange(len(repos))
        width = 0.35
        
        # Create grouped bar chart
        plt.bar(x - width/2, quality_scores, width, label='Avg Quality Score', color='teal')
        plt.bar(x + width/2, relevance_scores, width, label='Avg Relevance Score', color='orange')
        
        # Add labels and title
        plt.xlabel('Repository', fontsize=12)
        plt.ylabel('Score (0-1)', fontsize=12)
        plt.title('Quality Metrics Comparison Across Repositories', fontsize=15)
        plt.xticks(x, [repo.replace('_', '/') for repo in repos], rotation=45, ha='right')
        plt.legend()
        
        # Add value labels
        for i, value in enumerate(quality_scores):
            plt.text(i - width/2, value + 0.02, f"{value:.2f}", ha='center', va='bottom')
        for i, value in enumerate(relevance_scores):
            plt.text(i + width/2, value + 0.02, f"{value:.2f}", ha='center', va='bottom')
        
        # Add grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.ylim(0, 1.1)  # Set y-axis limits for consistent scale
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.figures_dir / "cross_repo_quality_scores.png", dpi=300)
        plt.close()
    
    def _generate_data_reduction_comparison(self) -> None:
        """Generate comparison of data reduction ratios across repositories."""
        if not self.repo_metrics:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Extract data reduction ratios for each repository
        repos = list(self.repo_metrics.keys())
        reduction_ratios = [metrics["data_reduction_ratio"] for metrics in self.repo_metrics.values()]
        
        # Sort by reduction ratio
        sorted_data = sorted(zip(repos, reduction_ratios), key=lambda x: x[1], reverse=True)
        repos, reduction_ratios = zip(*sorted_data)
        
        # Create bar chart
        bars = plt.bar(repos, reduction_ratios, color=sns.color_palette("viridis", len(repos)))
        
        # Add labels and title
        plt.xlabel('Repository', fontsize=12)
        plt.ylabel('Data Reduction Ratio', fontsize=12)
        plt.title('Data Reduction Ratio Comparison', fontsize=15)
        plt.xticks([repo.replace('_', '/') for repo in repos], rotation=45, ha='right')
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Add value labels
        for bar, value in zip(bars, reduction_ratios):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.02, 
                    f"{value:.1%}", ha='center', va='bottom')
        
        # Add grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.figures_dir / "cross_repo_data_reduction.png", dpi=300)
        plt.close()
    
    def _generate_summary_table(self) -> Dict:
        """
        Generate summary table for cross-repository comparison.
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.repo_metrics:
            return {}
        
        # Create summary dictionary
        summary = {
            "repositories": list(self.repo_metrics.keys()),
            "metrics": {}
        }
        
        # Key metrics to include
        key_metrics = [
            "total_prs", "passed_prs", "bot_filtered", "size_filtered", "content_filtered",
            "overall_pass_rate", "data_reduction_ratio", "avg_quality_score"
        ]
        
        # Collect values for each metric
        for metric in key_metrics:
            summary["metrics"][metric] = [
                self.repo_metrics[repo][metric] for repo in summary["repositories"]
            ]
        
        # Generate pandas DataFrame for visualization (not saved, just for analysis)
        df = pd.DataFrame({
            metric: summary["metrics"][metric] for metric in key_metrics
        }, index=summary["repositories"])
        
        # Save summary as CSV
        df.to_csv(self.results_dir / "cross_repo_summary.csv")
        
        # Create a visual table (heatmap)
        plt.figure(figsize=(12, 8))
        
        # Convert rates to percentages for better visualization
        display_df = df.copy()
        for col in ["overall_pass_rate", "data_reduction_ratio"]:
            display_df[col] = display_df[col] * 100
        
        # Rename columns for display
        display_df.columns = [
            "Total PRs", "Passed PRs", "Bot Filtered", "Size Filtered", "Content Filtered",
            "Pass Rate (%)", "Data Reduction (%)", "Avg Quality Score"
        ]
        
        # Generate heatmap
        sns.heatmap(display_df, annot=True, fmt=".1f", cmap="viridis", 
                   linewidths=0.5, cbar=False)
        
        plt.title("Cross-Repository Metrics Comparison", fontsize=15)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.figures_dir / "cross_repo_metrics_table.png", dpi=300)
        plt.close()
        
        return summary
    
    def generate_pr_quality_profiles(self, top_n: int = 3) -> None:
        """
        Generate quality profiles for top PRs across repositories.
        
        Args:
            top_n: Number of top PRs to profile per repository
        """
        logger.info(f"Generating quality profiles for top {top_n} PRs per repository...")
        
        for repo_key in self.repo_metrics:
            owner, repo = repo_key.split('_', 1)
            repo_dir = self.filtered_dir / repo_key
            
            # Load filter metadata
            metadata_path = repo_dir / "filter_metadata.json"
            filtered_index_path = repo_dir / "filtered_index.json"
            
            if not (metadata_path.exists() and filtered_index_path.exists()):
                logger.warning(f"Missing data for {repo_key}, skipping PR quality profiles")
                continue
                
            with open(metadata_path, "r") as f:
                filter_metadata = json.load(f)
                
            with open(filtered_index_path, "r") as f:
                filtered_prs = json.load(f)
            
            # Get PRs that passed filtering
            passed_prs = [(meta, idx) for idx, meta in enumerate(filter_metadata) 
                         if meta.get("passed_filter", False)]
            
            if not passed_prs:
                logger.warning(f"No passing PRs found for {repo_key}")
                continue
            
            # Sort by quality score
            passed_prs.sort(key=lambda x: x[0].get("quality_score", 0), reverse=True)
            
            # Generate profiles for top N
            top_prs = passed_prs[:top_n]
            
            for pr_meta, idx in top_prs:
                pr_number = pr_meta["pr_number"]
                
                # Find full PR data
                pr_data = next((pr for pr in filtered_prs if pr["pr_number"] == pr_number), None)
                
                if pr_data:
                    self._generate_detailed_pr_profile(pr_data, pr_meta, repo_key)
    
    def _generate_detailed_pr_profile(self, pr: Dict, meta: Dict, repo_key: str) -> None:
        """
        Generate a detailed profile for a high-quality PR.
        
        Args:
            pr: PR data
            meta: PR filter metadata
            repo_key: Repository key
        """
        pr_number = pr["pr_number"]
        logger.info(f"Generating detailed profile for PR #{pr_number} in {repo_key}")
        
        # Create a directory for this PR
        pr_dir = self.results_dir / f"{repo_key}_pr{pr_number}"
        pr_dir.mkdir(exist_ok=True)
        
        # Save PR data and metadata
        with open(pr_dir / "pr_data.json", "w") as f:
            json.dump(pr, f, indent=2)
            
        with open(pr_dir / "pr_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Generate a comprehensive scorecard (4-panel figure)
        self._generate_pr_scorecard(pr, meta, repo_key)
        
        # Generate a text report
        report = self._generate_pr_text_report(pr, meta, repo_key)
        
        with open(pr_dir / "quality_report.txt", "w") as f:
            f.write(report)
    
    def _generate_pr_text_report(self, pr: Dict, meta: Dict, repo_key: str) -> str:
        """
        Generate a text report for a PR.
        
        Args:
            pr: PR data
            meta: PR filter metadata
            repo_key: Repository key
            
        Returns:
            Text report
        """
        report = []
        report.append(f"== Quality Report for PR #{pr['pr_number']} - {repo_key} ==\n")
        
        # Basic PR info
        report.append("PR INFORMATION:")
        report.append(f"Title: {pr.get('title', 'N/A')}")
        report.append(f"Author: {pr.get('author', 'N/A')}")
        report.append(f"Created: {pr.get('created_at', 'N/A')}")
        report.append(f"Merged: {pr.get('merged_at', 'N/A')}")
        report.append("")
        
        # Quality scores
        report.append("QUALITY SCORES:")
        report.append(f"Overall Quality Score: {meta.get('quality_score', 0):.2f}")
        bot_confidence = meta["bot_filter"]["details"].get("confidence", 0.0)
        report.append(f"Bot Filter Score: {1.0 - bot_confidence:.2f}")
        report.append(f"Size Complexity Score: {meta['size_filter']['details'].get('normalized_score', 0.0):.2f}")
        report.append(f"Content Relevance Score: {meta['content_filter']['details'].get('relevance_score', 0.0):.2f}")
        report.append("")
        
        # Size metrics
        report.append("SIZE METRICS:")
        report.append(f"Files Changed: {meta['size_filter']['details'].get('total_files', 0)}")
        report.append(f"Code Files: {meta['size_filter']['details'].get('code_file_count', 0)}")
        report.append(f"Lines Added: {meta['size_filter']['details'].get('additions', 0)}")
        report.append(f"Lines Deleted: {meta['size_filter']['details'].get('deletions', 0)}")
        report.append(f"Total Line Changes: {meta['size_filter']['details'].get('total_changes', 0)}")
        report.append("")
        
        # Content quality indicators
        report.append("CONTENT QUALITY INDICATORS:")
        
        problem_solving = meta["content_filter"]["details"].get("problem_solving_indicators", [])
        if problem_solving:
            report.append("Problem-Solving Indicators:")
            for indicator in problem_solving[:5]:
                report.append(f"- {indicator}")
            if len(problem_solving) > 5:
                report.append(f"- ... and {len(problem_solving) - 5} more")
        else:
            report.append("Problem-Solving Indicators: None detected")
        
        report.append("")
        
        # Relevant files
        relevant_files = pr.get("relevant_files", [])
        report.append("RELEVANT FILES:")
        if relevant_files:
            for file in relevant_files:
                report.append(f"- {file}")
        else:
            report.append("None identified")
        report.append("")
        
        # Final assessment
        report.append("OVERALL ASSESSMENT:")
        
        quality_score = meta.get("quality_score", 0)
        if quality_score >= 0.8:
            assessment = "Excellent quality PR with significant problem-solving content."
        elif quality_score >= 0.6:
            assessment = "Good quality PR with valuable contributions."
        elif quality_score >= 0.4:
            assessment = "Moderate quality PR with some valuable aspects."
        else:
            assessment = "Lower quality PR that still passes minimum filtering criteria."
            
        report.append(assessment)
        
        return "\n".join(report)

def main():
    """Run the analysis module."""
    parser = argparse.ArgumentParser(description="Analyze filtering results")
    parser.add_argument("--data-dir", type=str, default="~/gh-data-curator/data", 
                      help="Base data directory")
    parser.add_argument("--repo", type=str, help="Specific repository to analyze (format: owner_repo)")
    parser.add_argument("--all", action="store_true", help="Analyze all repositories")
    parser.add_argument("--profiles", action="store_true", help="Generate detailed PR quality profiles")
    parser.add_argument("--top-n", type=int, default=3, 
                      help="Number of top PRs to profile per repository")
    args = parser.parse_args()
    
    # Expand user directory
    data_dir = Path(args.data_dir).expanduser()
    
    # Initialize analyzer
    analyzer = FilterResultsAnalyzer(data_dir)
    
    if args.repo:
        # Analyze a specific repository
        owner, repo = args.repo.split('_', 1)
        metrics = analyzer.analyze_repository(owner, repo)
        
        if metrics:
            logger.info(f"Analysis complete for {args.repo}")
            logger.info(f"Key metrics: Pass rate: {metrics['overall_pass_rate']:.1%}, "
                       f"Quality score: {metrics['avg_quality_score']:.2f}")
        else:
            logger.error(f"Failed to analyze repository {args.repo}")
    elif args.all:
        # Analyze all repositories
        cross_metrics = analyzer.analyze_all_repositories()
        logger.info("Cross-repository analysis complete")
    else:
        logger.error("Please specify either --repo or --all")
    
    # Generate detailed profiles if requested
    if args.profiles:
        analyzer.generate_pr_quality_profiles(args.top_n)
        logger.info(f"Generated quality profiles for top {args.top_n} PRs per repository")
    
    logger.info(f"Results saved to {analyzer.results_dir}")

if __name__ == "__main__":
    main()