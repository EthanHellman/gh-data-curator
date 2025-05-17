Enhanced Data Curation Report Generator
This package provides enhanced report generation capabilities for analyzing and visualizing the results of the data curation pipeline. It generates comprehensive PDF reports with advanced visualizations, cross-repository analysis, and clustering.
Features

Enhanced Visualizations: Improved charts with better styling, colors, and readability
Cross-Repository Analysis: Compare metrics across different repositories
Quality Clustering: Visualize PRs clustered by their quality dimensions
Correlation Analysis: Explore relationships between different quality metrics
Multi-dimensional Visualization: View PR quality dimensions across multiple axes
Advanced PR Quality Scorecards: Detailed visualization of exemplary PRs

Components

EnhancedReportGenerator: Main class that produces the comprehensive PDF report
run_enhanced_report_generator.py: Script to run the generator with command-line options

Installation
Clone this repository and ensure you have the required dependencies:
bashpip install matplotlib pandas numpy seaborn scikit-learn adjustText
Usage
Run the enhanced report generator:
bashpython run_enhanced_report_generator.py --data-dir /path/to/data
Command-line Options

--data-dir: Path to the data directory (default: ~/gh-data-curator/data)
--view: Open the report after generation
--skip-analysis: Skip running the analysis script before generating the report

Data Directory Structure
The report generator expects the following directory structure:
data_dir/
├── filtered/                # Directory containing filtered PR data
│   ├── repo1_repo1/         # Repository data
│   │   ├── filter_metadata.json    # Filtering metadata
│   │   ├── filtered_index.json     # Index of filtered PRs
│   │   ├── pr_123/                 # Data for specific PRs
│   ├── repo2_repo2/
│   │   └── ...
├── analysis_results/        # Directory for analysis outputs
│   ├── figures/             # Directory for generated figures
├── reports/                 # Directory where reports will be saved
Visualization Types

Filter Rates: Compare how PRs are filtered across repositories
Data Reduction: Visualize the data reduction ratio by repository
Quality Metrics: Compare quality scores across repositories
PR Quality Scatter Plot: Plot PRs by size vs. relevance scores
Dimension Correlation Heatmap: Correlation between quality dimensions
PR Clustering: PCA-based clustering of PRs by quality dimensions
Parallel Coordinates Plot: Multi-dimensional view of PR quality
Filter Funnel Analysis: Pipeline visualization for each repository
Quality Distribution: Distribution of quality scores by repository
PR Quality Scorecards: Detailed profiles of exemplary PRs

Extending the Visualizations
To add your own custom visualizations, extend the EnhancedReportGenerator class and add new visualization methods. Then update the generate_report method to include your new visualizations.
Requirements

Python 3.8+
matplotlib
pandas
numpy
seaborn
scikit-learn
adjustText (optional, for better text positioning in plots)

Note on Dependencies
The adjustText package is used for better text positioning in scatter plots. If not available, the code will fall back to default text positioning