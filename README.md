# gh-data-curator

GitHub Data Curation Pipeline for High-Quality PR Extraction
Project Summary
This project implements an advanced data curation pipeline for extracting high-quality software engineering data from GitHub repositories. The pipeline is inspired by the SWE-RL paper and features:

Multi-stage processing: Clone → Collect → Select → Process → Filter → Analyze → Report
Quality-based selection: Uses heuristic scoring to identify high-quality PRs before deep processing
Comprehensive filtering: Removes bot PRs, trivial changes, and low-relevance contributions
Relevant files prediction: Identifies semantically connected files for better context understanding
Advanced analytics: Generates metrics, visualizations, and reports on data quality

Quick Start for Interview Demo (Django Repository)
Run the following commands to dry-run the pipeline on the Django repository:
bash# Clone the repository (dry run)
python run_pipeline.py --org django --repo django --stages clone --dry-run

# Collect PRs (dry run)
python run_pipeline.py --org django --repo django --prs 30 --initial-sample 120 --stages collect --dry-run

# Select high-quality PRs (dry run)
python run_pipeline.py --org django --repo django --prs 30 --initial-sample 120 --stages select --dry-run

# Process selected PRs (dry run)
python run_pipeline.py --org django --repo django --stages process --dry-run

# Filter for high-quality content (dry run)
python run_pipeline.py --org django --repo django --stages filter --dry-run

# Analyze results (dry run)
python run_pipeline.py --org django --repo django --stages analyze --dry-run

# Generate report (dry run)
python run_pipeline.py --org django --repo django --stages report --dry-run

# Run the entire pipeline (dry run)
python run_pipeline.py --org django --repo django --prs 30 --initial-sample 120 --dry-run
To execute the full pipeline for real (remove the --dry-run flag):
bashpython run_pipeline.py --org django --repo django --prs 30 --initial-sample 120
Key Innovation: Quality-Based Selection
The pipeline introduces a quality-based PR selection stage that:

Scores PRs using lightweight heuristics: Author patterns, title content, code changes, etc.
Ranks and selects top PRs: Prioritizes high-potential candidates for deeper processing
Significantly improves filter pass rates: Pre-filtering ensures more PRs pass through the pipeline

This approach saves processing time and ensures higher-quality final data by:

Avoiding overrepresentation of bot PRs
Prioritizing meaningful software engineering content
Focusing on PRs with substantial problem-solving aspects

Repositories for Analysis
The pipeline is configured to work with the following repositories:

django/django (Python web framework)
expressjs/express (JavaScript web framework)
scikit-learn/scikit-learn (Python ML library)
rust-lang/rust (Rust language)
pandas-dev/pandas (Python data analysis)

Advanced Configuration Options
The pipeline supports various configuration options:
bash# Process multiple repositories
python run_pipeline.py --org django --repo django --prs 30
python run_pipeline.py --org expressjs --repo express --prs 30
python run_pipeline.py --org scikit-learn --repo scikit-learn --prs 30
python run_pipeline.py --org rust-lang --repo rust --prs 30
python run_pipeline.py --org pandas-dev --repo pandas --prs 30

# Control quality selection 
python run_pipeline.py --org django --repo django --prs 30 --initial-sample 200

# Only run quality selection to explore the data
python run_pipeline.py --org django --repo django --prs 30 --initial-sample 150 --selection-only

# Force reprocessing when data already exists
python run_pipeline.py --org django --repo django --stages filter --force
Output Structure
The pipeline generates structured outputs in the data directory:

/data/raw/: Raw PR data from GitHub API
/data/selected/: Selected high-quality PRs with quality metadata
/data/processed/: Structured PR data ready for filtering
/data/filtered/: PRs that passed all quality filters
/data/analysis_results/: Metrics, visualizations, and analysis outputs
/data/reports/: Comprehensive PDF reports on data quality

Future Improvements

Parallel processing for multiple repositories
More sophisticated quality metrics using machine learning
Automatic curation based on PR quality distributions
Integration with large language models for deeper content understanding