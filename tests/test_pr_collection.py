import argparse
import json  # Add this import at the top of the file
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataflow.acquisition.pr_collector import PRCollector
from src.dataflow.processing.pr_processor import PRProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test PR collection and processing")
    parser.add_argument("--owner", type=str, required=True, help="Repository owner/organization")
    parser.add_argument("--repo", type=str, required=True, help="Repository name")
    parser.add_argument("--limit", type=int, default=10, help="Number of PRs to collect")
    args = parser.parse_args()
    
    # Verify GitHub token
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.error("GitHub token not found. Please set GITHUB_TOKEN environment variable.")
        return
    
    # Create data directories
    data_dir = Path("data")
    
    # Step 1: Collect PRs
    logger.info(f"Collecting PRs for {args.owner}/{args.repo}...")
    pr_collector = PRCollector(github_token, data_dir)
    pr_dir = pr_collector.collect_and_save(args.owner, args.repo, limit=args.limit)
    
    if not pr_dir:
        logger.error("Failed to collect PRs.")
        return
    
    logger.info(f"PR collection completed. Data saved to {pr_dir}")
    
    # Step 2: Process PRs
    logger.info(f"Processing PRs for {args.owner}/{args.repo}...")
    pr_processor = PRProcessor(data_dir)
    processed_dir = pr_processor.process_repository(args.owner, args.repo)
    
    if not processed_dir:
        logger.error("Failed to process PRs.")
        return
    
    logger.info(f"PR processing completed. Data saved to {processed_dir}")
    
    # Step 3: Display summary statistics
    logger.info("Generating summary statistics...")
    
    # Load processed index
    with open(processed_dir / "processed_index.json", "r") as f:
        processed_prs = json.load(f)
    
    # Calculate statistics
    total_prs = len(processed_prs)
    bot_prs = sum(1 for pr in processed_prs if pr["is_bot_pr"])
    bug_fixes = sum(1 for pr in processed_prs if pr["is_bug_fix"])
    with_linked_issues = sum(1 for pr in processed_prs if pr["has_linked_issue"])
    
    avg_files = sum(pr["file_count"] for pr in processed_prs) / total_prs if total_prs > 0 else 0
    avg_lines = sum(pr["line_changes"] for pr in processed_prs) / total_prs if total_prs > 0 else 0
    
    # Display statistics
    logger.info(f"Summary Statistics for {args.owner}/{args.repo}:")
    logger.info(f"Total PRs processed: {total_prs}")
    logger.info(f"Bot PRs: {bot_prs} ({bot_prs / total_prs * 100:.1f}%)")
    logger.info(f"Bug fixes: {bug_fixes} ({bug_fixes / total_prs * 100:.1f}%)")
    logger.info(f"PRs with linked issues: {with_linked_issues} ({with_linked_issues / total_prs * 100:.1f}%)")
    logger.info(f"Average files changed per PR: {avg_files:.1f}")
    logger.info(f"Average lines changed per PR: {avg_lines:.1f}")

if __name__ == "__main__":
    main()