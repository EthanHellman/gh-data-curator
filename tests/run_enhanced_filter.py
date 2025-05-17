#!/usr/bin/env python3
"""
Enhanced script to run filtering on existing Django PR data with relevant files prediction.
"""

import json
import logging
import os
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the filtering components
from dataflow.filtering.bot_filter import BotFilter
from dataflow.filtering.size_complexity_filter import SizeComplexityFilter
from dataflow.filtering.content_relevance_filter import ContentRelevanceFilter
from dataflow.filtering.filtering_pipeline import FilterPipeline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run enhanced filtering pipeline with relevant files prediction")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API for relevant files prediction")
    parser.add_argument("--no-import-analysis", action="store_true", help="Disable import analysis for finding related files")
    parser.add_argument("--repo", type=str, default="django", help="Repository name (default: django)")
    parser.add_argument("--org", type=str, default="django", help="Organization name (default: django)")
    args = parser.parse_args()
    
    # Check for OpenAI API key if requested
    if args.openai and not os.environ.get("OPENAI_API_KEY"):
        logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return
    
    # Define paths based on your existing structure
    base_dir = Path("~/gh-data-curator/data").expanduser()
    processed_dir = base_dir / "processed" / f"{args.org}_{args.repo}"
    
    # Ensure output directory exists
    filtered_dir = base_dir / "filtered"
    filtered_dir.mkdir(exist_ok=True)
    
    # Test individual PRs first
    logger.info("Testing individual PRs...")
    
    # Load the processed PR index
    with open(processed_dir / "processed_index.json", "r") as f:
        pr_index = json.load(f)
    
    # Initialize filters
    bot_filter = BotFilter()
    size_filter = SizeComplexityFilter()
    content_filter = ContentRelevanceFilter()
    
    # Stats for individual testing
    stats = {
        "total": len(pr_index),
        "passed_bot": 0,
        "passed_size": 0,
        "passed_content": 0,
        "passed_all": 0
    }
    
    # Test each PR individually
    for pr_summary in pr_index:
        pr_number = pr_summary.get("pr_number")
        pr_path = processed_dir / f"pr_{pr_number}" / "processed.json"
        
        if not pr_path.exists():
            logger.warning(f"Processed file not found for PR #{pr_number}")
            continue
        
        # Load the PR data
        with open(pr_path, "r") as f:
            pr_data = json.load(f)
        
        # Apply each filter
        bot_result, bot_meta = bot_filter.apply(pr_data)
        if bot_result:
            stats["passed_bot"] += 1
            
            size_result, size_meta = size_filter.apply(pr_data)
            if size_result:
                stats["passed_size"] += 1
                
                content_result, content_meta = content_filter.apply(pr_data)
                if content_result:
                    stats["passed_content"] += 1
                    stats["passed_all"] += 1
                    
                    logger.info(f"PR #{pr_number} passed all filters: {pr_data.get('title', '')}")
    
    # Print individual test summary
    logger.info("\n===== INDIVIDUAL PR TEST SUMMARY =====")
    logger.info(f"Total PRs: {stats['total']}")
    logger.info(f"Passed bot filter: {stats['passed_bot']} ({stats['passed_bot']/stats['total']*100:.1f}%)")
    logger.info(f"Passed size filter: {stats['passed_size']} ({stats['passed_size']/stats['total']*100:.1f}%)")
    logger.info(f"Passed content filter: {stats['passed_content']} ({stats['passed_content']/stats['total']*100:.1f}%)")
    logger.info(f"Passed all filters: {stats['passed_all']} ({stats['passed_all']/stats['total']*100:.1f}%)")
    
    # Now run the full pipeline
    logger.info("\n===== RUNNING FULL FILTERING PIPELINE WITH RELEVANT FILES PREDICTION =====")
    
    # Create a FilterPipeline instance with optional OpenAI support
    # Note: This expects a specific directory structure
    pipeline = FilterPipeline(
        base_dir, 
        use_openai=args.openai,
        use_import_analysis=not args.no_import_analysis
    )
    
    # Run the filtering pipeline
    output_dir = pipeline.filter_repository(args.org, args.repo)
    
    if output_dir:
        logger.info(f"Pipeline completed. Output saved to {output_dir}")
        logger.info(f"Pipeline stats: {pipeline.stats}")
        
        # Load the filtered index if it exists
        filtered_index_path = output_dir / "filtered_index.json"
        if filtered_index_path.exists():
            with open(filtered_index_path, "r") as f:
                filtered_prs = json.load(f)
            
            logger.info(f"PRs passing all filters: {len(filtered_prs)}")
            for pr in filtered_prs:
                relevant_files = pr.get("relevant_files", [])
                logger.info(f"- PR #{pr['pr_number']}: {pr.get('title', '')}")
                if relevant_files:
                    logger.info(f"  Relevant files: {relevant_files}")
        else:
            logger.warning(f"Filtered index not found at {filtered_index_path}")
    else:
        logger.error("Filtering pipeline failed")

if __name__ == "__main__":
    main()