# src/test_filtering_pipeline.py
import json
import logging
from pathlib import Path
import sys

from src.dataflow.filtering.bot_filter import BotFilter
from src.dataflow.filtering.size_complexity_filter import SizeComplexityFilter
from src.dataflow.filtering.content_relevance_filter import ContentRelevanceFilter
from src.dataflow.filtering.file_relationship import RelatedFilePredictor
from src.dataflow.filtering.filtering_pipeline import FilterPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_pr_data():
    """Create sample PR data for testing."""
    return {
        "pr_number": 123,
        "title": "Fix bug in data processing logic",
        "body": "This PR fixes a critical bug in the data processing pipeline that was causing crashes.",
        "author": "developer-username",
        "file_count": 3,
        "line_changes": 45,
        "additions": 30,
        "deletions": 15,
        "is_bug_fix": True,
        "code_files": [
            {
                "filename": "src/data_processor.py",
                "patch": "def process_data(data):\n-    result = data.process()\n+    if data is None:\n+        return None\n+    result = data.process()\n    return result"
            },
            {
                "filename": "README.md",
                "patch": "# Data Processor\n\nProcesses data efficiently.\n\n+## Bug Fixes\n+- Fixed null data handling"
            },
            {
                "filename": "tests/test_processor.py",
                "patch": "+def test_null_data():\n+    assert process_data(None) is None"
            }
        ]
    }

def create_bot_pr_data():
    """Create sample bot PR data for testing."""
    return {
        "pr_number": 456,
        "title": "Bump dependency versions",
        "body": "This PR was automatically created to update dependencies",
        "author": "dependabot",
        "file_count": 1,
        "line_changes": 10,
        "additions": 5,
        "deletions": 5,
        "is_bug_fix": False,
        "code_files": [
            {
                "filename": "package.json",
                "patch": "-  \"version\": \"1.0.0\",\n+  \"version\": \"1.0.1\","
            }
        ]
    }

def test_individual_filters():
    """Test each filter individually."""
    logger.info("Testing individual filters...")
    
    # Create test data
    normal_pr = create_sample_pr_data()
    bot_pr = create_bot_pr_data()
    
    # Test Bot Filter
    bot_filter = BotFilter()
    normal_result, normal_meta = bot_filter.apply(normal_pr)
    bot_result, bot_meta = bot_filter.apply(bot_pr)
    
    logger.info(f"Bot Filter - Normal PR: {normal_result}")
    logger.info(f"Bot Filter - Bot PR: {bot_result}")
    
    # Test Size Filter
    size_filter = SizeComplexityFilter()
    size_normal_result, size_normal_meta = size_filter.apply(normal_pr)
    size_bot_result, size_bot_meta = size_filter.apply(bot_pr)
    
    logger.info(f"Size Filter - Normal PR: {size_normal_result}")
    logger.info(f"Size Filter - Bot PR: {size_bot_result}")
    
    # Test Content Filter
    content_filter = ContentRelevanceFilter()
    content_normal_result, content_normal_meta = content_filter.apply(normal_pr)
    content_bot_result, content_bot_meta = content_filter.apply(bot_pr)
    
    logger.info(f"Content Filter - Normal PR: {content_normal_result}")
    logger.info(f"Content Filter - Bot PR: {content_bot_result}")

def test_pipeline():
    """Test the complete filtering pipeline."""
    logger.info("Testing complete filtering pipeline...")
    
    # Create sample data directory structure
    data_dir = Path("./test_data")
    os.makedirs(data_dir, exist_ok=True)
    
    processed_dir = data_dir / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    repo_dir = processed_dir / "example_owner_example_repo"
    os.makedirs(repo_dir, exist_ok=True)
    
    # Create sample PRs
    prs = [create_sample_pr_data(), create_bot_pr_data()]
    
    # Save processed PR index
    with open(repo_dir / "processed_index.json", "w") as f:
        json.dump(prs, f, indent=2)
    
    # Save individual PR data
    for pr in prs:
        pr_dir = repo_dir / f"pr_{pr['pr_number']}"
        os.makedirs(pr_dir, exist_ok=True)
        
        with open(pr_dir / "processed.json", "w") as f:
            json.dump(pr, f, indent=2)
    
    # Initialize and run the filter pipeline
    pipeline = FilterPipeline(data_dir)
    output_dir = pipeline.filter_repository("example_owner", "example_repo")
    
    # Load and print filtered results
    filtered_index_path = data_dir / "filtered" / "example_owner_example_repo" / "filtered_index.json"
    if filtered_index_path.exists():
        with open(filtered_index_path, "r") as f:
            filtered_prs = json.load(f)
        logger.info(f"Filtered PRs: {len(filtered_prs)}")
        for pr in filtered_prs:
            logger.info(f"- PR #{pr['pr_number']}: {pr.get('title', '')}")
    
    logger.info(f"Filter stats: {pipeline.stats}")

if __name__ == "__main__":
    logger.info("Starting filter pipeline tests...")
    
    # Run individual filter tests
    test_individual_filters()
    
    # Run pipeline test
    test_pipeline()
    
    logger.info("Filter pipeline tests completed.")