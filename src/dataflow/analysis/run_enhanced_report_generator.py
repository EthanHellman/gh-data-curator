#!/usr/bin/env python3
"""
Run the enhanced report generator to create a comprehensive analysis report
for the data curation pipeline results.
"""
import sys
from pathlib import Path
import argparse
import logging
import os
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("__main__")

def main():
    """Run the enhanced report generator with the modular architecture."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate enhanced filtering report")
    parser.add_argument("--data-dir", type=str, default="~/gh-data-curator/data", 
                      help="Base data directory")
    parser.add_argument("--view", action="store_true", 
                      help="Open the report after generation")
    parser.add_argument("--skip-analysis", action="store_true",
                      help="Skip running the analysis script first")
    args = parser.parse_args()
    
    # Expand user directory
    data_dir = Path(args.data_dir).expanduser()
    
    # Check if the data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Run the analysis script to ensure we have the latest results
    if not args.skip_analysis:
        logger.info("Running filter results analysis first...")
        try:
            # First check if the analyze_filter_results.py script exists
            analysis_script = data_dir.parent / "src" / "dataflow" / "analysis" / "analyze_filter_results.py"
            
            if not analysis_script.exists():
                # Try to find the script in current directory or in a subdirectory
                possible_paths = [
                    Path("analyze_filter_results.py"),
                    Path("dataflow/analysis/analyze_filter_results.py"),
                    Path("src/dataflow/analysis/analyze_filter_results.py")
                ]
                
                for path in possible_paths:
                    if path.exists():
                        analysis_script = path
                        break
                else:
                    logger.warning("Could not find analyze_filter_results.py script. Skipping analysis.")
                    analysis_script = None
            
            if analysis_script:
                logger.info(f"Found analysis script at {analysis_script}")
                subprocess.run([sys.executable, str(analysis_script), "--data-dir", str(data_dir), "--all", "--profiles"],
                              check=True)
                logger.info("Analysis completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running analysis script: {e}")
            logger.warning("Continuing with report generation using existing analysis results")
    
    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent  # This will be ~/gh-data-curator
    sys.path.insert(0, str(project_root))
    
    # Import the core generator
    from dataflow.analysis.generation.core_generator import EnhancedReportGenerator
    
    # Initialize enhanced report generator
    logger.info("Initializing enhanced report generator...")
    generator = EnhancedReportGenerator(data_dir)
    
    # Generate the report
    logger.info("Generating enhanced report...")
    report_path = generator.generate_report()
    
    logger.info(f"Report generated successfully: {report_path}")
    
    # Open the report if requested
    if args.view and report_path.exists():
        logger.info(f"Opening report: {report_path}")
        try:
            if os.name == 'posix':  # Unix/Linux/MacOS
                subprocess.run(['xdg-open', str(report_path)], check=False)
            elif os.name == 'nt':  # Windows
                os.startfile(str(report_path))
            else:
                logger.warning("Automatic report opening not supported on this platform")
        except Exception as e:
            logger.warning(f"Could not open report: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())