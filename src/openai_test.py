# openai_test.py
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataflow.filtering.relevant_files_predictor import RelevantFilesPredictor

def test_openai_prediction():
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    logger.info(f"OpenAI API key found: {bool(api_key)}")
    
    # Simple PR data
    pr_data = {
        "pr_number": 12345,
        "title": "Fix bug in Django's static files storage",
        "body": "This PR fixes an issue with nested URLs in ManifestStaticFilesStorage.",
        "code_files": [
            {"filename": "django/contrib/staticfiles/storage.py"}
        ]
    }
    
    # Create predictor with OpenAI enabled
    logger.info("Creating predictor with OpenAI enabled")
    predictor = RelevantFilesPredictor(
        repo_path=None,  # No repo path needed for OpenAI
        use_openai=True,
        openai_api_key=api_key
    )
    
    # Verify settings
    logger.info(f"Predictor settings: use_openai={predictor.use_openai}, has_api_key={bool(predictor.openai_api_key)}")
    
    # Try prediction
    logger.info("Attempting to predict relevant files using OpenAI")
    try:
        relevant_files = predictor.predict_relevant_files(pr_data)
        logger.info(f"Prediction result: {relevant_files}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    test_openai_prediction()