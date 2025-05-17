!/bin/bash

# Django
echo "Starting Django pipeline..."
nohup python run_pipeline.py --org django --repo django --prs 40 --initial-sample 160 > django_pipeline.log 2>&1 &
echo "Django pipeline started. Check django_pipeline.log for progress."

# Wait for 5 seconds before starting the next one to avoid API rate limits
sleep 5

# Express.js
echo "Starting Express.js pipeline..."
nohup python run_pipeline.py --org expressjs --repo express --prs 40 --initial-sample 200 > express_pipeline.log 2>&1 &
echo "Express.js pipeline started. Check express_pipeline.log for progress."

sleep 5

# Scikit-learn
echo "Starting Scikit-learn pipeline..."
nohup python run_pipeline.py --org scikit-learn --repo scikit-learn --prs 40 --initial-sample 160 > scikit_pipeline.log 2>&1 &
echo "Scikit-learn pipeline started. Check scikit_pipeline.log for progress."

sleep 5

# Rust
echo "Starting Rust pipeline..."
nohup python run_pipeline.py --org rust-lang --repo rust --prs 40 --initial-sample 160 > rust_pipeline.log 2>&1 &
echo "Rust pipeline started. Check rust_pipeline.log for progress."

sleep 5

# Pandas
echo "Starting Pandas pipeline..."
nohup python run_pipeline.py --org pandas-dev --repo pandas --prs 40 --initial-sample 160 > pandas_pipeline.log 2>&1 &
echo "Pandas pipeline started. Check pandas_pipeline.log for progress."

echo "All pipelines are running in the background. Check the log files for progress."