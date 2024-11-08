#!/bin/bash

set -e

# OVERVIEW
# This script installs all necessary software for running the DeepComposer GAN notebook.

sudo -u ec2-user -i <<'EOF'
ENVIRONMENT=python3
source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

conda update --all --y 
pip install anthropic==0.39.0
pip install boto3==1.35.46
pip install botocore==1.35.46
pip install ffmpeg_python==0.2.0
pip install pandas==2.2.3
pip install pinecone==5.3.1
pip install python-dotenv==1.0.1
pip install streamlit==1.39.0
pip install transformers==4.45.2
pip install torch==2.5.0
pip install tqdm==4.66.2
source /home/ec2-user/anaconda3/bin/deactivate
EOF