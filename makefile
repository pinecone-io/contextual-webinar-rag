# Makefile

# Variables
DATA_DIR = data
SCRIPTS_DIR = preprocessing
NOTEBOOKS_DIR = notebooks
APP_FILE = $(SCRIPTS_DIR)/app.py
ENV_FILE = .env
TRANSCRIPTIONS_DIR = $(DATA_DIR)/transcriptions
FRAMES_AND_WORDS_DIR = $(DATA_DIR)/frames_and_words
FRAMES_DIR = $(DATA_DIR)/frames
CONDA_ENV_NAME = claude-pinecone-vqa  # Replace 'myenv' with your desired environment name

# Targets
.PHONY: all clean preprocess enrich upsert setup run-app create-env create-conda-env install-deps help

# Default target
all: setup

# Clean the data folder, removing everything except the videos
clean:
	@echo "Cleaning data folder..."
	find $(DATA_DIR) -type f ! -name '*.mp4' -delete
	rm -rf $(TRANSCRIPTIONS_DIR) $(FRAMES_AND_WORDS_DIR) $(FRAMES_DIR)
	@echo "Data folder cleaned."


# Create the Conda environment
create-conda-env:
	@echo "Creating Conda environment '$(CONDA_ENV_NAME)'..."
	conda create -n $(CONDA_ENV_NAME) python=3.11.9 -y
	@echo "Conda environment '$(CONDA_ENV_NAME)' created."

# Install dependencies using pip within the Conda environment
install-deps:
	@echo "Installing dependencies in Conda environment '$(CONDA_ENV_NAME)'..."
	conda run -n $(CONDA_ENV_NAME) pip install -r requirements.txt
	@echo "Dependencies installed."

# Preprocess the videos using the Conda environment
preprocess:
	@echo "Preprocessing videos..."
	conda run -n $(CONDA_ENV_NAME) python $(SCRIPTS_DIR)/preprocess_videos.py

# Run the vector enrichment using the Conda environment
enrich:
	@echo "Running vector enrichment..."
	conda run -n $(CONDA_ENV_NAME) python $(SCRIPTS_DIR)/enrich_and_create_vectors.py

# Run the upsertion process using the Conda environment
upsert:
	@echo "Running upsertion process..."
	conda run -n $(CONDA_ENV_NAME) python $(SCRIPTS_DIR)/upsert_vectors.py

# Data setup process: clean, create Conda env, install deps, preprocess, enrich, and upsert
setup: clean create-conda-env install-deps preprocess enrich upsert

# Run the Streamlit app using the Conda environment
run-app:
	@echo "Running the app..."
	conda activate $(CONDA_ENV_NAME)
	streamlit run $(APP_FILE)

# Create the .env file for new users and setup streamlit secrets in case you need it
new-dot-env:
	@echo "Creating .env file..."
	@touch $(ENV_FILE)
	@echo "PINECONE_API_KEY=" >> $(ENV_FILE)
	@echo "AWS_ACCESS_KEY_ID=" >> $(ENV_FILE)
	@echo "AWS_SECRET_ACCESS_KEY=" >> $(ENV_FILE)
	@echo "AWS_DEFAULT_REGION=" >> $(ENV_FILE)
	@echo ".env file created. Please update it with your API keys."
	# Create the secrets.toml file for Streamlit

streamlit-secrets:
	@echo "Creating secrets.toml file..."
	@mkdir -p .streamlit
	@touch .streamlit/secrets.toml
	@echo "PINECONE_API_KEY = \"$(PINECONE_API_KEY)\"" >> .streamlit/secrets.toml
	@echo "AWS_ACCESS_KEY_ID = \"$(AWS_ACCESS_KEY_ID)\"" >> .streamlit/secrets.toml
	@echo "AWS_SECRET_ACCESS_KEY = \"$(AWS_SECRET_ACCESS_KEY)\"" >> .streamlit/secrets.toml
	@echo "AWS_DEFAULT_REGION = \"$(AWS_DEFAULT_REGION)\"" >> .streamlit/secrets.toml
	@echo "secrets.toml file created. Please update it with your API keys if necessary."

# Display available commands
help:
	@echo "Available commands:"
	@echo "  make clean             - Clean the data folder"
	@echo "  make create-conda-env  - Create the Conda environment"
	@echo "  make install-deps      - Install dependencies"
	@echo "  make preprocess        - Preprocess the videos"
	@echo "  make enrich            - Run vector enrichment"
	@echo "  make upsert            - Run upsertion process"
	@echo "  make setup             - Full setup process"
	@echo "  make run-app           - Run the Streamlit app"
	@echo "  make create-env        - Create the .env file"
	@echo "  make help              - Display this help message"