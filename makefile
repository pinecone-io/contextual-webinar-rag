# Makefile

# Variables
DATA_DIR = data
SCRIPTS_DIR = preprocessing
NOTEBOOKS_DIR = notebooks
APP_FILE = app.py
ENV_FILE = .env
TRANSCRIPTIONS_DIR = $(DATA_DIR)/transcriptions
FRAMES_AND_WORDS_DIR = $(DATA_DIR)/frames_and_words
FRAMES_DIR = $(DATA_DIR)/frames

# Targets
.PHONY: all clean preprocess enrich upsert setup run-app create-env help

# Default target
all: setup

# Clean the data folder, removing everything except the videos
clean:
	@echo "Cleaning data folder..."
	find $(DATA_DIR) -type f ! -name '*.mp4' -delete	
	rm -rf $(TRANSCRIPTIONS_DIR) $(FRAMES_AND_WORDS_DIR) $(FRAMES_DIR)
	@echo "Data folder cleaned."

# Preprocess the videos
preprocess:
	@echo "Preprocessing videos..."
	python $(SCRIPTS_DIR)/preprocess_videos.py

# Run the vector enrichment
enrich:
	@echo "Running vector enrichment..."
	python $(SCRIPTS_DIR)/enrich_and_create_vectors.py

# Run the upsertion process
upsert:
	@echo "Running upsertion process..."
	python $(SCRIPTS_DIR)/upsert_vectors.py

# Data setup process: clean, preprocess, enrich, and upsert
setup: clean preprocess enrich upsert

# Run the Streamlit app
run-app:
	@echo "Running the app..."
	streamlit run $(APP_FILE)

# Create the .env file for new users and ask for the Pinecone API key
create-env:
	@echo "Creating .env file..."
	@touch $(ENV_FILE)
	@echo "PINECONE_API_KEY=" >> $(ENV_FILE)
	@echo "AWS_ACCESS_KEY_ID=" >> $(ENV_FILE)
	@echo "AWS_SECRET_ACCESS_KEY=" >> $(ENV_FILE)
	@echo "AWS_DEFAULT_REGION=" >> $(ENV_FILE)
	@echo "Please enter your Pinecone API key:"
	@read api_key; echo "PINECONE_API_KEY=$$api_key" >> $(ENV_FILE)
	@echo ".env file created. Please update it with your AWS credentials."

# Display available commands
help:
	@echo "Available commands:"
	@echo "  make clean         - Clean the data folder, removing everything except the videos"
	@echo "  make preprocess    - Preprocess the videos"
	@echo "  make enrich        - Run the vector enrichment"
	@echo "  make upsert        - Run the upsertion process"
	@echo "  make setup         - Data setup process: clean, preprocess, enrich, and upsert"
	@echo "  make run-app       - Run the Streamlit app"
	@echo "  make create-env    - Create the .env file for new users and ask for the Pinecone API key"
	@echo "  make help          - Display available commands"