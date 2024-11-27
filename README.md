# pc-yt-rag
A simplified Contextual Video RAG implementation using Pinecone, AWS, and Claude

Ever wanted to ask questions over your video data, such as Youtube, Zoom webinars, recorded meetings, etc? This application aims to create a RAG chatbot over these content using contextual retrieval and Pinecone, AWS, and Claude.

This branch contains the **Demo Streamlit Web App** version of the implementation. This allows you to run a local web app to interact with the RAG chatbot, and uses a makefile to make the data preprocessing smoother. And, this version already has a sample video in the repo, so you can get started preprocessing and embedding. 

Please read the following section to ensure you have the appropriate prerequisites before proceeding.

If you'd rather work in Sagemaker Notebook, use the webinar-notebook branch above!
If you'd rather make your own video dataset, use the main branch above!
## Before you Begin

This repo presents the RAG solution in two ways: one using scripting and makefiles, to create a Streamlit application, and another using a notebook intended for use on Sagemaker.

You'll also need access to AWS Bedrock, Pinecone (via an API Key), and Claude specifically via Bedrock.

Finally, you need to add the videos you'd like to process under a folder called data, with a subfolder called videos. Leave them in .mp4 format. If you have access to your own Youtube channel, downloading videos from the console there will be perfect!

### Running the Scripts Locally

Before beginning, authenthicate your session with AWS using your preferred method. You can
save the access key, default region, and secret access key as environmental variables, or use
'aws sso login' if you have that setup.

**You'll still need access to AWS Bedrock and Claude via Bedrock, as well as a Pinecone API Key**

**For this branch, after setting up your environmental variables (Pinecone, AWS), you can simply run the create-conda-env, install-deps, and upsert commands for setup**

To run the scripts locally, you can use the provided Makefile. Below are the available commands:
1. **Create the .env file**:
    ```sh
    make create-env
    ```
    This command will create the .env file for new users and prompt you to add your API keys.

2. **Clean the data folder**:
    ```sh
    make clean
    ```
    This command will clean the data folder, removing everything except the videos. Useful for resetting the environment.

3. **Create the Conda environment**:
    ```sh
    make create-conda-env
    ```
    This command will create the Conda environment specified in the Makefile.

4. **Install dependencies**:
    ```sh
    make install-deps
    ```
    This command will install the required dependencies within the Conda environment.

5. **Preprocess the videos**:
    ```sh
    make preprocess
    ```
    This command will preprocess the videos using the specified script.

6. **Run the vector enrichment**:
    ```sh
    make enrich
    ```
    This command will run the Claude Contextual embedding step process.

7. **Run the upsertion process**:
    ```sh
    make upsert
    ```
    This command will run the upsertion process into Pinecone.

8. **Data setup process**:
    ```sh
    make setup
    ```
    This command will clean the data folder, create the Conda environment, install dependencies, preprocess the videos, do the Claude contextual preprocessing step, and upsert the data into Pinecone

## Launching the Streamlit App

To launch the Streamlit app, use the following command:

```sh
make run-app
```

This command will run the Streamlit app defined in `app.py`.

For more information on available commands, you can use:

```sh
make help
```

It's easiest to run the whole pipeline (setup) and then run the Streamlit app.

From there, the Streamlit app should pop up locally and you can start querying!


