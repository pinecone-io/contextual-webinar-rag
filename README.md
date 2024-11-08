# pc-yt-rag
A simplified Contextual Video RAG implementation using Pinecone, AWS, and Claude

Ever wanted to ask questions over your video data, such as Youtube, Zoom webinars, recorded meetings, etc? This application aims to create a RAG chatbot over these content using contextual retrieval and Pinecone, AWS, and Claude.

## Before you Begin

This repo presents the RAG solution in two ways: one using scripting and makefiles, to create a Streamlit application, and another using a notebook intended for use on Sagemaker.

You'll also need access to AWS Bedrock, Pinecone (via an API Key), and Claude specifically via Bedrock.

Finally, you need to add the videos you'd like to process under a folder called data, with a subfolder called videos. Leave them in .mp4 format. If you have access to your own Youtube channel, downloading videos from the console there will be perfect!


### Using Sagemaker Notebooks

First, ensure you have the appropriate permissions to use Sagemaker, Bedrock, and Bedrock inside Sagemaker.

Then, create a notebook instance with the following configurations:

- a powerful compute instance, we used ml.p3.2xlarge.
- link to this public repo, so you can import all scripts (you can also fork this repo and link that instead, in that case you will need to auth your access)
- the lifecycle_configuration.sh script, which will install packages on notebooks startup
- 16gb volume size, in case you add a lot of videos

**It's extremely important to use the lifecycle config script, otherwise you may run into compatibility issues**

When selecting the kernel, use the conda_python3 environment.

Next, upload your data as described above (video mp4 files under ./data/videos)

### Running the Scripts Locally

Before beginning, authenthicate your session with AWS using your preferred method. You can
save the access key, default region, and secret access key as environmental variables, or use
'aws sso login' if you have that setup.

**You'll still need access to AWS Bedrock and Claude via Bedrock, as well as a Pinecone API Key**

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

It's easiest to run the whole pipeline (setup) and then run the streamlit app.

From there, the streamlit app should pop up locally and you can start querying!




