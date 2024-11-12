# pc-yt-rag
A simplified Contextual Video RAG implementation using Pinecone, AWS, and Claude (SageMaker notebook branch)

Ever wanted to ask questions over your video data, such as Youtube, Zoom webinars, recorded meetings, etc? This application aims to create a RAG chatbot over these content using contextual retrieval and Pinecone, AWS, and Claude.

This branch contains the notebook version of the repository, which was demonstrated during the webinar. To use the code within this branch,
you must have access to SageMaker notebook instances.

## Before you Begin

You'll also need access to AWS Bedrock, Pinecone (via an API Key), and Claude specifically via Bedrock.

Finally, you need to add the videos you'd like to process under a folder called data, with a subfolder called videos. Leave them in .mp4 format. If you have access to your own Youtube channel, downloading videos from the console there will be perfect!


### Using Sagemaker Notebooks

First, ensure you have the appropriate permissions to use Sagemaker, Bedrock, and Bedrock inside Sagemaker.

Then, create a notebook instance with the following configurations:

- a powerful compute instance
- link to this public repo branch, so you can import all scripts (you can also fork this repo and link that instead, in that case you will need to auth your access)
- the lifecycle_configuration.sh script, which will install packages on notebooks startup
- 16gb volume size, in case you add a lot of videos

**It's extremely important to use the lifecycle config script, otherwise you may run into compatibility issues**

When selecting the kernel, use the conda_python3 environment.

Next, upload your data as described above (video mp4 files under ./data/videos), and run the notebook.




