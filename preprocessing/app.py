import streamlit as st
import os
from dotenv import load_dotenv



# Load AWS credentials from Streamlit secrets and set environment variables
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ["AWS_DEFAULT_REGION"] = st.secrets["AWS_DEFAULT_REGION"]

# Include AWS_SESSION_TOKEN if using temporary credentials
if "AWS_SESSION_TOKEN" in st.secrets:
    os.environ["AWS_SESSION_TOKEN"] = st.secrets["AWS_SESSION_TOKEN"]
load_dotenv()

# Add the starting timestamp in seconds to the youtube url to get the video at that time
BASE_URL = "https://youtu.be/moHIBWZiYdY?feature=shared&t="

# from boto_testing import titan_multimodal_embedding
from upsert_vectors import titan_text_embedding
from claude_utils import ask_claude_vqa_response
from config import index_name

# Initialize Pinecone
from pinecone import Pinecone

# Load secrets from Streamlit secrets
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

index = pc.Index(index_name)

sample_queries = [
    "What is multilingual search?",
    "How does Pinecone work?",
    "Explain the concept of vector search.",
    "What are the benefits of using AWS Bedrock?",
    "How to integrate Claude with Pinecone?"
]


def convert_to_streamlit_path(path):
    # get list of streamlit path images
    root_path = "./data/frames/mlsearch_webinar"
    # get image filename, which is last part and is always .png
    image_filename = path.split("/")[-1]
    return os.path.join(root_path,image_filename)

# Streamlit app
st.title("Visual QA over Webinar Videos with Pinecone, Claude and AWS")

explanation = '''
Welcome! This app demonstrates how to use Pinecone, Claude, and AWS to perform visual QA over webinar videos.

The app takes an input query, conducts a semantic search over pre-embedded contextual transcript chunks in Pinecone,
submits those results to Claude, and returns an explanation of the results.

'''




st.markdown(explanation)

with st.expander("Application Architecture"):
    diagram_path = "./diagrams"
    diagram_1 = convert_to_streamlit_path("./diagrams/Video_Preprocessing.png")
    diagram_2 = convert_to_streamlit_path("./diagrams/Contextual_Retrieval_With_Video_RAG.png")
    diagram_3 = convert_to_streamlit_path("./diagrams/Pinecone_Upsertion.png")
    diagram_4 = convert_to_streamlit_path("./diagrams/RAG_Workflow_for_Video_Contextual_RAG.png")

    col1, col2 = st.columns(2)
    with col1:
        st.image(diagram_1, caption="Video Preprocessing")
        st.image(diagram_3, caption="Pinecone Upsertion")

    with col2:
        st.image(diagram_2, caption="Contextual Embeddings Process with Claude")
        st.image(diagram_4, caption="RAG Workflow for Video Contextual RAG")



# Input text for query
selected_query = st.selectbox("Select a sample query:", [""] + sample_queries)

query_text = st.text_input("Enter text to query the Pinecone index:", selected_query)



    


if st.button("Query"):
    if query_text:
        # Embed the query
        query_embedding = titan_text_embedding(text=query_text)

        response = index.query(
            vector=query_embedding["embedding"], top_k=5, include_metadata=True
        )

        # st.write(response)
        for r in response["matches"]:
            with st.expander(f"Match with Score: {r['score']}"):

                st.markdown(f"**Score:** {r['score']}")
                st.image(convert_to_streamlit_path(r["metadata"]["filepath"]), caption="Matched Image")
                st.markdown(f"**Transcript:** {r['metadata']['transcript']}")
                st.markdown(
                    f"**Contextual Frame Description:** {r['metadata']['contextual_frame_description']}"
                )

                start_time = r["metadata"]["timestamp_start"]
                end_time = r["metadata"]["timestamp_end"]

                st.markdown(f"**Timestamp Start:** {start_time}")
                st.markdown(f"**Timestamp End:** {end_time}")
                
                
                youtube_url = BASE_URL + str(r["metadata"]["timestamp_start"])


                st.markdown(f"[Watch on YouTube]({youtube_url})")
                st.markdown("Watch the clip:")
                st.video(data=youtube_url, start_time=int(start_time),
                         end_time=int(end_time))

        # ask claude for an explanation of the returned results.

        # replace response["matches"]["metadata"]["filepath"] with transformed filepaths
        final_responses = response["matches"]
        for r in final_responses:
            r["metadata"]["filepath"] = convert_to_streamlit_path(r["metadata"]["filepath"])

        claude_explanation = ask_claude_vqa_response(query_text, response["matches"])
        st.markdown(f"**Claude Explanation:** {claude_explanation}")
    else:
        st.write("Please enter text or image path to query.")
