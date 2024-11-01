import streamlit as st
import os
#from boto_testing import titan_multimodal_embedding
from upsert_vectors import titan_text_embedding
from claude_utils import ask_claude_vqa_response
# Initialize Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Define the Pinecone index name
#index_name = "bedrock-testing"
index_name = "enriched-claude-vqa"
index = pc.Index(index_name)

# Streamlit app
st.title("Visual QA over Videos with Pinecone, Claude and AWS")

# Input text for query
query_text = st.text_input("Enter text to query the Pinecone index:")
# List files in the ./mlsearch_webinar directory
directory_path = "./data/frames/mlsearch_webinar"
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Dropdown for selecting a file
selected_file = st.selectbox("Select a file from the mlsearch_webinar directory:", files)

# Checkbox to ignore image selection
ignore_image = st.checkbox("Ignore image selection")

# Construct the full path of the selected file
image_path = os.path.join(directory_path, selected_file)

# Render the image if a file is selected and checkbox is not checked
if selected_file and not ignore_image:
    st.image(image_path, caption="Query Image")

if st.button("Query"):
    if query_text or (image_path and not ignore_image):
        # Perform the query
        # Embed the query
        #query_embedding = titan_multimodal_embedding(description=query_text, image_path=image_path if not ignore_image else None)

        # just embed with the text-only model
        query_embedding =  titan_text_embedding(text=query_text)

        response = index.query(vector=query_embedding["embedding"], top_k=5, include_metadata=True)

        #st.write(response)
        for r in response["matches"]:
            with st.expander(f"Match with Score: {r['score']}"):
                st.markdown(f"**Score:** {r['score']}")
                st.image(r["metadata"]["filepath"], caption="Matched Image")
                st.markdown(f"**Transcript:** {r['metadata']['transcript']}")
                st.markdown(f"**Contextual Frame Description:** {r['metadata']['contextual_frame_description']}")
                st.markdown(f"**Timestamp Start:** {r['metadata']['timestamp_start']}")
                st.markdown(f"**Timestamp End:** {r['metadata']['timestamp_end']}")


        # ask claude for an explanation of the returned results.

        claude_explanation = ask_claude_vqa_response(query_text, response["matches"])
        st.markdown(f"**Claude Explanation:** {claude_explanation}")
    else:
        st.write("Please enter text or image path to query.")


# good queries
# The interview spoke about mad libs for robots. What is that about?
# At some point, the interviewer spoke about marrying customs. What was that about?