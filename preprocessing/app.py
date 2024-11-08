import streamlit as st
import os
#from boto_testing import titan_multimodal_embedding
from upsert_vectors import titan_text_embedding
from claude_utils import ask_claude_vqa_response
from config import index_name
# Initialize Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Define the Pinecone index name
index_name = "test-vqa"
index = pc.Index(index_name)

# Streamlit app
st.title("Visual QA over Videos with Pinecone, Claude and AWS")

# Input text for query
query_text = st.text_input("Enter text to query the Pinecone index:")

if st.button("Query"):
    if query_text:
        # Embed the query
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