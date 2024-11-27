from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from config import index_name
import os
import json
import boto3
import base64
import pandas as pd
import streamlit as st
from tqdm import tqdm
import toml


load_dotenv()

boto3_session = boto3.session.Session(
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"]
)
region_name = 'us-east-1'


bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=region_name)


# Embedding code


def titan_multimodal_embedding(
    image_path: str = None,  # maximum 2048 x 2048 pixels
    description: str = None,  # English only and max input tokens 128
    dimension: int = 1024,  # 1,024 (default), 384, 256
    model_id: str = "amazon.titan-embed-image-v1",
):

    payload_body = {}
    embedding_config = {"embeddingConfig": {"outputEmbeddingLength": dimension}}

    # You can specify either text or image or both
    if image_path:
        with open(image_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode("utf8")
        payload_body["inputImage"] = input_image
    if description:
        payload_body["inputText"] = description

    assert payload_body, "please provide either an image and/or a text description"
    print("\n".join(payload_body.keys()))

    response = bedrock_client.invoke_model(
        body=json.dumps({**payload_body, **embedding_config}),
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    return json.loads(response.get("body").read())


def titan_text_embedding(
    text: str,  # English only and max input tokens 128
    dimension: int = 1024,  # 1,024 (default), 384, 256
    model_id: str = "amazon.titan-embed-text-v2:0",
):
    payload_body = {
        "inputText": text,
    }

    response = bedrock_client.invoke_model(
        body=json.dumps(payload_body),
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())

    finish_reason = response_body.get("message")

    if finish_reason is not None:
        raise Exception(f"Embeddings generation error: {finish_reason}")

    return response_body


# transform dataframe into metadata, ids, and values (where values are the vectors)



if __name__ == "__main__":
    # read in as json
    file_path = "./data/finalized_data.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    values_to_embed = [item["contextual_frame_description"] for item in data]
    ids = [item["frame_path"] for item in data]

    embeddings = []

    for v in tqdm(values_to_embed):
        embedding = titan_text_embedding(text=v)
        embeddings.append(embedding["embedding"])

    final_vectors = []
    ids = [x for x in range(0, len(values_to_embed))]

    for v, e, id in tqdm(zip(data, embeddings, ids)):
        final_vectors.append(
            {
                "id": str(id),
                "values": e,
                "metadata": {
                    "transcript": v["words"],
                    "filepath": v["frame_path"],
                    "timestamp_start": v["timestamp"][0],
                    "timestamp_end": v["timestamp"][1],
                    "contextual_frame_description": v["contextual_frame_description"],
                },
            }
        )

    # turn the list of dictionaries into a dataframe
    df = pd.DataFrame(final_vectors)
    print(df.head())
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # create index
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)
    index.upsert_from_dataframe(df)
