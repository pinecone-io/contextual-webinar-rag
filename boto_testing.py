
import boto3
import json
import base64
from dotenv import load_dotenv

load_dotenv()
 
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name,
)

def titan_multimodal_embedding(
    image_path:str=None,  # maximum 2048 x 2048 pixels
    description:str=None, # English only and max input tokens 128
    dimension:int=1024,   # 1,024 (default), 384, 256
    model_id:str="amazon.titan-embed-image-v1"
):


    payload_body = {}
    embedding_config = {
        "embeddingConfig": { 
             "outputEmbeddingLength": dimension
         }
    }

    # You can specify either text or image or both
    if image_path:
        with open(image_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode('utf8')
        payload_body["inputImage"] = input_image
    if description:
        payload_body["inputText"] = description

    assert payload_body, "please provide either an image and/or a text description"
    print("\n".join(payload_body.keys()))

    response = bedrock_client.invoke_model(
        body=json.dumps({**payload_body, **embedding_config}), 
        modelId=model_id,
        accept="application/json", 
        contentType="application/json"
    )

    return json.loads(response.get("body").read())




# upsert into Pinecone index

# first, we need to massage the resultant data so it is in the right format for Pinecone
# this is gonna be values, ids, and metadata


def transform_data_into_pc_format(embeddings, frame_caption_pairs):
    ids = range(len(embeddings))
    data_for_upsertion = []
    for e, pair, i in zip(embeddings, frame_caption_pairs, ids):
        d = {
            "values": e["embedding"],
            "id": str(i),
            "metadata": {
                "transcript": pair["words"],
                "filepath": pair["frame_path"],
                "timestamp_start": pair["timestamp"][0],
                "timestamp_end": pair["timestamp"][1]
            }
        }
        data_for_upsertion.append(d)
    return data_for_upsertion


from pinecone import Pinecone

if __name__ == "__main__":
    # read in the dialogue, frame pairs

    import json
    with open("frames_and_words.json", "r") as f:
        frame_caption_pairs = json.load(f)

    # read in the frames from the frame filepaths

    print(frame_caption_pairs[0])

    # compute distribution of char counts for each caption

    char_counts = [len(pair["words"]) for pair in frame_caption_pairs]
    mean_char = sum(char_counts) / len(char_counts)
    max_char =  max(char_counts)
    min_char = min(char_counts)
    print(f"Mean character count: {mean_char}")
    print(f"Max character count: {max_char}")
    print(f"Min character count: {min_char}")

    # embed the first three as a test and return


    embeddings = []

    for pair in frame_caption_pairs:
        embedding = titan_multimodal_embedding(image_path = pair["frame_path"], description=pair["words"])
        embeddings.append(embedding)




    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("bedrock-testing")



    data_for_upsertion = transform_data_into_pc_format(embeddings, frame_caption_pairs)

    index.upsert(data_for_upsertion)
    import time
    time.sleep(5)

    query_text = "show me who the speaker is"
    query_embedding = titan_multimodal_embedding(description=query_text)["embedding"]


    response = index.query(
        vector=query_embedding,
        top_k=2,
        include_metadata=True
    )
        
    print(response)
        



